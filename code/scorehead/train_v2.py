"""Anchor score head v2: per-candidate tokens + scene cross-attention.

Same data/labels/loss/metrics as v1; only the architecture changes from
mean-pool twin towers to candidate-token transformer blocks attending over
scene tokens. Baseline to beat: v1 dev_top1_gt_rms = 4.02m (oracle 0.194).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).resolve().parents[2] / 'server_export_20260916' / 'batch2'


class Block(nn.Module):
    """Pre-LN block on query tokens attending key/value tokens."""

    def __init__(self, h, heads=4, ffn=4 * 256):
        super().__init__()
        self.ln_q = nn.LayerNorm(h)
        self.ln_kv = nn.LayerNorm(h)
        self.attn = nn.MultiheadAttention(h, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, ffn), nn.GELU(), nn.Linear(ffn, h))

    def forward(self, q, kv, kv_mask=None):
        # q [B,N,H], kv [B,S,H], kv_mask True = invalid
        qn = self.ln_q(q)
        attn_out, _ = self.attn(qn, self.ln_kv(kv), self.ln_kv(kv),
                                key_padding_mask=kv_mask, need_weights=False)
        q = q + attn_out
        q = q + self.ffn(self.ln2(q))
        return q


class AnchorScoreHeadV2(nn.Module):
    def __init__(self, vocab, context_dim=256, h=256, depth=2, heads=4, corridor_dim=50):
        super().__init__()
        self.register_buffer('anchors', torch.as_tensor(vocab, dtype=torch.float32))
        self.cand_in = nn.Sequential(nn.Linear(80, h), nn.GELU(), nn.Linear(h, h))
        self.scene_in = nn.Linear(context_dim, h)
        self.corridor_in = nn.Sequential(nn.Linear(corridor_dim, h), nn.GELU(), nn.Linear(h, h))
        self.pos_in = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, h))
        self.blocks = nn.ModuleList([Block(h, heads) for _ in range(depth)])
        self.self_block = Block(h, heads)
        self.ln_out = nn.LayerNorm(h)
        self.score = nn.Linear(h, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, context, valid, corridor=None, cand=None):
        # context [B,225,256], valid [B,225]; corridor [B,51] raw EF feature
        b = context.shape[0]
        cand = self.anchors if cand is None else cand
        m = cand.shape[0]
        ct = self.cand_in(cand.flatten(1).float())  # [M,H]
        dev = context.device
        pos = self.pos_in(torch.linspace(0, 1, 40, device=dev).unsqueeze(-1).repeat(1, 2).float())  # [40,H]
        ct = ct[None] + pos.mean(0)[None]  # [1,M,H] (mild positional cue)
        ct = ct.expand(b, -1, -1)
        st = self.scene_in(context)  # [B,225,H]
        if corridor is not None:
            kt = self.corridor_in(corridor.float())[:, None, :].expand(-1, m, -1)  # [B,M,H]
            ct = ct + kt
        invalid = ~valid
        for blk in self.blocks:
            ct = blk(ct, st, kv_mask=invalid)
        ct = self.self_block(ct, ct)  # candidate self-attention
        return self.score(self.ln_out(ct)).squeeze(-1)  # [B,M]


def corridor_feature(row):
    # BranchCorridor.feature: [scene_type, 16x(x,y,enterable), expanded] = 51 dims
    c = row['corridor']
    v = np.asarray(c['polygon'], dtype=np.float32)  # [16,2]
    e = np.asarray(c['enterable'], dtype=np.float32)  # [16]
    return np.concatenate([[c['scene_type']],
                           np.column_stack([v, e[:, None]]).ravel(),
                           [c['is_expanded']]]).astype(np.float32)


def load_data():
    rec = torch.load(DATA / 'records_turn2048_straight512.pt', map_location='cpu', weights_only=False)
    gt_all = torch.load(DATA / 'all_gt.pt', map_location='cpu', weights_only=False)['gt']
    vocab = np.load(DATA / 'anchors.npy')
    rows = {'train': [], 'eval': []}
    for split in ('train', 'eval'):
        for r in rec[split]:
            if r['key'] in gt_all:
                rows[split].append(r)
    return rows, gt_all, rec, vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--tau-scale', type=float, default=0.25)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default=str(Path(__file__).parent / 'run_v2'))
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    torch.manual_seed(617)

    keys_map, gt_all, rec, vocab = load_data()
    train_rows, eval_rows = keys_map['train'], keys_map['eval']
    train_keys = [r['key'] for r in train_rows]
    eval_keys = [r['key'] for r in eval_rows]
    print(f'train {len(train_keys)} eval {len(eval_keys)} vocab {vocab.shape}')

    model = AnchorScoreHeadV2(vocab, depth=a.depth).to(dev)
    print('params(M):', sum(x.numel() for x in model.parameters()) / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps_per = math.ceil(len(train_keys) / a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * steps_per)

    def tensors(rows):
        ctx = torch.cat([r['context'] for r in rows]).float()
        val = torch.cat([r['valid'] for r in rows]).bool()
        cor = torch.as_tensor(np.stack([corridor_feature(r) for r in rows]))
        gt = torch.stack([torch.as_tensor(np.asarray(gt_all[r['key']]), dtype=torch.float32) for r in rows])
        return ctx.to(dev), val.to(dev), cor.to(dev), gt.to(dev)

    ctx_tr, val_tr, cor_tr, gt_tr = tensors(train_rows)
    ctx_ev, val_ev, cor_ev, gt_ev = tensors(eval_rows)

    def labels_for(gt):
        with torch.no_grad():
            d = (model.anchors[None] - gt[:, None]).square().sum(-1).mean(-1).sqrt()
            tau = d.median(dim=1, keepdim=True).values * a.tau_scale
            return F.softmax(-d / tau.clamp_min(1e-6), dim=1), d

    best = None
    history = []
    for epoch in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_keys))
        tot = 0.; nb = 0
        for i in range(0, len(train_keys), a.batch):
            idx = perm[i:i + a.batch]
            logits = model(ctx_tr[idx], val_tr[idx], cor_tr[idx])
            labels, _ = labels_for(gt_tr[idx])
            loss = -(labels * F.log_softmax(logits, dim=1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1

        model.eval()
        with torch.no_grad():
            d1_list = []
            for i in range(0, len(eval_keys), a.batch):
                logits = model(ctx_ev[i:i + a.batch], val_ev[i:i + a.batch], cor_ev[i:i + a.batch])
                top1 = logits.argmax(1)
                d = (model.anchors[top1] - gt_ev[i:i + a.batch]).square().sum(-1).mean(-1).sqrt()
                d1_list.append(d)
            d1 = torch.cat(d1_list)
            d_all = (model.anchors[None] - gt_ev[:, None]).square().sum(-1).mean(-1).sqrt()
            oracle = d_all.min(1).values
        row = dict(epoch=epoch, train_loss=tot / max(nb, 1),
                   dev_top1_gt_rms=float(d1.mean()),
                   dev_oracle_gt_rms=float(oracle.mean()),
                   top1_hits_oracle=float((d1 - oracle < 1e-4).float().mean()))
        history.append(row)
        print(json.dumps(row), flush=True)
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        if best is None or row['dev_top1_gt_rms'] < best['dev_top1_gt_rms']:
            best = row
            torch.save(dict(model=model.state_dict(), epoch=epoch, row=row,
                           baseline_v1=4.018), out / 'best.pt')
    (out / 'final_summary.json').write_text(json.dumps(dict(best=best), indent=2))
    print('BEST', json.dumps(best))


if __name__ == '__main__':
    main()
