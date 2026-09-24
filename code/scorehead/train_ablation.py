"""Ablation variants for the anchor score head. Based on v3 (corridor) architecture.

Architectures:
  A: shared 1D-conv candidate encoder (replaces per-anchor MLP; parameter sharing)
  B: FiLM corridor conditioning (replaces additive injection)
  C: no candidate self-attention block
  D: no positional embedding (pos_in removed)
  base: v3 unchanged (for re-reference)
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

DATA = Path("/tmp/score_head_data")


class Block(nn.Module):
    def __init__(self, h, heads=4, ffn=4 * 256):
        super().__init__()
        self.ln_q = nn.LayerNorm(h)
        self.ln_kv = nn.LayerNorm(h)
        self.attn = nn.MultiheadAttention(h, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(h)
        self.ffn = nn.Sequential(nn.Linear(h, ffn), nn.GELU(), nn.Linear(ffn, h))

    def forward(self, q, kv, kv_mask=None):
        qn = self.ln_q(q)
        attn_out, _ = self.attn(qn, self.ln_kv(kv), self.ln_kv(kv),
                                key_padding_mask=kv_mask, need_weights=False)
        q = q + attn_out
        q = q + self.ffn(self.ln2(q))
        return q


class Head(nn.Module):
    def __init__(self, vocab, arch, context_dim=256, h=256, depth=2, heads=4, corridor_dim=50):
        super().__init__()
        self.arch = arch
        self.register_buffer('anchors', torch.as_tensor(vocab, dtype=torch.float32))
        if arch in ('A', 'FA', 'FAG', 'H'):
            # shared 1D-conv along waypoints: [M,40,2] -> [M,H]
            self.cand_conv = nn.Sequential(
                nn.Conv1d(2, h // 2, 5, padding=2), nn.GELU(),
                nn.Conv1d(h // 2, h, 5, padding=2), nn.GELU(),
                nn.Conv1d(h, h, 3, padding=1))
        else:
            self.cand_in = nn.Sequential(nn.Linear(80, h), nn.GELU(), nn.Linear(h, h))
        if arch == 'D':
            pass
        else:
            self.pos_in = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, h))
        self.scene_in = nn.Linear(context_dim, h)
        if arch == 'B':
            # FiLM: corridor -> gamma, beta
            self.film = nn.Sequential(nn.Linear(corridor_dim, h), nn.GELU(), nn.Linear(h, 2 * h))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        else:
            self.corridor_in = nn.Sequential(nn.Linear(corridor_dim, h), nn.GELU(), nn.Linear(h, h))
        self.blocks = nn.ModuleList([Block(h, heads) for _ in range(depth)])
        if arch != 'C':
            self.self_block = Block(h, heads)
        self.ln_out = nn.LayerNorm(h)
        if arch == 'H':
            # MultiPath-style aux heads: per-anchor [dx,dy] offset + K neighbor
            # mixing logits, appended to the classification logit.
            self.k_mix = 8
            self.out = nn.Linear(h, 1 + 2 + self.k_mix)
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        else:
            self.score = nn.Linear(h, 1)
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)

    def forward(self, context, valid, corridor=None, cand=None):
        b = context.shape[0]
        cand = self.anchors if cand is None else cand
        m = cand.shape[0]
        dev = context.device
        if self.arch in ('A', 'FA', 'FAG', 'H'):
            xx = cand.float().transpose(1, 2)  # [M,2,40]
            ct = self.cand_conv(xx)  # [M,H,40]
            ct = ct.mean(-1)  # [M,H] global pool over waypoints
        else:
            ct = self.cand_in(cand.flatten(1).float())
        if self.arch != 'D':
            pos = self.pos_in(torch.linspace(0, 1, 40, device=dev).unsqueeze(-1).repeat(1, 2).float())
            ct = ct + pos.mean(0)[None]
        ct = ct[None].expand(b, -1, -1)
        st = self.scene_in(context)
        if corridor is not None:
            if self.arch == 'B':
                gb = self.film(corridor.float())  # [B,2H]
                gamma, beta = gb[:, :ct.shape[-1]], gb[:, ct.shape[-1]:]
                ct = ct * (1 + gamma[:, None, :]) + beta[:, None, :]
            else:
                kt = self.corridor_in(corridor.float())[:, None, :].expand(-1, m, -1)
                ct = ct + kt
        invalid = ~valid
        for blk in self.blocks:
            ct = blk(ct, st, kv_mask=invalid)
        if self.arch != 'C':
            ct = self.self_block(ct, ct)
        if self.arch == 'H':
            o = self.out(self.ln_out(ct))  # [B,M,1+2+K]
            return dict(logits=o[..., 0], offset=o[..., 1:3], mix_logits=o[..., 3:])
        return self.score(self.ln_out(ct)).squeeze(-1)


def corridor_feature(row):
    c = row['corridor']
    v = np.asarray(c['polygon'], dtype=np.float32)
    e = np.asarray(c['enterable'], dtype=np.float32)
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
    return rows, gt_all, vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True, choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'FG', 'FA', 'FAG', 'H', 'base'])
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--tau-scale', type=float, default=0.25)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    out = Path('/tmp/score_head_abl_%s' % a.arch); out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    torch.manual_seed(617)

    rows, gt_all, vocab = load_data()
    train_rows, eval_rows = rows['train'], rows['eval']
    print(f'arch {a.arch} | train {len(train_rows)} eval {len(eval_rows)}')

    # E: metric normalization scales
    anchor_scale = float(np.sqrt((vocab ** 2).mean()))  # global RMS ~ magnitude of coords
    if a.arch == 'E':
        vocab_in = vocab / anchor_scale
    else:
        vocab_in = vocab

    model = Head(vocab_in, a.arch).to(dev)
    print('params(M): %.2f' % (sum(x.numel() for x in model.parameters()) / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps_per = math.ceil(len(train_rows) / a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * steps_per)

    def tensors(rs):
        ctx = torch.cat([r['context'] for r in rs]).float()
        val = torch.cat([r['valid'] for r in rs]).bool()
        cor = torch.as_tensor(np.stack([corridor_feature(r) for r in rs]))
        if a.arch == 'E':
            # normalize corridor: coordinates /40m, flags unchanged
            cor = cor.clone()
            cor[:, 1:33] = cor[:, 1:33] / 40.0
        gt = torch.stack([torch.as_tensor(np.asarray(gt_all[r['key']]), dtype=torch.float32) for r in rs])
        return ctx.to(dev), val.to(dev), cor.to(dev), gt.to(dev)

    ctx_tr, val_tr, cor_tr, gt_tr = tensors(train_rows)
    ctx_ev, val_ev, cor_ev, gt_ev = tensors(eval_rows)

    def labels_for(gt):
        with torch.no_grad():
            anchors_m = model.anchors * (anchor_scale if a.arch == 'E' else 1.0)
            if a.arch in ('G', 'FG', 'FAG', 'H'):
                # endpoint-weighted distance: 0.5 uniform RMS + 0.5 endpoint
                per_pt = (anchors_m[None] - gt[:, None]).square().sum(-1).mean(-1)  # [B,M]
                end_pt = (anchors_m[None, :, -1] - gt[:, None, -1]).square().sum(-1)  # [B,M]
                d = torch.sqrt(0.5 * per_pt + 0.5 * end_pt)
            else:
                d = (anchors_m[None] - gt[:, None]).square().sum(-1).mean(-1).sqrt()
            # F/FA/FG: near-neighbor tau (5th NN distance) instead of global median
            if a.arch in ('F', 'FA', 'FG', 'FAG', 'H'):
                tau = d.topk(5, largest=False, sorted=True).values[:, -1:]
            else:
                tau = d.median(dim=1, keepdim=True).values * a.tau_scale
            return F.softmax(-d / tau.clamp_min(1e-6), dim=1)

    best = None
    history = []

    def forward_arch(ctx, val, cor):
        o = model(ctx, val, cor)
        return (o['logits'], o) if a.arch == 'H' else (o, None)

    for epoch in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_rows))
        tot = 0.; nb = 0
        for i in range(0, len(train_rows), a.batch):
            idx = perm[i:i + a.batch]
            gt = gt_tr[idx]
            logits, aux = forward_arch(ctx_tr[idx], val_tr[idx], cor_tr[idx])
            loss = -(labels_for(gt) * F.log_softmax(logits, dim=1)).sum(1).mean()
            if aux is not None:
                # H aux losses: offset regression + neighbor mixing, masked to the
                # label head (top-K near neighbors) where GT semantics exist.
                with torch.no_grad():
                    d = (model.anchors[None] - gt[:, None]).square().sum(-1).mean(-1).sqrt()
                    nn_idx = d.topk(model.k_mix, largest=False, sorted=False).indices  # [B,K]
                    nn_w = F.softmax(-d.gather(1, nn_idx) / d.topk(5, largest=False).values[:, -1:].clamp_min(1e-6), dim=1)
                off = aux['offset']  # [B,M,2]
                mix = aux['mix_logits']  # [B,M,K]
                b, m = off.shape[0], off.shape[1]
                # per-neighbor offset target: mean waypoint translation GT - anchor_j
                # anchors[nn_idx]: [B,K,40,2]; mean over waypoints -> [B,K,2]
                per_off = (gt[:, None, :, :] - model.anchors[nn_idx]).mean(2)  # [B,K,2]
                off_pred = off.gather(1, nn_idx[:, :, None].expand(-1, -1, 2).clamp(0, m - 1))  # [B,K,2]
                off_loss = F.smooth_l1_loss(off_pred, per_off, reduction='none').mean((1, 2))
                # mix target: near-neighbor soft weights
                mix_pred = mix.gather(1, nn_idx[:, :, None].expand(-1, -1, model.k_mix).clamp(0, m - 1))  # [B,K,K]
                mix_loss = -(nn_w * F.log_softmax(mix_pred[:, :, 0], dim=1)).sum(1)
                loss = loss + 0.3 * off_loss.mean() + 0.1 * mix_loss.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1

        model.eval()
        with torch.no_grad():
            d1_list = []; dc_list = []
            for i in range(0, len(eval_rows), 64):
                gt = gt_ev[i:i + 64]
                logits, aux = forward_arch(ctx_ev[i:i + 64], val_ev[i:i + 64], cor_ev[i:i + 64])
                top1 = logits.argmax(1)
                d1_list.append((model.anchors[top1] - gt).square().sum(-1).mean(-1).sqrt())
                if aux is not None:
                    # corrected error: top-1 anchor + predicted offset
                    off = aux['offset'][torch.arange(len(gt), device=gt.device), top1]  # [B,2]
                    corrected = model.anchors[top1] + off[:, None, :]
                    dc_list.append((corrected - gt).square().sum(-1).mean(-1).sqrt())
            d1 = torch.cat(d1_list)
            d_all = (model.anchors[None] - gt_ev[:, None]).square().sum(-1).mean(-1).sqrt()
            oracle = d_all.min(1).values
        row = dict(epoch=epoch, train_loss=tot / max(nb, 1),
                   dev_top1_gt_rms=float(d1.mean()),
                   top1_hits_oracle=float((d1 - oracle < 1e-4).float().mean()))
        if dc_list:
            row['dev_top1_corrected_rms'] = float(torch.cat(dc_list).mean())
        history.append(row)
        print(json.dumps(row), flush=True)
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        if best is None or row['dev_top1_gt_rms'] < best['dev_top1_gt_rms']:
            best = row
            torch.save(dict(model=model.state_dict(), epoch=epoch, row=row, arch=a.arch), out / 'best.pt')
    (out / 'final_summary.json').write_text(json.dumps(dict(best=best), indent=2))
    print('BEST', json.dumps(best))


if __name__ == '__main__':
    main()
