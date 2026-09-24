"""Bridge: load the FAG score head into the closed-loop planner and score anchors.

The head was trained on the same frozen v15 encoder context (225x256) and the
same 3000-anchor vocabulary; corridor feature is BranchCorridor.feature (50-d).
"""
import numpy as np
import torch

_CKPT = '/tmp/score_head_abl_FAG/best.pt'
_MODEL = None


def get_score_head():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('abl', '/tmp/train_ablation.py')
    abl = importlib.util.module_from_spec(spec)
    sys.modules['abl'] = abl
    # load_data only used for vocab here; import Head directly needs vocab arg
    # read vocab from checkpoint's anchors buffer via Head construction below
    spec.loader.exec_module(abl)
    ck = torch.load(_CKPT, map_location='cpu', weights_only=False)
    vocab = ck['model']['anchors'].cpu().numpy()
    model = abl.Head(vocab, 'FAG')
    model.load_state_dict(ck['model'])
    model.eval()
    _MODEL = model
    return model


@torch.no_grad()
def score_head_topk(context, valid, corridor, k):
    """Return top-k anchor indices by score head, on the same device as context."""
    model = get_score_head()
    dev = context.device
    m = model.to(dev)
    # corridor is a BranchCorridor in closed loop; feature is [50]
    feat = np.asarray(corridor.feature, dtype=np.float32)
    cor = torch.as_tensor(feat, device=dev)[None]
    ctx = context.float()
    val = valid.bool()
    if ctx.shape[0] != 1:
        raise ValueError('score_head_topk expects single-scene context')
    logits = m(ctx, val, cor)[0]
    return logits.topk(int(k)).indices
