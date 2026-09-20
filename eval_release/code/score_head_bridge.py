"""Bridge: load the FAG score head into the closed-loop planner and score anchors.

The head was trained on the same frozen v15 encoder context (225x256) and the
same 3000-anchor vocabulary; corridor feature is BranchCorridor.feature (50-d).
"""
import numpy as np
import torch

_CKPT = '/tmp/fag_repro/fag_best.pt'
_MODEL = None


def get_score_head():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('abl', '/tmp/driveanchor_selector_v8_full_runtime/train_ablation.py')
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
def score_head_scores(context, valid, corridor):
    """Return the head's score for every anchor: [3000], on context's device.

    The progress rerank needs the whole vector, not a truncated top-k: its
    ``log_softmax`` normaliser is over the full vocabulary, so a pre-sliced
    input would renormalise against a different denominator and change the
    scale of the term it is added to.
    """
    model = get_score_head()
    dev = context.device
    m = model.to(dev)
    # corridor is a BranchCorridor in closed loop; feature is [50]
    feat = np.asarray(corridor.feature, dtype=np.float32)
    cor = torch.as_tensor(feat, device=dev)[None]
    ctx = context.float()
    val = valid.bool()
    if ctx.shape[0] != 1:
        raise ValueError('score head expects single-scene context')
    return m(ctx, val, cor)[0]


@torch.no_grad()
def score_head_topk(context, valid, corridor, k):
    """Return top-k anchor indices by score head, on the same device as context."""
    return score_head_scores(context, valid, corridor).topk(int(k)).indices
