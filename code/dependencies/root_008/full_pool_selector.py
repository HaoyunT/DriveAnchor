"""Full clean-vocabulary selection; no residual prefilter or learned score."""
import copy
import numpy as np

INFERENCE_VERSION='all500_ef_fm_feasible_quality_v1'


def select_full_pool(chooser,xy,route,tracks,ego,speed,lights,lanes,stage='FM2'):
    xy=np.asarray(xy)
    if xy.shape!=(500,40,2) or not np.isfinite(xy).all():
        raise ValueError('Expected every finite candidate in original [500,40,2] anchor order')
    if stage not in ('FM1','FM2'):raise ValueError('Only FM1/FM2 output stages are supported')
    # The original choose() sees all500 in their original order. Its tie and
    # no-feasible fallback semantics remain unchanged.
    index=int(chooser.choose(xy,route,tracks,ego,speed,lights,lanes))
    diagnostic=copy.deepcopy(chooser.selection_diagnostic)
    if diagnostic['candidate_count']!=500 or not 0<=index<500:
        raise ValueError('Selector did not score the full candidate pool')
    diagnostic.update(inference_version=INFERENCE_VERSION,prefilter='none',
        output_stage=stage,selected_anchor_id=index,
        score_source='unchanged feasible_quality_v1 analytic Q',
        polygon_exit_constraint_in_selector=False)
    return index,diagnostic


def generate_full_pool(model,corridor,context,valid,fm_steps=2,chunk_size=64):
    """Actual EF once and one/two FM refinements; reuse one encoded context.

    Chunking is numerical batching only. It never selects/drops candidates.
    The normalized-state recurrence matches the common joint evaluator.
    """
    import torch
    if fm_steps not in (1,2) or not 1<=chunk_size<=500:
        raise ValueError('Expected FM1/FM2 and a batch size between1 and500')
    if len(model.anchors)!=500 or model.ef_head is None:
        raise ValueError('Full500 vocabulary and trained EF head required')
    if any(module.training for module in model.modules()):
        raise ValueError('Runtime inference requires eval mode for every module')
    with torch.no_grad():
        anchors=model.anchors
        feature=torch.as_tensor(corridor.feature,device=anchors.device,dtype=anchors.dtype)
        ef=anchors+model.ef_head(anchors,feature,context,valid)
        paths={'anchor':anchors,'EF':ef}
        state=model.normalize(ef)
        for step in range(1,fm_steps+1):
            state=torch.cat([model.raw(part,context,valid) for part in state.split(chunk_size)])
            paths['FM'+str(step)]=model.denormalize(state)
        if any(value.shape!=(500,40,2) or not torch.isfinite(value).all() for value in paths.values()):
            raise ValueError('Nonfinite/incomplete full-pool model output')
        return paths
