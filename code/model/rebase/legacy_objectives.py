"""Adapt two existing server baselines to deterministic new-model outputs.

Source: /workspace/driveanchor_nuplan/train.py, audited SHA in
review/server_baseline_inventory.json. These are group-normalized finite
 differences and safe-output regression, NOT likelihood GRPO or DPO.
"""
import torch

METHODS=('group_fd','safe_repair')

def group_fd_loss(x,rewards):
    """x=[groups,members,D]; choose maximum reward-std group; repair unsafe only."""
    with torch.no_grad():
        std=rewards.std(-1);index=std.argmax()
        r=rewards[index];adv=(r-r.mean())/std[index].clamp_min(1e-4)
        a=x[index].detach();diff=a[:,None]-a[None,:]
        gradient=((adv[:,None]-adv[None,:])[...,None]*diff/diff.square().sum(-1,keepdim=True).clamp_min(1e-6)).sum(1)/(len(a)-1)
        unsafe=r<41
    if std[index]<1e-4 or not unsafe.any():return x.sum()*0
    return -(gradient[unsafe]*x[index][unsafe]).sum(-1).mean()

def safe_repair_loss(x,rewards,tau=300.,severity=False,min_unsafe=4):
    """Regress unsafe outputs toward detached nearest safe outputs; no density ratio."""
    safe=rewards>=41
    if not safe.any() or int((~safe).sum())<min_unsafe:return x.sum()*0
    loser=x[~safe];pool=x[safe].detach()
    with torch.no_grad():
        distance=(loser.detach()[:,None]-pool[None,:]).square().mean(-1)
        nearest=distance.argmin(-1);target=pool[nearest]
        weight=(tau/(distance.gather(1,nearest[:,None]).squeeze(1)+1e-6)).clamp(max=1)
        if severity:weight=weight*(1+(41-rewards[~safe])/41)
    return (weight*(loser-target).square().mean(-1)).mean()

def select_indices(anchors,method,queries,k,generator):
    """At most Q(K+1) predictions scored; replaces the legacy full-vocabulary scan."""
    n=len(anchors);device=anchors.device
    if method=='safe_repair':
        return torch.randperm(n,device=device,generator=generator)[:min(n,queries*(k+1))]
    seeds=torch.randint(n,(queries,),device=device,generator=generator)
    distances=torch.cdist(anchors[seeds],anchors)
    distances.scatter_(1,seeds[:,None],float('inf')) # Fix legacy duplicated seed neighbor.
    neighbors=distances.topk(k,largest=False).indices
    return torch.cat([seeds[:,None],neighbors],1).flatten()
