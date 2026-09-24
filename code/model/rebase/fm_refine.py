"""Versioned, time-free clean-state refinement objective; distinct from legacy weighting."""
import torch

@torch.no_grad()
def fit_scales(model,gt,k=50):
    model.fit_scale(gt)
    # A nearly straight subset must not amplify lateral centimetres by 1e6.
    model.delta_scale.clamp_(min=.01)
    sums=gt.new_zeros(2);count=0
    for batch in gt.split(128):
        ids=torch.cdist(batch.flatten(1),model.anchors.flatten(1)).topk(min(k,len(model.anchors)),largest=False).indices
        difference=batch[:,None]-model.anchors[ids]
        sums+=difference[:,:,1:].square().sum((0,1,2));count+=difference.shape[0]*difference.shape[1]*(difference.shape[2]-1)
    model.refinement_scale.copy_((sums/count).sqrt().clamp_min(.1))

def sample_states(source,target):
    n=len(source);choice=torch.rand(n,device=source.device)
    t=torch.rand(n,device=source.device)
    t=torch.where(choice<.5,torch.zeros_like(t),torch.where(choice>.75,torch.ones_like(t),t))
    return (1-t[:,None,None])*source+t[:,None,None]*target,t

def errors(model,prediction,target):
    delta=(prediction-target)[:,1:].square().mean((1,2))
    xy=model.denormalize(prediction);gt=model.denormalize(target)
    waypoint=((xy-gt)/model.refinement_scale)[:,1:].square().mean((1,2))
    velocity=torch.diff(xy,dim=1)/.1;acceleration=torch.diff(velocity,dim=1)/.1
    dynamics=(torch.relu(velocity.norm(dim=-1)/25.-1).square().mean(1)+torch.relu(acceleration.norm(dim=-1)/4.5-1).square().mean(1))
    return delta+waypoint+.1*dynamics,delta,waypoint,dynamics

def loss(model,context,valid,gt,k=50,unroll=True):
    gt=gt.reshape(1,40,2);table=model.normalize(model.anchors)
    ids=(model.anchors-gt).flatten(1).norm(dim=-1).topk(min(k,len(table)),largest=False).indices
    target=model.normalize(gt).expand(len(ids),-1,-1);state,t=sample_states(table[ids],target)
    available=torch.ones(len(table),device=table.device,dtype=torch.bool);available[ids]=False
    far_ids=torch.nonzero(available).flatten();far_ids=far_ids[torch.randperm(len(far_ids),device=table.device)[:8]]
    far=table[far_ids];state=torch.cat([state,far]);target=torch.cat([target,far])
    predicted=model.raw(state,context,valid)
    def combined(p):
        e,d,w,dyn=errors(model,p,target)
        objective=e[:len(ids)].mean()+(.1*e[len(ids):].mean() if len(far) else 0)
        return objective,d,w,dyn
    total,d,w,dyn=combined(predicted)
    second=total.new_zeros(())
    if unroll:
        p2=model.raw(predicted.detach(),context,valid)
        second=combined(p2)[0];total=total+.5*second
    return total,{'normalized_delta_mse':float(d[:len(ids)].mean().detach()),'normalized_waypoint_mse':float(w[:len(ids)].mean().detach()),'dynamics_penalty':float(dyn.mean().detach()),'second_step_loss':float(second.detach()),'source_endpoint_fraction':float((t==0).float().mean()),'identity_endpoint_fraction':float((t==1).float().mean())}
