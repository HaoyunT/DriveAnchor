"""Local encoder warmup and strict stage-transition checks.

The reconstruction head is training-only: production V6 weights are unavailable
and this cache lacks the all-agent labels used by the upstream auxiliary task.
"""
import torch

class EncoderHead(torch.nn.Module):
    def __init__(self, dimension=256, points=40):
        super().__init__()
        self.query=torch.nn.Parameter(torch.randn(1,1,dimension)*.02)
        self.attention=torch.nn.MultiheadAttention(dimension,2,batch_first=True)
        self.norm=torch.nn.LayerNorm(dimension)
        self.output=torch.nn.Sequential(torch.nn.Linear(dimension,dimension),torch.nn.GELU(),torch.nn.Linear(dimension,points*2))
    def forward(self,context,valid):
        q=self.query.expand(context.shape[0],-1,-1)
        z=self.attention(q,context,context,key_padding_mask=~valid,need_weights=False)[0]
        return self.output(self.norm(z+q)).reshape(-1,40,2)


def validate_transition(stage, checkpoint):
    previous=checkpoint.get('stage',checkpoint.get('manifest',{}).get('args',{}).get('stage')) if checkpoint else None
    if stage=='fm' and previous not in ('encoder','fm'):
        raise ValueError('Stage FM requires an encoder-pretrained or FM checkpoint; use stage encoder first')
    if stage=='rl' and previous not in ('fm','ef','ef_only','rl'):
        raise ValueError('Stage RL requires an FM-trained checkpoint, not random/encoder-only weights')
    if stage in ('ef','ef_only') and previous not in ('fm','ef','ef_only'):
        raise ValueError('EF requires an FM-trained checkpoint')
    if stage=='encoder' and checkpoint and previous!='encoder':
        raise ValueError('Encoder warmup can only continue an encoder-stage checkpoint')


def configure_stage(model,stage):
    model.requires_grad_(False)
    if stage not in ('encoder','fm','ef','ef_only','rl'):raise ValueError('Unknown stage '+stage)
    if stage!='ef_only':(model.encoder if stage=='encoder' else model.denoiser).requires_grad_(True)
    if stage in ('ef','ef_only'):
        if model.ef_head is None:raise ValueError('EF head missing')
        model.ef_head.requires_grad_(True)


def encoder_loss(model,head,scene):
    context,valid=model.encode(scene[0],scene[1])
    prediction=head(context,valid)
    target=model.normalize(scene[2].reshape(1,40,2))
    flow=(prediction-target).square().sum(-1).mean()
    waypoint=(model.denormalize(prediction)-scene[2].reshape(1,40,2)).square().sum(-1).mean()
    return flow+.01*waypoint,{'encoder_delta_loss':float(flow.detach()),'waypoint_loss':float(waypoint.detach())}


@torch.no_grad()
def evaluate_encoder(model,head,cache,keys):
    model.eval();head.eval();rows=[]
    for key in keys:
        scene=cache.scene(key);context,valid=model.encode(scene[0],scene[1])
        prediction=model.denormalize(head(context,valid))
        distance=(prediction-scene[2].reshape(1,40,2)).norm(dim=-1)
        rows.append({'scene':key,'encoder_ADE40':float(distance.mean()),'encoder_FDE40':float(distance[:,-1].mean())})
    return {'scenes':len(rows),'summary':{k:sum(r[k] for r in rows)/len(rows) for k in rows[0] if k!='scene'},'per_scene':rows}
