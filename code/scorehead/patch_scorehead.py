"""Patch fm2_planner.py: add FAG score-head prefilter (usage A: top-K by learned score)."""
import re
from pathlib import Path

P = Path('/tmp/driveanchor_selector_v8_full_runtime/fm2_planner.py')
src = P.read_text()

# 1) add imports + head loader after the vnorm_topk import
anchor_import = 'from pdm_model_selector import vnorm_topk\n'
assert src.count(anchor_import) == 1
head_loader = '''from pdm_model_selector import vnorm_topk
try:
    from score_head_bridge import get_score_head, score_head_topk
except ImportError:
    get_score_head = None
'''
src = src.replace(anchor_import, head_loader)

# 2) replace the kept selection
old_sel = "        kept=torch.arange(3000,device=norm.device) if topk==3000 else vnorm_topk(norm,topk)\n"
assert src.count(old_sel) == 1
new_sel = '''        prefilter_mode=os.environ.get('DRIVEANCHOR_PREFILTER','vnorm')
        if prefilter_mode=='scorehead' and get_score_head is not None:
            kept=score_head_topk(context,valid,corridor,topk)
        else:
            kept=torch.arange(3000,device=norm.device) if topk==3000 else vnorm_topk(norm,topk)
'''
src = src.replace(old_sel, new_sel)
P.write_text(src)
print('patched fm2_planner prefilter')
