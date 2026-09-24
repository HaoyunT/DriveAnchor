"""Bind the unchanged feasible_quality_v1 implementation by source hashes."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

RULES_SHA = '60fffe5b17e902caa61d53d83d72ac1add7de88c53bcdee79e113311b9b6f235'
PLANNER_SHA = '78b036bd670e6013f7b458484244e53b580f2dd8030a3d2e082fc7439d021991'
CACHED_SHA = 'a606608c55794ca32f9a245f20674ecd91f2d4f9dcf5675b1b3caedf9bb2dd33'
NATIVE_GEOMETRY_SHA = '0604abe32afb486f5e17cffa824c2450d0a60abddb1a2ee8cc7b85c1145ee650'


def checked_file(path, expected):
    path=Path(path)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:raise ValueError('Frozen source changed: '+str(path))
    return path


def load_rules(path):
    path=checked_file(path,RULES_SHA)
    name='full_pool_frozen_selection_rules'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def load_offline_choose(selector_root):
    """Compile the actual local()/choose() AST, using real SciPy/Shapely.

    This avoids importing torch/nuPlan for offline scoring. Neither function's
    body, thresholds nor geometry primitives is rewritten. Enum attributes are
    string sentinels only for the minimal serialized current-map API.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from shapely.geometry import Point,Polygon
    from shapely.ops import unary_union
    from shapely.vectorized import contains,touches
    from types import SimpleNamespace
    root=Path(selector_root);rules=load_rules(root/'selection.py')
    path=checked_file(root/'driveanchor_planner.py',PLANNER_SHA)
    tree=ast.parse(path.read_text())
    local=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='local')
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='DriveAnchorPlanner')
    choose=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='choose')
    namespace=dict(np=np,cKDTree=cKDTree,Point=Point,Polygon=Polygon,unary_union=unary_union,
        contains=contains,touches=touches,RULES=rules.RULES,
        constrained_choice=rules.constrained_choice,stable_heading=rules.stable_heading,
        TrafficLightStatusType=SimpleNamespace(RED='RED'),
        SemanticMapLayer=SimpleNamespace(LANE_CONNECTOR='LANE_CONNECTOR'))
    exec(compile(ast.Module(body=[local,choose],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['choose'],rules
