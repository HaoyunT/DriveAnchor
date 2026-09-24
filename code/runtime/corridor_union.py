"""Precision-normalized union; caller supplies topology-approved polygons.

Millimetre quantization addresses sub-nanometre seams, not missing connectors.
Never use a large positive buffer to force disconnected roads together.
"""
import shapely
from shapely.ops import unary_union


def road_union(polygons, grid_size=0.001):
    if not 0 < grid_size <= 0.001:
        raise ValueError('Only sub-millimetre/millimetre precision repair allowed')
    polygons = list(polygons)
    if any(not p.is_valid for p in polygons):
        raise ValueError('Invalid map polygon requires explicit diagnosis')
    return unary_union([shapely.set_precision(p, grid_size) for p in polygons])
