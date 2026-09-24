"""GPU map parity checks; run on the execution host, not at module import.

    python validate_map.py --device cuda
    python validate_map.py --device cuda --context-json context.json --xy-npy xy.npy

Real callers may import ``validate_context(xy, context, device='cuda',
features=actual_device_features, reference_bad=direction_guard._bad_uncached)``.
The JSON form contains ``offset`` and ``lanes`` entries with ``polygon_wkt``,
``vertices`` and ``tangents``. All host copies/reference queries are test-only.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import shapely
import torch
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon, box

from device_map import DirectionCache, PolygonCache, RoadCache


def reference_direction(xy, context):
    """Frozen direction_guard._bad_uncached semantics, including KDTree ties."""
    xy = np.asarray(xy)
    delta = np.gradient(xy, axis=1)
    norm = np.linalg.norm(delta, axis=-1)
    direction = delta / np.maximum(norm[..., None], 1e-8)
    center = xy + context[1] * direction
    flat, unit = center.reshape(-1, 2), direction.reshape(-1, 2)
    forward, reverse = np.zeros(len(flat), bool), np.zeros(len(flat), bool)
    for polygon, index, tangents in context[0]:
        ids = np.flatnonzero(shapely.intersects_xy(polygon, flat[:, 0], flat[:, 1]))
        if not len(ids):
            continue
        _, k = index.query(flat[ids])
        dot = (unit[ids] * tangents[k]).sum(1)
        forward[ids] |= dot >= 0
        reverse[ids] |= dot < -.1
    return ((reverse & ~forward).reshape(norm.shape) & (norm > .005)).any(1)


def direction_features(xy, offset, device):
    tensor = torch.as_tensor(xy, device=device)
    delta = torch.gradient(tensor, dim=1)[0]
    norm = torch.linalg.vector_norm(delta, dim=-1)
    unit = delta / norm.clamp_min(1e-8)[..., None]
    return dict(direction_centers=tensor + offset * unit,
                direction_unit=unit, direction_norm=norm)


def validate_context(xy, context, *, device="cuda", features=None,
                     reference_bad=reference_direction, query_chunk=31, vertex_chunk=3):
    """Fail on unflagged parity differences; report only genuine unresolved ties.

    A nonzero tie count requires the adapter's sparse exact-reference refinement;
    it is not counted as proof that an arbitrary GPU nearest-index choice agrees.
    """
    xy = np.asarray(xy)
    features = direction_features(xy, context[1], device) if features is None else features
    result = DirectionCache(context, device=device, query_chunk=query_chunk,
                            vertex_chunk=vertex_chunk, edge_chunk=3).evaluate(features)
    expected = np.asarray(reference_bad(xy, context), dtype=bool)
    actual = result["bad"].cpu().numpy()
    ambiguous = result["tie_ambiguous"].cpu().numpy()
    invalid = result["invalid"].cpu().numpy()
    mismatches = np.flatnonzero((actual != expected) & ~ambiguous)
    if len(mismatches):
        raise AssertionError(f"Unflagged direction parity mismatch: rows {mismatches[:20].tolist()}")
    corrected = actual.copy()
    corrected[ambiguous] = expected[ambiguous]
    np.testing.assert_array_equal(corrected, expected)
    return dict(candidates=len(xy), unambiguous_mismatches=len(mismatches),
                tie_candidates=np.flatnonzero(ambiguous).tolist(),
                tie_reference_bad=expected[ambiguous].tolist(), invalid_count=int(invalid.sum()),
                reference_rejected=int(expected.sum()))


def polygon_cases(device):
    shell = [(-4., -3.), (4., -3.), (4., 3.), (-4., 3.)]
    hole = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]
    geometries = [Polygon(shell), Polygon(shell, [hole]),
                  MultiPolygon([Polygon(shell, [hole]), box(8., -2., 12., 2.)]),
                  Polygon([(0., 0.), (5., 0.), (5., 2.), (2., 2.), (2., 5.), (0., 5.)]),
                  Polygon([(0., 0.), (8., 8.), (0., 8.)]),
                  Polygon(shell[::-1], [hole[::-1]]), Polygon()]
    rng = np.random.default_rng(200)
    random_points = rng.uniform([-6., -5.], [14., 10.], (2001, 2))
    reports = []
    for geometry in geometries:
        boundary = []
        parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        for part in parts:
            if part.is_empty:
                continue
            for ring in [part.exterior, *part.interiors]:
                a = np.asarray(ring.coords)[:-1]
                boundary.extend(a)
                boundary.extend((a + np.roll(a, -1, axis=0)) / 2.)
        boundary = np.asarray(boundary, dtype=float).reshape(-1, 2)
        if len(boundary):
            # One representable step on either side, with vertices included.
            left, right = boundary.copy(), boundary.copy()
            left[:, 0] = np.nextafter(left[:, 0], -np.inf)
            right[:, 0] = np.nextafter(right[:, 0], np.inf)
            points = np.concatenate((random_points, boundary, left, right))
        else:
            points = random_points
        cache = PolygonCache(geometry, device=device, query_chunk=29, edge_chunk=2)
        for dtype in (np.float32, np.float64):
            sample = points.astype(dtype)
            expected = shapely.intersects_xy(geometry, sample[:, 0], sample[:, 1])
            actual = cache.covers(torch.as_tensor(sample, device=device)).cpu().numpy()
            np.testing.assert_array_equal(actual, expected)
        empty = cache.covers(torch.empty((0, 4, 2), device=device))
        assert empty.shape == (0, 4)
        invalid = cache.query(torch.tensor([[np.nan, 0.], [0., np.inf]], device=device))
        assert invalid["invalid"].all().cpu().item()
        assert not invalid["covered"].any().cpu().item()
        reports.append(dict(geometry=geometry.geom_type, empty=geometry.is_empty,
                            points_per_dtype=len(points)))
    try:
        PolygonCache(Polygon([(0, 0), (2, 2), (0, 2), (2, 0)]), device=device)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid topology was accepted")
    return reports


def road_cases(device):
    ego = SimpleNamespace(x=17.25, y=-4.5, heading=.37)
    local_region = Polygon([(-10, -5), (10, -5), (10, 5), (-10, 5)],
                           [[(-1, -1), (1, -1), (1, 1), (-1, 1)]])
    c, s = np.cos(ego.heading), np.sin(ego.heading)
    from shapely.affinity import affine_transform
    world_region = affine_transform(local_region, [c, -s, s, c, ego.x, ego.y])
    rng = np.random.default_rng(91)
    xy = rng.uniform([-12, -6], [12, 6], (13, 7, 2)).astype(np.float32)
    yaw = rng.uniform(-np.pi, np.pi, (13, 7)).astype(np.float32)
    xy[0], xy[1], xy[2], xy[3] = [-5., 2.5], [0., 0.], [9., 0.], [-5., -2.5]
    yaw[:4] = 0.
    forward = np.stack((np.cos(yaw), np.sin(yaw)), -1)
    side = np.stack((-np.sin(yaw), np.cos(yaw)), -1)
    centers = xy + 1.461 * forward
    corners = np.stack([centers + l * 2.3 * forward + w * .95 * side
                        for l, w in ((1, 1), (1, -1), (-1, -1), (-1, 1))], 2)
    world = corners @ np.array([[c, s], [-s, c]]) + [ego.x, ego.y]
    expected = ~shapely.intersects_xy(world_region, world[..., 0], world[..., 1]).all((1, 2))
    assert expected.any() and (~expected).any(), "Road fixture must exercise both outcomes"
    cache = RoadCache(world_region, ego, half_length=2.3, half_width=.95,
                      device=device, query_chunk=17, edge_chunk=2)
    features = {k: torch.as_tensor(v, device=device) for k, v in
                dict(box_centers=centers, forward=forward, side=side).items()}
    np.testing.assert_array_equal(cache.outside(features).cpu().numpy(), expected)
    features["box_centers"] = features["box_centers"].clone()
    features["box_centers"][0, 0, 0] = float("nan")
    result = cache.evaluate(features)
    assert result["invalid"][0].cpu().item() and result["outside"][0].cpu().item()
    return dict(candidates=len(xy), parity=True, nonfinite_rejected=True)


def _entry(polygon, vertices, tangents=None):
    vertices = np.asarray(vertices, dtype=float)
    if tangents is None:
        tangents = np.gradient(vertices, axis=0)
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-8)
    return polygon, cKDTree(vertices), np.asarray(tangents, dtype=float)


def direction_cases(device):
    road = box(-10., -4., 10., 4.)
    forward = _entry(road, [[-9, 0], [-3, 0], [3, 0], [9, 0]])
    reverse = _entry(road, [[9, 0], [3, 0], [-3, 0], [-9, 0]])
    t = np.arange(15, dtype=float)
    xy = np.stack((np.column_stack((-.7 + .1 * t, np.zeros_like(t))),
                   np.column_stack((.7 - .1 * t, np.zeros_like(t))),
                   np.zeros((15, 2)),
                   np.column_stack((.7 - .004 * t, np.zeros_like(t))),
                   np.column_stack((.7 - .006 * t, np.zeros_like(t)))))
    result = []
    for dtype in (np.float32, np.float64):
        for label, lanes in (("forward", [forward]), ("reverse", [reverse]),
                             ("overlap_forward_exemption", [forward, reverse]),
                             ("no_lanes", [])):
            report = validate_context(xy.astype(dtype), (lanes, 1.461), device=device,
                                      query_chunk=7, vertex_chunk=1)
            result.append(dict(case=label, dtype=str(dtype), **report))
    # Conflicting equidistant nearest vertices deliberately straddle chunks.
    tied = _entry(road, [[0., 1.], [100., 100.], [0., -1.]],
                  [[1., 0.], [1., 0.], [-1., 0.]])
    cache = DirectionCache(([tied], 0.), device=device, query_chunk=2, vertex_chunk=2)
    centers = torch.zeros((1, 3, 2), device=device, dtype=torch.float64)
    unit = torch.tensor([[[1., 0.]] * 3], device=device, dtype=torch.float64)
    norm = torch.ones((1, 3), device=device, dtype=torch.float64)
    ambiguity = cache.evaluate_tensors(centers, unit, norm)
    assert ambiguity["tie_ambiguous"][0].cpu().item(), "Cross-chunk tie was lost"
    # A separately certain forward lane resolves even a conflicting tie.
    resolved = DirectionCache(([tied, forward], 0.), device=device, vertex_chunk=2)
    out = resolved.evaluate_tensors(centers, unit, norm)
    assert not out["bad"].any().cpu().item()
    assert not out["tie_ambiguous"].any().cpu().item()
    centers[0, 0, 0] = float("nan")
    invalid = resolved.evaluate_tensors(centers, unit, norm)
    assert invalid["invalid"][0].cpu().item() and invalid["bad"][0].cpu().item()
    result.append(dict(case="cross_chunk_tie_and_forward_resolution", passed=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-json", type=Path)
    parser.add_argument("--xy-npy", type=Path)
    args = parser.parse_args()
    if bool(args.context_json) != bool(args.xy_npy):
        parser.error("--context-json and --xy-npy must be supplied together")
    report = dict(device=args.device, polygons=polygon_cases(args.device),
                  road=road_cases(args.device), directions=direction_cases(args.device))
    if args.context_json:
        data = json.loads(args.context_json.read_text())
        context = ([_entry(shapely.from_wkt(lane["polygon_wkt"]), lane["vertices"],
                           lane["tangents"]) for lane in data["lanes"]], data["offset"])
        report["real_context"] = validate_context(np.load(args.xy_npy, allow_pickle=False),
                                                  context, device=args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
