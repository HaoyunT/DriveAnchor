"""Reuse map parity QA against the independent fused CUDA implementation.

    python fused_proposal_r3/validate_triton.py --device cuda
    python validate_triton.py --base-dir /path/to/v6_code --device cuda

Optional real context: --context-json CONTEXT --xy-npy XY. The format matches
validate_map.py. Python callers may import validate_context_fused and pass the
existing context plus actual CUDA features/reference_bad directly.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def _load(base_dir=None):
    base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    import device_map_triton as fused
    spec = importlib.util.spec_from_file_location("_fused_map_qa", base_dir / "validate_map.py")
    qa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qa)
    qa.PolygonCache, qa.RoadCache, qa.DirectionCache = (
        fused.PolygonCache, fused.RoadCache, fused.DirectionCache)
    return fused, qa


def validate_context_fused(xy, context, *, base_dir=None, device="cuda", features=None,
                           reference_bad=None):
    """Inject an existing real context/feature set without executing at import."""
    _, qa = _load(base_dir)
    kwargs = dict(device=device, features=features)
    if reference_bad is not None:
        kwargs["reference_bad"] = reference_bad
    return qa.validate_context(xy, context, **kwargs)


def _kernel_regressions(fused, qa, device):
    """Exercise >64 edges/vertices, padding, ties and float64 -.1 threshold."""
    import numpy as np
    import torch
    from shapely.geometry import Polygon, box
    from device_map import DirectionCache as ReferenceDirection
    theta = np.arange(133) * (2 * np.pi / 133)
    ring = np.stack((10 * np.cos(theta), 10 * np.sin(theta)), -1)
    geometry = Polygon(ring)
    rng = np.random.default_rng(82)
    points = np.concatenate((rng.uniform(-12, 12, (2053, 2)), ring))
    expected = qa.shapely.intersects_xy(geometry, points[:, 0], points[:, 1])
    query = torch.as_tensor(points, device=device, dtype=torch.float64)
    got = fused.PolygonCache(geometry, device=device).covers(query).cpu().numpy()
    np.testing.assert_array_equal(got, expected)

    vertices = np.column_stack((np.arange(130, dtype=float) + 100., np.full(130, 100.)))
    tangents = np.tile([1., 0.], (130, 1))
    vertices[0], vertices[129] = [0., 1.], [0., -1.]
    tangents[129] = [-1., 0.]
    context = ([qa._entry(box(-5, -5, 5, 5), vertices, tangents)], 0.)
    centers = torch.zeros((1, 3, 2), device=device, dtype=torch.float64)
    unit = torch.tensor([[[1., 0.]] * 3], device=device, dtype=torch.float64)
    norm = torch.ones((1, 3), device=device, dtype=torch.float64)
    result = fused.DirectionCache(context, device=device).evaluate_tensors(centers, unit, norm)
    assert result["tie_ambiguous"][0].cpu().item(), "Tie across Triton vertex tiles disappeared"

    # Do not silently promote the float32 encoding of -.1 to float64.
    negative = [-.1, np.nextafter(-.1, -np.inf), np.nextafter(-.1, np.inf)]
    tangent = np.array([[v, 0.] for v in negative], order="F")
    vertices = np.array([[0., 0.], [2., 0.], [4., 0.]])
    context = ([qa._entry(box(-10, -10, 10, 10), vertices, tangent)], 0.)
    centers = torch.as_tensor(vertices[:, None, :].repeat(3, axis=1), device=device)
    unit = torch.tensor([[[1., 0.]] * 3] * 3, device=device, dtype=torch.float64)
    norm = torch.ones((3, 3), device=device, dtype=torch.float64)
    fused_result = fused.DirectionCache(context, device=device).evaluate_tensors(centers, unit, norm)
    reference = ReferenceDirection(context, device=device).evaluate_tensors(centers, unit, norm)
    for key in ("bad", "tie_ambiguous", "tie_points", "invalid"):
        np.testing.assert_array_equal(fused_result[key].cpu().numpy(), reference[key].cpu().numpy())
    np.testing.assert_array_equal(fused_result["bad"].cpu().numpy(), [False, True, False])

    # Empty candidate batches must not launch a zero-sized GPU grid.
    empty = fused.DirectionCache(context, device=device).evaluate_tensors(
        centers[:0], unit[:0], norm[:0])
    assert empty["bad"].shape == (0,)
    return dict(ring_edges=133, vertex_count=130, cross_tile_tie=True,
                float64_threshold=True, noncontiguous_tangents=True, empty_batch=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-json", type=Path)
    parser.add_argument("--xy-npy", type=Path)
    args = parser.parse_args()
    if bool(args.context_json) != bool(args.xy_npy):
        parser.error("--context-json and --xy-npy must be provided together")
    fused, qa = _load(args.base_dir)
    import torch
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available() or fused.triton is None:
            raise RuntimeError("CUDA QA requires an available GPU and Triton")
    started = time.perf_counter()
    report = dict(device=args.device, torch_version=torch.__version__,
                  triton_version=None if fused.triton is None else getattr(fused.triton, "__version__", "unknown"),
                  arithmetic="libdevice_explicit_rn_f64", launch_fp_fusion_flag=False, polygons=qa.polygon_cases(args.device),
                  road=qa.road_cases(args.device), directions=qa.direction_cases(args.device),
                  kernel_regressions=_kernel_regressions(fused, qa, args.device))
    if args.context_json:
        data = json.loads(args.context_json.read_text())
        context = ([qa._entry(qa.shapely.from_wkt(lane["polygon_wkt"]), lane["vertices"],
                             lane["tangents"]) for lane in data["lanes"]], data["offset"])
        xy = qa.np.load(args.xy_npy, allow_pickle=False)
        report["real_context"] = qa.validate_context(xy, context, device=args.device)
    report["qa_wall_seconds_including_compile"] = time.perf_counter() - started
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
