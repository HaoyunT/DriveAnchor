"""Small exact-source parity suite; --device cuda is run only by the GPU owner.

CPU synthetic validation requires only NumPy. A real context factory returns
(planner, histories), where histories is a list of native planner histories.
"""
import argparse
import ast
from collections import OrderedDict
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

try:
    from .batched_features import FeatureBuilder, SHAPES, WorldCache, reference_boundary_flips
except ImportError:
    from batched_features import FeatureBuilder, SHAPES, WorldCache, reference_boundary_flips


def array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compare(reference, actual, label, atol=0.0):
    rf, rm, rl, rt = reference
    af, am, al, at = actual
    assert [x.id for x in rl] == [x.id for x in al], (label, "lane ordering")
    assert [x.track_token for x in rt] == [x.track_token for x in at], (label, "track ordering")
    assert np.array_equal(array(rm), array(am)), (label, "mask")
    report = {}
    for name, shape in SHAPES.items():
        r, a = array(rf[name]), array(af[name])
        assert r.shape == a.shape == shape, (label, name, "shape")
        assert r.dtype == a.dtype == np.float32, (label, name, "dtype")
        assert np.isfinite(r).all() and np.isfinite(a).all(), (label, name, "nonfinite")
        error = float(np.max(np.abs(r.astype(np.float64) - a.astype(np.float64))))
        differing = int(np.count_nonzero(r != a))
        report[name] = {"max_abs": error, "differing_values": differing}
        assert error <= atol, (label, name, report[name], "atol", atol)
    return report


def validate_context(planner, histories, *, reference_features=None, backend="device", atol=0.0,
                     flip_backend="device", artifacts=None):
    """Calls each backend independently; synchronizes only in this validation utility."""
    reference_features = reference_features or getattr(type(planner), "_planner_features_v1_original", type(planner).features)
    builder = FeatureBuilder(reference_features.__globals__, planner.device, backend, flip_backend)
    reports = []
    previous = None
    for frame, history in enumerate(histories):
        reference = reference_features(planner, history)
        actual = builder(planner, history)
        if artifacts is not None:
            destination = Path(artifacts)
            destination.mkdir(parents=True, exist_ok=True)
            raw = {"reference_" + k: array(v) for k, v in reference[0].items()}
            raw.update({"actual_" + k: array(v) for k, v in actual[0].items()})
            raw.update(reference_mask=array(reference[1]), actual_mask=array(actual[1]),
                       lane_ids=np.array([str(lane.id) for lane in actual[2]]),
                       actual_flips=array(builder.last_flips))
            np.savez_compressed(destination / f"frame_{frame:03d}.npz", **raw)
        report = compare(reference, actual, f"frame_{frame}", atol)
        ego = list(history.ego_states)[-1].rear_axle
        expected_flips = reference_boundary_flips(builder.cache.lane_world, ego, reference_features.__globals__["local"])
        assert np.array_equal(expected_flips, array(builder.last_flips)), (frame, "boundary flip difference")
        if previous is not None:
            previous_output, previous_snapshot = previous
            for name, value in previous_output.items():
                assert np.array_equal(array(value), previous_snapshot[name]), (frame, "prior-frame storage changed", name)
        previous = actual[0], {k: array(v).copy() for k, v in actual[0].items()}
        reports.append({"frame": frame, "features": report, "flips_different": 0,
                        "lane_ids_different": 0, "mask_different": 0,
                        "world_uploads_cumulative": builder.gpu.uploaded_geometries if builder.gpu else 0})
    return reports


def benchmark_context(planner, history, *, repeats=10, warmup=3, flip_backend="device"):
    """Complete features() wall time; explicit synchronization belongs only to QA."""
    original = getattr(type(planner), "_planner_features_v1_original", type(planner).features)
    torch = original.__globals__["torch"]
    synchronize = torch.cuda.synchronize if str(planner.device).startswith("cuda") else lambda: None
    result = {}
    for name, fn in [("reference", lambda: original(planner, history))] + [
        (backend, (lambda builder: lambda: builder(planner, history))(
            FeatureBuilder(original.__globals__, planner.device, backend, flip_backend)))
        for backend in ("numpy", "device")
    ]:
        values = []
        for i in range(warmup + repeats):
            synchronize()
            start = time.perf_counter()
            output = fn()
            synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if i >= warmup:
                values.append(elapsed)
        result[name] = {"p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
                        "samples_ms": values}
    return result


class NumpyTensor(np.ndarray):
    def to(self, device):
        assert str(device) == "cpu"
        return self


class NumpyTorch:
    @staticmethod
    def from_numpy(value):
        return value.view(NumpyTensor)


class Line:
    def __init__(self, points):
        self.points = np.asarray(points, np.float64).reshape(-1, 2)
        self.is_empty = len(self.points) == 0
        self.length = float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    @property
    def wkb(self):
        return self.points.tobytes()

    def interpolate(self, fraction, normalized=True):
        assert normalized and not self.is_empty
        lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        distance = fraction * self.length
        cumulative = np.r_[0., np.cumsum(lengths)]
        x = np.interp(distance, cumulative, self.points[:, 0])
        y = np.interp(distance, cumulative, self.points[:, 1])
        return NS(coords=[(x, y)])


class Polygon:
    def __init__(self, center, scale=2.):
        self.center, self.scale = np.asarray(center, np.float64), scale
        self.exterior = Line(self.center + np.array([[-scale, -scale], [scale, -scale], [scale, scale], [-scale, scale], [-scale, -scale]]))

    @property
    def wkb(self):
        return self.exterior.wkb

    def distance(self, point):
        d = np.maximum(np.abs(np.array([point.x, point.y]) - self.center) - self.scale, 0.)
        return float(np.linalg.norm(d))

    def simplify(self, tolerance, preserve_topology):
        assert tolerance == .1 and preserve_topology
        return self


LAYERS = NS(**{k: k for k in ("LANE", "LANE_CONNECTOR", "INTERSECTION", "CROSSWALK", "STOP_LINE", "WALKWAYS", "CARPARK_AREA")})


def load_reference(path, torch):
    tree = ast.parse(Path(path).read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "IndexedFeaturesPlanner")
    def local(points, ego):
        points = np.asarray(points, dtype=float)
        c, s = np.cos(ego.heading), np.sin(ego.heading)
        return (points - np.array([ego.x, ego.y])) @ np.array([[c, -s], [s, c]])

    class Base:
        def sample_line_cached(self, line, n):
            if not hasattr(self, "samples"):
                self.samples = {}
                self.sample_count = 0
            key = (id(self.map), line.wkb, n)
            if key not in self.samples:
                self.samples[key] = np.array([line.interpolate(t, normalized=True).coords[0][:2] for t in np.linspace(0, 1, n)])
                self.sample_count += 1
            return self.samples[key]

    symbols = dict(np=np, torch=torch, OrderedDict=OrderedDict, CachedGeometryPlanner=Base,
                   local=local, SemanticMapLayer=LAYERS, Point=lambda x, y: NS(x=x, y=y), Point2D=lambda x, y: NS(x=x, y=y))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), symbols)
    return symbols["IndexedFeaturesPlanner"]


def make_map(n_lanes, n_polygons):
    origin = np.array([600000., 4500000.])
    near = {key: [] for key in vars(LAYERS)}
    # Equal polygon distances exercise stable layer-concatenation sorting.
    for i in range(n_lanes):
        center = origin + [i % 11 - 5, i // 11 - 3]
        baseline = center + np.array([[0., 0.], [2., .17], [5., .4], [9., -.2]])
        left, right = baseline + [0., 1.7], baseline - [0., 2.3]
        if i % 2:
            left = left[::-1]
        if i % 3:
            right = right[::-1]
        lane = NS(id=f"lane_{i}", polygon=Polygon(center), baseline_path=NS(linestring=Line(baseline)),
                  left_boundary=NS(linestring=Line(left)), right_boundary=NS(linestring=Line(right)))
        near["LANE" if i % 2 else "LANE_CONNECTOR"].append(lane)
    for points in ([], [[600000., 4500000.], [600000.001, 4500000.]]):
        near["LANE"].insert(0, NS(id="invalid", polygon=Polygon(origin), baseline_path=NS(linestring=Line(points))))
    kinds = ["INTERSECTION", "CROSSWALK", "STOP_LINE", "WALKWAYS", "CARPARK_AREA"]
    for i in range(n_polygons):
        near[kinds[i % 5]].append(NS(id=f"polygon_{i}", polygon=Polygon(origin + [i % 8 - 4, i // 8 - 3])))
    class Map:
        map_name = "synthetic"
        def get_proximal_map_objects(self, point, radius, layers):
            assert radius == 100 and layers == list(vars(LAYERS))
            self.last_query = (point.x, point.y)
            return near
    return Map()


def make_history(frame, n_actors):
    states, observations = [], []
    for i in range(6):
        ego = NS(x=600000. + .3 * frame + .1 * i, y=4500000. + .2 * frame,
                 heading=(.0, .3, np.pi / 2, -np.pi + 1e-9)[frame % 4])
        states.append(NS(rear_axle=ego, time_point=NS(time_s=frame + i * .1), dynamic_car_state=NS(speed=2.)))
        actors = []
        for j in range(n_actors):
            if i == 2 and j % 2:
                continue
            actor = NS(track_token=f"actor_{j}", velocity=NS(x=.5 if j == 0 else .7, y=0.),
                       center=NS(x=600001. + j * 2 + .1 * i, y=4500001., heading=.25),
                       box=NS(width=2., length=4.3, height=1.8))
            actors.append(actor)
            if j == 2:  # Duplicate token: reference keeps the first historical object.
                actors.append(NS(**{**vars(actor), "center": NS(x=1., y=1., heading=0.)}))
        observations.append(NS(tracked_objects=NS(tracked_objects=actors)))
    return NS(ego_states=states, observations=observations)


def synthetic(reference, device="cpu", atol=0.0):
    torch = NumpyTorch if device == "numpy" else importlib.import_module("torch")
    cls = load_reference(reference, torch)
    reports = []
    for n_lanes, n_polygons, n_actors in [(1, 0, 0), (17, 7, 10), (105, 106, 10)]:
        planner = cls()
        planner.device = "cpu" if device == "numpy" else device
        planner.map = make_map(n_lanes, n_polygons)
        histories = [make_history(i, n_actors) for i in range(4)]
        reports.append({"lanes": n_lanes, "polygons": n_polygons,
                        "frames": validate_context(planner, histories, backend="numpy" if device == "numpy" else "device", atol=atol)})
    # Content and map invalidation, independently from the old polygon id-only cache.
    cache = WorldCache()
    planner = cls()
    planner.device, planner.map = "cpu", make_map(1, 1)
    ego = make_history(0, 0).ego_states[-1].rear_axle
    symbols = cls.features.__globals__
    cache.prepare(planner, ego, symbols)
    before, generation = cache.polygon_world.copy(), cache.generation
    near = planner.map.get_proximal_map_objects(NS(x=ego.x, y=ego.y), 100, list(vars(LAYERS)))
    near["INTERSECTION"][0].polygon = Polygon([600030., 4500040.])
    cache.prepare(planner, ego, symbols)
    assert not np.array_equal(before, cache.polygon_world)
    planner.map = make_map(1, 1)
    cache.prepare(planner, ego, symbols)
    assert cache.generation == generation + 1
    return {"device": device, "atol": atol, "cases": reports, "cache_invalidation": "passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True, help="Exact indexed_features.py")
    parser.add_argument("--device", default="numpy", help="numpy (no torch), cpu, or cuda")
    parser.add_argument("--atol", type=float, default=0.0, help="Feature tolerance only; masks, IDs and flips must match exactly")
    parser.add_argument("--context-factory", help="module:function returning planner, histories")
    parser.add_argument("--flip-backend", choices=("device", "reference"), default="device")
    parser.add_argument("--artifacts", type=Path, help="Save native reference/actual raw features before assertions")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the first injected native history after parity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = synthetic(args.reference, args.device, args.atol)
    if args.context_factory:
        module, name = args.context_factory.split(":")
        planner, histories = getattr(importlib.import_module(module), name)()
        histories = list(histories)
        report["native_context"] = validate_context(planner, histories, atol=args.atol,
                                                     flip_backend=args.flip_backend, artifacts=args.artifacts)
        if args.benchmark:
            report["benchmark"] = benchmark_context(planner, histories[0], flip_backend=args.flip_backend)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    if args.output:
        print(json.dumps({"output": str(args.output), "status": "passed", "device": args.device}))
    else:
        print(text)


if __name__ == "__main__":
    main()
