"""Independent geometry checks for device_costmap; defaults to CPU.

Use --device cuda only inside the project's existing shared GPU execution lock.
No server, case runner, lock acquisition, or background GPU work is launched.
Test-only NumPy downloads are intentional; the production module has none.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'device_providers'))
from device_obstacles import CVField, collision_masks_tensor, prepare_cv_field
from device_costmap import build_costmap


HALF_LENGTH, HALF_WIDTH = 2.4, 1.0


def tensor(value, device, dtype=torch.float64):
    return torch.as_tensor(value, device=device, dtype=dtype)


def numpy_braking_centers(position, velocity, deceleration, steps, dt):
    """Independent per-actor scalar braking integration, from observed inputs."""
    times = np.arange(steps) * dt
    result = np.empty((steps, len(position), 2), dtype=np.float64)
    for actor, (p, v, a) in enumerate(zip(position, velocity, deceleration)):
        speed = np.linalg.norm(v)
        if a < 0 and speed >= 1e-6:
            elapsed = np.minimum(times, speed / -a)
            distance = speed * elapsed + .5 * a * elapsed ** 2
            result[:, actor] = p + distance[:, None] * (v / speed)
        else:
            result[:, actor] = p + times[:, None] * v
    return result


def independent_overlap(ego_center, ego_heading, actor_center, actor_heading,
                        actor_length, actor_width):
    """Polygon-corner projection SAT, independent of production radius algebra."""
    def corners(center, heading, length, width):
        forward = np.stack((np.cos(heading), np.sin(heading)), -1)
        side = np.stack((-forward[..., 1], forward[..., 0]), -1)
        signs = np.array([[-1., -1.], [-1., 1.], [1., 1.], [1., -1.]])
        points = (center[..., None, :]
                  + forward[..., None, :] * np.asarray(length)[..., None, None] / 2 * signs[:, :1]
                  + side[..., None, :] * np.asarray(width)[..., None, None] / 2 * signs[:, 1:])
        return points, forward, side
    ec, ef, es = corners(ego_center, ego_heading, 2 * HALF_LENGTH, 2 * HALF_WIDTH)
    ac, af, ass = corners(actor_center, actor_heading[None],
                          actor_length[None], actor_width[None])
    ec = ec[:, :, None]
    ac = ac[None]
    result = np.ones((ego_center.shape[0], ego_center.shape[1], len(actor_heading)), bool)
    for axis in (ef[:, :, None], es[:, :, None], af[None], ass[None]):
        ep = (ec * axis[..., None, :]).sum(-1)
        ap = (ac * axis[..., None, :]).sum(-1)
        # Touching projections are NOT strict collisions.
        result &= ((ep.max(-1) > ap.min(-1)) & (ap.max(-1) > ep.min(-1)))
    return result


def field_from_inputs(position, velocity, deceleration, heading, length, width,
                      static, steps, dt, device):
    expected = numpy_braking_centers(position, velocity, deceleration, steps, dt)
    if torch.device(device).type == 'cuda':
        field = prepare_cv_field(*[tensor(v, device) for v in
                                   (position, velocity, deceleration, heading, length, width)],
                                 tensor(static, device, torch.bool), steps=steps, dt=dt)
        np.testing.assert_allclose(field.centers.cpu().numpy(), expected, atol=1e-12, rtol=1e-12)
    else:
        field = CVField(tensor(expected, device), tensor(heading, device),
                        tensor(length, device), tensor(width, device),
                        tensor(static, device, torch.bool), tensor(False, device, torch.bool), dt)
    return field, expected


def check_case(name, field, ego_center, ego_heading, *, bounds=(-150., -150., 150., 150.)):
    device = field.centers.device
    costmap = build_costmap(field, half_length=HALF_LENGTH, half_width=HALF_WIDTH, bounds=bounds)
    xy, yaw = tensor(ego_center, device), tensor(ego_heading, device)
    query = costmap.query(xy)
    hit = query['hit'].cpu().numpy()
    invalid = query['invalid'].cpu().numpy()
    exact = independent_overlap(ego_center, ego_heading, field.centers.cpu().numpy(),
                                field.heading.cpu().numpy(), field.length.cpu().numpy(),
                                field.width.cpu().numpy())
    assert not (exact.any(-1) & ~hit).any(), name + ': map missed a strict exact overlap'
    static = field.static.cpu().numpy()
    selected = (query['hit'] | query['invalid']).any(1)
    if device.type == 'cuda':
        # Mirror adapter candidate filtering: exact SAT is still authoritative.
        indices = selected.nonzero(as_tuple=True)[0]
        sub = collision_masks_tensor(xy[indices], yaw[indices], field,
                                     half_length=HALF_LENGTH, half_width=HALF_WIDTH)
        full_static = torch.zeros(xy.shape[:2], device=device, dtype=torch.bool)
        full_dynamic = torch.zeros_like(full_static)
        full_static[indices] = sub['static_by_time']
        full_dynamic[indices] = sub['dynamic_by_time']
        np.testing.assert_array_equal(full_static.cpu().numpy(), (exact & static).any(-1))
        np.testing.assert_array_equal(full_dynamic.cpu().numpy(), (exact & ~static).any(-1))
    else:
        # CPU has no production CV adapter; independently prove the same filter.
        filtered = exact & selected.cpu().numpy()[:, None, None]
        np.testing.assert_array_equal(filtered, exact)
    return costmap, dict(name=name, candidates=len(ego_center),
                        exact_candidates=int(selected.sum().cpu()),
                        exact_colliding_samples=int(exact.any(-1).sum()),
                        invalid_queries=int(invalid.sum()), layers=len(costmap.layers),
                        horizon=costmap.horizon, false_negatives=0)


def run_tests(device):
    rng = np.random.default_rng(61894)
    passed = []
    cases = []
    for n, actors, steps, dt in [(37, 33, 40, .1), (19, 7, 41, .1), (11, 5, 22, .13)]:
        position = rng.uniform(-165, 165, (actors, 2))
        velocity = rng.uniform(-12, 12, (actors, 2))
        deceleration = rng.uniform(-5, 1, actors)
        heading = rng.uniform(-np.pi, np.pi, actors)
        length = rng.uniform(.2, 24, actors)
        width = rng.uniform(.2, 4.5, actors)
        static = np.arange(actors) % 3 == 0
        # A static label is a partition only; retain supplied velocities.
        field, actor_centers = field_from_inputs(position, velocity, deceleration, heading,
                                                length, width, static, steps, dt, device)
        ego = rng.uniform(-170, 170, (n, steps, 2))
        ego[:min(n, actors)] = actor_centers[:, :min(n, actors)].transpose(1, 0, 2)
        yaw = rng.uniform(-np.pi, np.pi, (n, steps))
        name = 'random_braking_%s_%s_%s_%s' % (n, actors, steps, dt)
        _, report = check_case(name, field, ego, yaw)
        cases.append(report)
        passed.append(name)

    # Long crossing vehicles, stopping and moving/static membership together.
    position = np.array([[-35., 0.], [0., -28.], [148., 4.], [154., -10.], [-155., 20.]])
    velocity = np.array([[14., 0.], [0., 11.], [0., 0.], [0., 0.], [0., 0.]])
    deceleration = np.array([-4., 0., 0., 0., 0.])
    heading = np.array([0., np.pi / 2, .25, 0., 0.])
    length = np.array([22., 18., 12., 20., 20.])
    width = np.array([3., 3.2, 3., 3., 3.])
    static = np.array([False, False, True, True, True])
    field, actor_centers = field_from_inputs(position, velocity, deceleration, heading,
                                            length, width, static, 40, .1, device)
    ego = np.zeros((8, 40, 2))
    ego[1, :, 0] = np.linspace(-25., 25., 40)
    ego[2] = actor_centers[:, 0]
    ego[3] = [149.5, -10.]
    ego[4] = [-149.5, 20.]
    ego[5] = [149.99, 4.]
    ego[6] = [150., 0.]
    ego[7] = [-150.01, 0.]
    costmap, report = check_case('crossing_large_vehicles_and_outside_actor_inflation',
                                field, ego, np.zeros((8, 40)))
    cases.append(report)
    assert report['layers'] == 8 and abs(costmap.horizon - 3.9) < 1e-12
    assert costmap.query(tensor(ego, device))['hit'][3:5].all()
    passed.append(report['name'])

    # 0.5 s boundary and final partial-bin endpoint must be accepted.
    boundary_xy = tensor(np.stack((actor_centers[5, 0], actor_centers[39, 0])), device)
    boundary_times = tensor([.5, 3.9], device)
    boundary = costmap.query(boundary_xy, boundary_times)
    assert boundary['hit'].all() and not boundary['invalid'].any()
    outside = costmap.query(boundary_xy, tensor([-.001, 3.90001], device))
    assert outside['hit'].all() and outside['invalid'].all()
    passed += ['time_boundary_0_5_and_partial_horizon_3_9', 'out_of_time_fails_closed']

    # Interpolation points between CV samples are covered even for turning
    # center polylines. This tests actor sweep, not ego between-sample safety.
    polyline = tensor([[[-20., -15.]], [[20., -15.]], [[20., 15.]], [[-20., 15.]]], device)
    turning = CVField(polyline, tensor([0.], device), tensor([8.], device),
                      tensor([2.], device), tensor([False], device, torch.bool),
                      tensor(False, device, torch.bool), .3)
    turning_map = build_costmap(turning, half_length=HALF_LENGTH, half_width=HALF_WIDTH)
    sweep_xy = torch.stack((polyline[:-1, 0] * .75 + polyline[1:, 0] * .25,
                            polyline[:-1, 0] * .25 + polyline[1:, 0] * .75), 1).reshape(-1, 2)
    sweep_times = tensor([.075, .225, .375, .525, .675, .825], device)
    assert turning_map.query(sweep_xy, sweep_times)['hit'].all()
    passed.append('continuous_actor_polyline_between_samples')

    # Interval query must see an actor entering in a later bin, while a point
    # query before entry is free; intervals beyond the actual horizon fail closed.
    line = np.zeros((40, 1, 2))
    line[:, 0, 0] = -30 + np.arange(40) * 1.5
    moving = CVField(tensor(line, device), tensor([0.], device), tensor([4.], device),
                     tensor([2.], device), tensor([False], device, torch.bool),
                     tensor(False, device, torch.bool), .1)
    moving_map = build_costmap(moving, half_length=HALF_LENGTH, half_width=HALF_WIDTH)
    origin = tensor([[0., 0.]], device)
    assert not moving_map.query(origin, tensor([0.], device))['hit'].any()
    interval = moving_map.query_interval(origin, tensor([0.], device), tensor([2.1], device))
    assert interval['hit'].all() and not interval['invalid'].any()
    for start, end in [(-.1, .2), (3.8, 4.), (1., .9), (float('nan'), 1.)]:
        out = moving_map.query_interval(origin, tensor([start], device), tensor([end], device))
        assert out['hit'].all() and out['invalid'].all()
    passed += ['interval_unions_later_bins', 'invalid_intervals_fail_closed']

    # A=0 is a valid empty field, including points near all four map edges.
    empty = CVField(field.centers[:, :0], field.heading[:0], field.length[:0],
                    field.width[:0], field.static[:0], field.invalid, field.dt)
    empty_ego = np.tile([[-149.99, -149.99], [149.99, 149.99], [0., 0.]], (40, 1, 1)).transpose(1, 0, 2)
    empty_map, report = check_case('empty_actor_field', empty, empty_ego, np.zeros((3, 40)))
    assert not empty_map.layers.any() and report['exact_candidates'] == 0
    cases.append(report)
    bad_xy = tensor([[150., 0.], [-150.001, 0.], [0., 150.], [0., -150.001],
                     [float('nan'), 0.], [float('inf'), 0.], [1e300, -1e300]], device)
    bad = empty_map.query(bad_xy, tensor(0., device))
    assert bad['hit'].all() and bad['invalid'].all()
    passed += ['empty_actor_field', 'outside_nan_infinite_extreme_positions_fail_closed']

    for bad_field in (replace(field, invalid=tensor(True, device, torch.bool)),
                      replace(field, heading=field.heading * float('nan')),
                      replace(field, centers=field.centers * float('nan')),
                      replace(field, length=-field.length)):
        bad_map = build_costmap(bad_field, half_length=HALF_LENGTH, half_width=HALF_WIDTH)
        out = bad_map.query(origin, tensor([0.], device))
        assert out['hit'].all() and out['invalid'].all()
    passed.append('invalid_field_geometry_fails_closed')

    # A broadphase false positive is resolved by exact SAT, never hard rejected.
    stopped = replace(moving, centers=torch.zeros_like(moving.centers))
    sparse_ego = np.full((96, 40, 2), 80.)
    sparse_ego[0] = [0., 0.]
    sparse_ego[1] = [5., 0.]  # circumdisc hits, but length-axis SAT is separated
    sparse_map, report = check_case('sparse_scene_reduces_exact_candidates', stopped,
                                    sparse_ego, np.zeros((96, 40)))
    assert 0 < report['exact_candidates'] < report['candidates']
    assert sparse_map.query(tensor(sparse_ego, device))['hit'][1].all()
    assert not independent_overlap(sparse_ego[1:2], np.zeros((1, 40)),
                                    stopped.centers.cpu().numpy(), np.array([0.]),
                                    np.array([4.]), np.array([2.])).any()
    cases.append(report)
    passed += ['broadphase_positive_requires_exact_resolution', report['name']]
    # Invalid headings are not in the query API: adapter MUST retain its own
    # exact-input invalid flag even for map-empty candidates (documented contract).
    return dict(tests=passed, cases=cases, false_negatives=0,
                exact_comparison='production CUDA four-axis + independent polygon SAT'
                if torch.device(device).type == 'cuda' else 'independent polygon-corner SAT',
                excludes='DTPP, final reserve, road constraints, moving ego between samples')


def benchmark(device, candidates=500, actors=32):
    if torch.device(device).type != 'cuda':
        raise ValueError('--benchmark requires --device cuda under the shared GPU lock')
    generator = torch.Generator(device=device).manual_seed(718)
    centers = torch.randn((40, actors, 2), generator=generator, device=device,
                          dtype=torch.float64) * 12
    field = CVField(centers, torch.zeros(actors, device=device, dtype=torch.float64),
                    torch.full((actors,), 4., device=device, dtype=torch.float64),
                    torch.full((actors,), 2., device=device, dtype=torch.float64),
                    torch.zeros(actors, device=device, dtype=torch.bool),
                    torch.tensor(False, device=device), .1)
    xy = torch.randn((candidates, 40, 2), generator=generator, device=device,
                     dtype=torch.float64) * 60
    build = lambda: build_costmap(field, half_length=HALF_LENGTH, half_width=HALF_WIDTH)
    costmap = build()
    def timing(call):
        for _ in range(3):
            call()
        samples = []
        for _ in range(20):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return dict(p50_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)))
    return dict(candidates=candidates, actors=actors, grid=[300, 300], layers=len(costmap.layers),
                build=timing(build), query=timing(lambda: costmap.query(xy)),
                excludes='actor preparation, upload, exact SAT, adapter compaction, DTPP and final checks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--output', help='optional JSON report path; omitted means stdout only')
    args = parser.parse_args()
    result = run_tests(args.device)
    result['device'] = args.device
    if args.benchmark:
        result['benchmark'] = benchmark(args.device)
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + '\n')
    print(rendered)
