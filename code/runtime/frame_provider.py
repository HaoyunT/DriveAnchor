"""Frame-local CV field cache; explicit static classification, no policy changes."""
import numpy as np
import torch
from device_obstacles import prepare_cv_field, collision_masks_tensor

class FrameCollisionProvider:
    def __init__(self):
        self._frame = None
        self._fields = {}

    def begin_frame(self, frame_id):
        if frame_id != self._frame:
            self._frame = frame_id
            self._fields.clear()

    def prepare_tracks(self, tracks, ego, local, decelerations, static_tokens,
                       *, frame_id, steps, dt):
        """CPU actor extraction boundary, cached per exact geometry/motion variant.

        static_tokens MUST come from the existing static history classifier.
        Do not derive it from instantaneous speed. Lead-inflated dimensions are
        part of the key, as are braking values, pose, membership and time grid.
        """
        self.begin_frame(frame_id)
        static_tokens = set(static_tokens)
        rows = []
        for obj in tracks:
            p = local([[obj.center.x, obj.center.y]], ego)[0]
            v = (local([[ego.x + obj.velocity.x, ego.y + obj.velocity.y]], ego)[0]
                 if hasattr(obj, 'velocity') else np.zeros(2))
            token = getattr(obj, 'track_token', None)
            rows.append([*p, *v, decelerations.get(token, 0.),
                         obj.center.heading - ego.heading, obj.box.length,
                         obj.box.width, token in static_tokens])
        values = np.asarray(rows, dtype=np.float64).reshape(-1, 9)
        key = (steps, dt, values.shape, values.tobytes())
        if key not in self._fields:
            a = torch.as_tensor(values, device='cuda', dtype=torch.float64)
            self._fields[key] = prepare_cv_field(
                a[:, :2], a[:, 2:4], a[:, 4], a[:, 5], a[:, 6], a[:, 7],
                a[:, 8].bool(), steps=steps, dt=dt)
        return self._fields[key]

    @staticmethod
    def collision_mask_tensor(centers, heading, field, rules):
        out = collision_masks_tensor(centers, heading, field,
                                    half_length=rules.half_length,
                                    half_width=rules.half_width)
        # Keep invalid independent, but always include it in the normal gate.
        out['normal_rejected'] = (out['static_collision'] |
                                  out['dynamic_collision'] | out['invalid'])
        return out
