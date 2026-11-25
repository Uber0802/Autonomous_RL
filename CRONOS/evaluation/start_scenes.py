"""Start-scene perturbations for standalone eval (`eval_only.py --start-scene`).

Applied once per unit, right after the reset + first `set_task` of a round and
before the first rollout step (`SequentialEvaluator(start_scene_fn=...)`):

  default                   no perturbation
  overlap_obj_nontarget     task object placed on the scene's other receptacle
  overlap_nonobj_target     the scene's other object placed on the task receptacle
  overlap_nonobj_nontarget  the other object placed on the other receptacle
  chaos                     every object / receptacle re-dropped from above the
                            table at random poses, overlap allowed
  overlapped                every object rests on a random receptacle, never the
                            task object on its target receptacle
  cluttered                 one receptacle rests partly on the other

overlapped / cluttered are defined on the out-of-domain base layout: unless
`--eval-domains` is given, eval_only.py runs them on out_of_domain only.

"Other" = the first object / receptacle of the env's scene that is not the one
of its current task. Random perturbations are seeded from
`start_scene|seed|domain|set|scene` (`envs/rng_streams.py`), so every round of a
pose set gets the same perturbed start, whatever the checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from envs import rng_streams

START_SCENES = ("default", "overlap_obj_nontarget", "overlap_nonobj_target",
                "overlap_nonobj_nontarget", "chaos", "overlapped", "cluttered")
# Start scenes whose default eval domain is out_of_domain (rand_ood base layout).
OOD_BASE_SCENES = ("overlapped", "cluttered")


def start_scene_seed(layout_seed: int, domain: str, pose_set: int, start_scene: str) -> int:
    key = rng_streams.stream_key("start_scene", seed=layout_seed, domain=domain, set=pose_set,
                                 scene=start_scene)
    return rng_streams.stream_seed(key) % (2 ** 31 - 1)


def _scene_names(scene):
    """(objects, receptacles) of a scene in training-order first appearance."""
    from envs.scheduler import TaskScheduler
    objs, receps = [], []
    for task in scene.training_order:
        o, r = TaskScheduler._extract_obj_recep(task)
        if o not in objs:
            objs.append(o)
        if r not in receps:
            receps.append(r)
    return objs, receps


def _other(names: List[str], current: str):
    return next((n for n in names if n != current), None)


class StartScene:
    """Callable hook: (obs, *, eval_kind, p, seq, objs, receps) -> obs."""

    def __init__(self, args, env, layout_seed: int):
        if args.start_scene not in START_SCENES:
            raise ValueError(f"Unknown --start-scene {args.start_scene!r}; expected one of {START_SCENES}.")
        self.args = args
        self.env = env
        self.layout_seed = layout_seed

    @property
    def name(self) -> str:
        return self.args.start_scene

    def params(self) -> dict:
        """Everything that changes the perturbed scene (goes into the plan fingerprint)."""
        a = self.args
        if a.start_scene == "chaos":
            return dict(start_scene=a.start_scene, half=(a.chaos_half_x, a.chaos_half_y),
                        drop_height=a.chaos_drop_height, layer_gap=a.chaos_layer_gap,
                        tilt_deg=a.chaos_tilt_deg)
        if a.start_scene.startswith("overlap_"):
            return dict(start_scene=a.start_scene, overlap_dz=a.overlap_dz)
        return dict(start_scene=a.start_scene)

    def _others(self, p, objs, receps):
        other_obj, other_rec = [None] * len(objs), [None] * len(receps)
        for sc, (lo, hi) in zip(p.scenes, p.env_ranges):
            s_objs, s_receps = _scene_names(sc)
            for e in range(lo, hi):
                other_obj[e], other_rec[e] = _other(s_objs, objs[e]), _other(s_receps, receps[e])
        if any(x is None for x in other_obj + other_rec):
            raise ValueError(f"--start-scene {self.name} needs two objects and two receptacles in every scene")
        return other_obj, other_rec

    def __call__(self, obs, *, eval_kind, p, seq, objs, receps):
        a, name = self.args, self.args.start_scene
        if name == "default":
            return obs
        seed = start_scene_seed(self.layout_seed, eval_kind, seq.pose_set, name)
        if name == "overlapped":
            return self.env.overlapped_start(seed)
        if name == "cluttered":
            return self.env.cluttered_start(seed)
        if name == "chaos":
            return self.env.chaos_drop(seed, half_edge=(a.chaos_half_x, a.chaos_half_y),
                                       drop_height=a.chaos_drop_height, layer_gap=a.chaos_layer_gap,
                                       tilt_max_deg=a.chaos_tilt_deg)
        other_obj, other_rec = self._others(p, objs, receps)
        if name == "overlap_obj_nontarget":
            return self.env.overlap_start(objs, other_rec, dz=a.overlap_dz)
        if name == "overlap_nonobj_target":
            return self.env.overlap_start(other_obj, receps, dz=a.overlap_dz)
        return self.env.overlap_start(other_obj, other_rec, dz=a.overlap_dz)


def save_tiled(obs, path: Path, max_envs: int = 64, cols: int = 8) -> None:
    """Tile the first `max_envs` env images (half resolution) into one PNG."""
    import cv2
    imgs = obs[:max_envs, ::2, ::2].cpu().numpy()
    n, h, w = imgs.shape[:3]
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        tile = np.ascontiguousarray(imgs[i])
        cv2.putText(tile, f"env{i}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
