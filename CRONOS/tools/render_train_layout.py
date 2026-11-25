"""Render start scenes of the training layout and the in-domain eval layout.

No policy is loaded. For each of `--n-resets` resets it saves a tiled PNG of
all env camera images, once with the training layout (`--layout`, e.g. wide)
and once with the default in-domain eval layout, and prints per reset:
  - envs with any loaded actor off the table (z < 0.80)
  - envs whose CURRENT training task already counts as success at t=0

Usage (from CRONOS/):
  python tools/render_train_layout.py --config configs/four_group_sequential_2x2_wide.yaml \
      --out figures/train_layout_wide --n-resets 3
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging; logging.getLogger("mani_skill").setLevel(logging.ERROR)
from envs.wrapper import CronosWrapper
from envs.suite import TaskSuite
from envs.config import load_cronos_config, resolve_symbolic_task, has_symbolic_refs, build_obj_recep_name_maps
from envs.scheduler import TaskScheduler, GroupState
import envs.bridge_multi  # noqa: F401  (env registration)


def tile(obs, labels, cols=8, scale=2):
    import cv2
    imgs = obs[:, ::scale, ::scale].cpu().numpy()
    n, h, w = imgs.shape[:3]
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        t = np.ascontiguousarray(imgs[i])
        cv2.putText(t, labels[i], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-resets", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", default=None, help="override YAML train_layout")
    cli = ap.parse_args()
    import cv2

    cfg = load_cronos_config(cli.config)
    a = argparse.Namespace(
        env_id="PickPlaceNxM-v1", num_envs=sum(g.num_envs for g in cfg.groups),
        env_n=max(len(g.obj) for g in cfg.groups), env_m=max(len(g.recep) for g in cfg.groups),
        obj_set="rand", seed=cli.seed, obj1_index=7, obj2_index=2, obj3_index=10,
        plate1_index=1, plate2_index=2, plate3_index=3, scene="", segment_len=80,
        episode_len=80, task_len=80, reset_mode="per_episode", reset_robot=True,
        reset_unsuitable=False, unsuitable_detector="low_z", enable_backward=False,
        backward_interval=1, hsr_reset_scope="per_env", policy="spatialvla",
        wide_center_x=cfg.wide_center_x if cfg.wide_center_x is not None else -0.165,
        wide_center_y=cfg.wide_center_y if cfg.wide_center_y is not None else -0.025,
        wide_half_x=cfg.wide_half_x if cfg.wide_half_x is not None else 0.12,
        wide_half_y=cfg.wide_half_y if cfg.wide_half_y is not None else 0.20,
        wide_table_z=0.869,
    )
    layout = cli.layout or cfg.train_layout or "default"
    np.random.seed(a.seed); torch.manual_seed(a.seed)

    env = CronosWrapper(a, {"q01": np.zeros(7), "q99": np.ones(7)}, TaskSuite(), device=torch.device("cuda:0"))
    u = env.env.unwrapped
    states = []
    for g in cfg.groups:
        on, rn = build_obj_recep_name_maps(g.obj, g.recep, u.model_db_carrot, u.model_db_plate)
        seq = [resolve_symbolic_task(t, on, rn) if has_symbolic_refs(t) else t for t in g.task_sequence]
        states.append(GroupState.from_sequence(name=g.name, task_sequence=seq, eval_tasks=seq, num_envs=g.num_envs))
    sched = TaskScheduler(group_states=states, mode=cfg.task_order, num_envs=a.num_envs, fan_out=cfg.fan_out)
    env.set_scheduler(sched)
    env.set_group_specs(cfg.groups)

    out = Path(cli.out); out.mkdir(parents=True, exist_ok=True)
    print(f"layout={layout}  wide center=({a.wide_center_x},{a.wide_center_y}) half=({a.wide_half_x},{a.wide_half_y})")
    eval_layout = layout if (cfg.eval_in_domain_layout or "default") == "train" else None
    # in-domain eval task assignment = main.eval_all_groups rotation, episode 0
    ev_obj, ev_rec = [], []
    for gs in states:
        n_t = len(gs.task_pool)
        for i in range(gs.num_envs):
            o, rc = TaskScheduler._extract_obj_recep(gs.task_pool[i % n_t])
            ev_obj.append(o); ev_rec.append(rc)
    print(f"in-domain eval layout={eval_layout or 'default'}")
    for r in range(cli.n_resets):
        for kind in ("train", "eval_in_domain"):
            if kind == "train":
                obs, instr, _ = env.reset(layout=layout)
            else:
                env.reset(skip_scheduler=True, layout=eval_layout)
                env.set_task(ev_obj, ev_rec)
            obs, hist = env.fix_wide_starts()
            instr = env.get_language_instructions()
            if hasattr(u, "_recompute_bbox_world"):
                u._recompute_bbox_world()
            info = u.get_info()
            solved = info["success"].bool()
            fallen = u._fallen_mask()
            labels = [f"env{i} {instr[i].replace('put ', '')}"[:44] + (" [SOLVED]" if solved[i] else "")
                      for i in range(a.num_envs)]
            path = out / f"{kind}_reset{r}.png"
            cv2.imwrite(str(path), cv2.cvtColor(tile(obs, labels), cv2.COLOR_RGB2BGR))
            starts = np.cumsum([0] + [g.num_envs for g in cfg.groups])[:-1]
            close = tile(obs[starts.tolist()], [labels[i] for i in starts], cols=len(starts), scale=1)
            cv2.imwrite(str(out / f"{kind}_reset{r}_closeup.png"), cv2.cvtColor(close, cv2.COLOR_RGB2BGR))
            print(f"[{kind} reset{r}] re-sample rounds (bad envs): {hist}  final off-table: "
                  f"{int(fallen.sum())}/{a.num_envs}  final already-solved: {int(solved.sum())}/{a.num_envs}  -> {path}")

if __name__ == "__main__":
    main()
