"""CRONOS — background/table catalog figure.

Renders the *same* scene — same objects, same receptacles, same WidowX arm, same
poses — under every available background, one image per background.

Mechanism, mirroring what `four_group_sequential_2x2.yaml` does with its
per-group ``background: k`` key: `CronosWrapper._build_options` turns that key
into ``options["select_overlay_ids"]``, which
`GenericNxMPickPlace._initialize_episode_pre` uses to pick the overlay composited
by `_green_sceen_rgb`. This script sets the same option to a different index per
env, so one vectorized reset yields every background at once.

Everything else is pinned so the background is the only thing that moves:
``obj_set="fixed"`` plus one shared ``episode_id`` makes object choice, position
and rotation identical in every env.

Two notes on what "background" means in this env:

- Background and table are not separate axes. ``more_table/imgs/*.png`` are full
  640x480 photographs containing both the table and the room behind it, and
  `_green_sceen_rgb` replaces every non-robot, non-object pixel with one of
  them. Swapping the index swaps both at once. (`GroupSpec.table` is parsed but
  read nowhere in the repo.)
- ``more_table/textures/*.png`` and the ``mix`` weights are loaded and uploaded
  each reset, but the lines that would blend them are commented out in
  `_green_sceen_rgb`, so surface texture currently has no effect on the image.
  This catalog therefore has exactly 21 distinct looks, tiled 7x3 edge to edge.

Usage::

    cd <repo>/CRONOS
    CUDA_VISIBLE_DEVICES=2 python tools/render_background_catalog.py
    CUDA_VISIBLE_DEVICES=2 python tools/render_background_catalog.py --backgrounds config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CRONOS_DIR = Path(__file__).resolve().parents[1]
CARROT_DATASET_DIR = (
    Path(__file__).resolve().parents[2] / "ManiSkill" / "mani_skill" / "assets" / "carrot"
)
DEFAULT_CONFIG = CRONOS_DIR / "configs" / "four_group_sequential_2x2.yaml"
DEFAULT_OUT = CRONOS_DIR / "plotting" / "figures" / "background_catalog.png"


def parse_backgrounds(spec: str, config) -> list[int]:
    """Resolve --backgrounds into a list of 0-based overlay indices."""
    n_total = len(json.loads((CARROT_DATASET_DIR / "more_table" / "model_db.json").read_text()))
    if spec == "all":
        return list(range(n_total))
    if spec == "config":
        seen = []
        for g in config.groups:
            if isinstance(g.background, int) and g.background not in seen:
                seen.append(g.background)
        if not seen:
            raise ValueError(
                f"no group in the config declares an int `background:` key; "
                f"use --backgrounds all or an explicit list")
        return seen
    indices = [int(tok) for tok in spec.split(",") if tok.strip()]
    bad = [i for i in indices if not 0 <= i < n_total]
    if bad:
        raise ValueError(f"background indices out of range 0..{n_total - 1}: {bad}")
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML config supplying the fixed obj/recep indices")
    parser.add_argument("--group", type=int, default=0,
                        help="which config group's obj/recep to hold fixed")
    parser.add_argument("--backgrounds", default="all",
                        help="'all' (every overlay), 'config' (the ones the config uses), "
                             "or a comma list like '0,1,2,3'")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=42,
                        help="shared episode id; fixes object pose identically in every env")
    parser.add_argument("--cols", type=int, default=7,
                        help="tiles per row in the contact sheet")
    parser.add_argument("--no-tiles", action="store_true",
                        help="skip the per-background PNGs, write only the contact sheet")
    args = parser.parse_args()

    import torch
    import gymnasium as gym

    from envs.config import load_cronos_config
    import envs.bridge_multi  # noqa: F401 — triggers PickPlaceNxM-v1 registration

    config = load_cronos_config(args.config)
    if not 0 <= args.group < len(config.groups):
        raise SystemExit(f"--group {args.group} out of range (config has {len(config.groups)})")
    group = config.groups[args.group]
    backgrounds = parse_backgrounds(args.backgrounds, config)

    table_db = json.loads((CARROT_DATASET_DIR / "more_table" / "model_db.json").read_text())
    table_keys = list(table_db)

    n_obj, n_recep = len(group.obj), len(group.recep)
    num_envs = len(backgrounds)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[config] {args.config.name} group '{group.name}': "
          f"obj={group.obj} recep={group.recep} (N={n_obj}, M={n_recep})")
    print(f"[backgrounds] {len(backgrounds)}: {backgrounds}")

    env = gym.make(
        id="PickPlaceNxM-v1",
        num_envs=num_envs,
        N=n_obj,
        M=n_recep,
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        sim_backend="gpu",
        sim_config={"sim_freq": 500, "control_freq": 5},
        max_episode_steps=80,
        sensor_configs={"shader_pack": "default"},
    )

    # `obj_set="fixed"` skips the per-env random pose branch, so a single shared
    # episode_id pins position and rotation identically across every env; the
    # only thing that differs is select_overlay_ids.
    options = {
        "obj_set": "fixed",
        "episode_id": torch.full((num_envs,), args.episode_id, dtype=torch.long, device=device),
        "select_overlay_ids": torch.tensor(backgrounds, dtype=torch.long, device=device),
    }
    for i in range(n_obj):
        options[f"obj{i + 1}_index"] = group.obj[i]
    for i in range(n_recep):
        options[f"plate{i + 1}_index"] = group.recep[i]

    obs, _ = env.reset(seed=[args.seed * 1000 + i for i in range(num_envs)], options=options)
    rgb = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8).cpu().numpy()
    print(f"[render] {rgb.shape[0]} frames at {rgb.shape[2]}x{rgb.shape[1]}")
    env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    if not args.no_tiles:
        tile_dir = args.out.parent / f"{args.out.stem}_tiles"
        tile_dir.mkdir(parents=True, exist_ok=True)
        for frame, b in zip(rgb, backgrounds):
            iio.imwrite(tile_dir / f"bg{b:02d}__{Path(table_keys[b]).stem}.png", frame)
        print(f"[tiles] wrote {len(rgb)} PNGs to {tile_dir}")

    cols = min(args.cols, len(rgb))
    # Butt the frames straight together: no gutters, no captions, no figure
    # margin. Short final rows are filled with white so the sheet stays
    # rectangular.
    n, h, w, _ = rgb.shape
    rows = int(np.ceil(n / cols))
    sheet = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for idx in range(n):
        r, c = divmod(idx, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = rgb[idx]
    print(f"[grid] {cols}x{rows} tiles")
    iio.imwrite(args.out, sheet)
    print(f"[sheet] {sheet.shape[1]}x{sheet.shape[0]} px")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
