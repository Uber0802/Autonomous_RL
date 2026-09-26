"""CRONOS — asset catalog figure.

Lays every manipulable object (``more_carrot``, 25) and every receptacle
(``more_plate``, 17) out on **one shared grid** inside a single SAPIEN scene and
captures the whole arrangement in one camera shot on a pure-white background.
Objects and receptacles are not split into separate blocks: 25 + 17 = 42 fills a
7-column grid exactly, six rows deep, so both categories share the same rows and
columns. Nothing is composited or stitched — this is one real render of 42
assets.

This deliberately does **not** go through ``PickPlaceNxM-v1``. That env's
``3rd_view_camera`` is the calibrated Bridge 640x480 sensor framing a ~15x15 cm
workspace, so 42 assets do not fit in it, and every instance also drags in the
arena mesh plus the WidowX arm. Re-aiming that camera would mean touching the
training env. Instead this builds a throwaway render scene and loads the meshes
with the same rules ``BasePickPlace._build_actor_helper`` uses — same files
(``textured.obj`` -> ``.dae`` -> ``.glb``), same ``model_db`` scale — so the
geometry shown is the geometry the env simulates.

Assets keep their true relative size and rest with their bounding-box bottom on
z=0, as they would on a table. Columns are sized by their widest member and rows
by their deepest, so the grid stays regular without stranding a golf ball in as
much space as a kitchen shovel. The shot is taken from an oblique angle (``--elev`` above the ground, ``--azim``
off head-on) so the grid recedes and the assets read as solids rather than as
flat top-down silhouettes.

Usage::

    cd <repo>/CRONOS
    CUDA_VISIBLE_DEVICES=2 python plotting/render_asset_catalog.py
    CUDA_VISIBLE_DEVICES=2 python plotting/render_asset_catalog.py --elev 62 --azim -90
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CARROT_DATASET_DIR = (
    Path(__file__).resolve().parents[2] / "ManiSkill" / "mani_skill" / "assets" / "carrot"
)
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "plotting" / "figures" / "asset_catalog.png"

# Mesh filename preference, matching `BasePickPlace._build_actor_helper`.
MESH_CANDIDATES = ("textured.obj", "textured.dae", "textured.glb")

# Filled in grid order, objects first: 25 + 17 = 42 = 6 rows x 7 columns.
SUBDIRS = ("more_carrot", "more_plate")

LIGHTS = (
    ((-0.6, -0.5, -1.0), (1.6, 1.6, 1.6)),
    ((0.7, 0.4, -0.8), (0.7, 0.7, 0.7)),
    ((0.0, 0.0, 1.0), (0.35, 0.35, 0.35)),
)
AMBIENT = (0.55, 0.55, 0.55)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """SAPIEN camera pose looking from `eye` at `target`.

    SAPIEN cameras look down local +X, with local +Y left and +Z up. Verified to
    agree with `mani_skill.utils.sapien_utils.look_at` to float precision;
    reimplemented here so the catalog does not need the simulation stack.
    """
    from transforms3d.quaternions import mat2quat

    eye = np.asarray(eye, dtype=float)
    fwd = np.asarray(target, dtype=float) - eye
    fwd /= np.linalg.norm(fwd)
    left = np.cross(np.asarray(up, dtype=float), fwd)
    norm = np.linalg.norm(left)
    if norm < 1e-6:  # view direction parallel to `up`
        left = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
        norm = np.linalg.norm(left)
    left /= norm
    rot = np.stack([fwd, left, np.cross(fwd, left)], axis=1)
    return eye, mat2quat(rot), rot


def load_model_db(subdir: str) -> dict:
    with open(CARROT_DATASET_DIR / subdir / "model_db.json") as f:
        return json.load(f)


def mesh_path(subdir: str, key: str) -> str:
    """First existing visual mesh for an asset, in `_build_actor_helper` order."""
    base = CARROT_DATASET_DIR / subdir / key
    for candidate in MESH_CANDIDATES:
        if (base / candidate).exists():
            return str(base / candidate)
    raise FileNotFoundError(f"no visual mesh for {subdir}/{key} (tried {MESH_CANDIDATES})")


def asset_scale(entry: dict) -> float:
    """Scale from model_db.

    The JSON key is ``scales``; note `bridge_multi.py` reads ``scale`` and so
    always falls back to 1.0. Every shipped value is [1.0], making the two agree
    today, but this reads the key that is actually present so the catalog stays
    honest if that ever changes.
    """
    scales = entry.get("scales", entry.get("scale", [1.0]))
    return float(scales[0])


def grid_slots(extents, cols, pad_frac, elev_rad, azim_rad):
    """Slot centres for one shared grid, with per-column widths and row heights.

    A single pitch for the whole grid would have to accommodate the largest asset
    anywhere in it, leaving a golf ball marooned in as much space as a kitchen
    shovel. Sizing each column by the widest asset *in that column* and each row
    by the deepest asset *in that row* keeps the grid regular — objects and
    receptacles still line up on shared rows and columns — and keeps every asset
    at its true scale, while removing most of the dead space.

    Cells also have to pay for height. Under a camera tilted `elev_rad` above the
    ground, a point at height h lands where the ground point ``h / tan(elev)``
    further along the view's horizontal direction would, so a tall asset climbs
    into its neighbour. That displacement is ``-(h / tan(elev)) * (cos azim,
    sin azim)``, so an oblique azimuth spends part of it sideways: columns are
    charged the x component and rows the y component. Without this the champagne
    glass overlaps whatever sits behind it.

    Returns (centres, width, height) with centres in world XY, origin at the grid
    centre and the first row on top.
    """
    n = len(extents)
    rows = int(np.ceil(n / cols))
    col_w = np.zeros(cols)
    row_h = np.zeros(rows)
    lean = 1.0 / np.tan(elev_rad)
    lean_x, lean_y = abs(np.cos(azim_rad)) * lean, abs(np.sin(azim_rad)) * lean
    for idx, (ex, ey, ez) in enumerate(extents):
        r, c = divmod(idx, cols)
        col_w[c] = max(col_w[c], ex + ez * lean_x)
        row_h[r] = max(row_h[r], ey + ez * lean_y)
    col_w *= 1.0 + pad_frac
    row_h *= 1.0 + pad_frac

    # Cumulative centres, then shift so the grid is centred on the origin.
    col_x = np.cumsum(col_w) - col_w / 2.0 - col_w.sum() / 2.0
    row_y = row_h.sum() / 2.0 - (np.cumsum(row_h) - row_h / 2.0)

    centres = [(float(col_x[idx % cols]), float(row_y[idx // cols])) for idx in range(n)]
    return centres, float(col_w.sum()), float(row_h.sum())


def fit_distance(rot, target, points, fovy, aspect, margin):
    """Smallest camera distance along the view axis that keeps `points` in frame.

    With the eye at ``target + d * u`` the camera-space coordinates of a point
    are ``a + d * (1, 0, 0)`` where ``a = rotᵀ (P - target)``, because the camera
    looks down local +X straight back along ``u``. The frustum test is therefore
    linear in ``d`` and solves in closed form.
    """
    tan_y = np.tan(fovy / 2.0)
    tan_x = tan_y * aspect
    a = (np.asarray(points, dtype=float) - np.asarray(target, dtype=float)) @ rot
    needed = np.maximum(np.abs(a[:, 1]) / tan_x, np.abs(a[:, 2]) / tan_y) - a[:, 0]
    return float(needed.max()) * margin


def collect_assets():
    """Every asset in grid order, as (subdir, key, entry, bbox_min, bbox_max)."""
    assets = []
    for subdir in SUBDIRS:
        for key, entry in load_model_db(subdir).items():
            scale = asset_scale(entry)
            lo = np.array(entry["bbox"]["min"]) * scale
            hi = np.array(entry["bbox"]["max"]) * scale
            assets.append((subdir, key, entry, lo, hi))
    return assets


def render_catalog(args):
    """Build the whole arrangement in one scene and take a single picture."""
    import sapien

    scene = sapien.Scene()
    scene.set_ambient_light(list(AMBIENT))
    for direction, color in LIGHTS:
        scene.add_directional_light(list(direction), list(color))

    assets = collect_assets()
    extents = [(float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]))
               for *_, lo, hi in assets]
    centres, width, height = grid_slots(extents, args.cols, args.pad_frac,
                                        np.radians(args.elev), np.radians(args.azim))

    corners = []
    for (subdir, key, entry, lo, hi), (x, y) in zip(assets, centres):
        centre = (lo + hi) / 2.0

        builder = scene.create_actor_builder()
        builder.add_visual_from_file(filename=mesh_path(subdir, key),
                                     scale=[asset_scale(entry)] * 3)
        # Centre the bbox on the slot in XY and rest its bottom on z=0, so
        # assets sit on a common ground plane rather than on mesh origins.
        builder.initial_pose = sapien.Pose(p=[x - centre[0], y - centre[1], -lo[2]])
        builder.build_kinematic(name=f"{subdir}/{key}")

        span = (hi - lo) / 2.0
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (0, 1):
                    corners.append([x + sx * span[0], y + sy * span[1],
                                    sz * float(hi[2] - lo[2])])

    target = np.array([0.0, 0.0, 0.0])
    elev, azim = np.radians(args.elev), np.radians(args.azim)
    view_dir = np.array([np.cos(elev) * np.cos(azim),
                         np.cos(elev) * np.sin(azim),
                         np.sin(elev)])
    _, _, rot = look_at(target + view_dir, target)
    distance = fit_distance(rot, target, corners, args.fovy,
                            args.width / args.height, args.margin)
    pos, quat, _ = look_at(target + view_dir * distance, target)

    camera = scene.add_camera(name="catalog_cam", width=args.width, height=args.height,
                              fovy=args.fovy, near=0.01, far=distance * 4.0)
    camera.set_local_pose(sapien.Pose(p=pos, q=quat))
    rows = int(np.ceil(len(assets) / args.cols))
    print(f"[grid] {rows}x{args.cols} shared cells, {width:.2f}x{height:.2f} m")
    print(f"[camera] elev={args.elev}° azim={args.azim}° fovy={args.fovy} "
          f"distance={distance:.2f} m")

    scene.update_render()
    camera.take_picture()
    rgb = np.asarray(camera.get_picture("Color"))[..., :3]
    # Segmentation channel 1 is the per-entity id; 0 is empty space, which is
    # what gets painted white. Compositing beats clearing to white because it
    # leaves no renderer background bleeding through at the silhouettes.
    hit = np.asarray(camera.get_picture("Segmentation"))[..., 1] > 0
    image = np.where(hit[..., None], np.clip(rgb, 0.0, 1.0) * 255.0, 255.0).astype(np.uint8)
    return image, hit, assets


def crop_to_content(image, hit, pad):
    """Trim the uniform white border down to `pad` pixels around the assets."""
    rows = np.flatnonzero(hit.any(axis=1))
    cols = np.flatnonzero(hit.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return image
    r0 = max(int(rows[0]) - pad, 0)
    r1 = min(int(rows[-1]) + pad + 1, image.shape[0])
    c0 = max(int(cols[0]) - pad, 0)
    c1 = min(int(cols[-1]) + pad + 1, image.shape[1])
    return image[r0:r1, c0:c1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output PNG (default: {DEFAULT_OUT})")
    parser.add_argument("--cols", type=int, default=7,
                        help="columns in the shared grid (42 assets / 7 = 6 rows)")
    parser.add_argument("--width", type=int, default=3200, help="render width in px")
    parser.add_argument("--height", type=int, default=2600, help="render height in px")
    parser.add_argument("--pad-frac", type=float, default=0.14,
                        help="spacing between neighbours, as a fraction of cell content")
    parser.add_argument("--elev", type=float, default=46.0,
                        help="camera elevation in degrees (90 = straight down)")
    parser.add_argument("--azim", type=float, default=-80.0,
                        help="camera azimuth in degrees")
    parser.add_argument("--fovy", type=float, default=0.30,
                        help="vertical field of view in radians; small = near-orthographic")
    parser.add_argument("--margin", type=float, default=1.02,
                        help="framing slack on the fitted camera distance")
    parser.add_argument("--pad", type=int, default=24,
                        help="white border kept around the content, in px")
    args = parser.parse_args()

    import imageio.v3 as iio

    image, hit, assets = render_catalog(args)
    image = crop_to_content(image, hit, args.pad)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, image)

    for idx, (subdir, _, entry, *_) in enumerate(assets):
        r, c = divmod(idx, args.cols)
        print(f"    r{r}c{c}  {subdir:12s} {entry['name']}")
    print(f"[out] {args.out}  {image.shape[1]}x{image.shape[0]}")


if __name__ == "__main__":
    main()
