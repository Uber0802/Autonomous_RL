"""CRONOS — pull selected frames out of a run's recorded videos.

A training run already records everything this asks for: `main.py` writes one
mp4 per (episode, segment, env) under ``glob/train_videos/rollout_ep{E}_seg{S}/
env{N}.mp4``. This script picks frames out of them by index, so a figure that
needs "env 6, segments 1 / 5 / 9, first frame" is one command rather than a
manual scrub through three videos::

    cd V0.93/CRONOS
    python tools/extract_run_frames.py <run> --envs 6 --segments 1,5,9 --frames first

No GPU, no simulator, no policy — it only decodes mp4s that already exist. That
also means it can only show what was recorded: to render a state that no run
produced, use `tools/render_segment_frames.py` instead.

**Episode and segment are 1-indexed**, matching the directory names
(``save_video_segment`` writes ``rollout_ep{iteration+1}_seg{segment_id+1}``).
Env indices are 0-based, as in the filenames.

**What "the first frame" actually is.** Training videos hold exactly `task_len`
frames and every one of them is a *post-step* observation (`main.py:1262-1268`,
matching AutoRL) — frame 0 is the state after the segment's first control step,
not the state the segment started from. The segment's true initial state is not
in the video at all: it is the observation the first action was computed from,
and it is deliberately not recorded, because at a segment boundary the camera
buffer is still stale relative to HSR's respawn and EER's `reset_robot()`. If
you need the state a segment *began* from, the closest recorded thing is the
previous segment's last frame (``--segments <S-1> --frames last``), which is
before that boundary rather than after it.

Eval videos are the other way round: `main.py:1042-1061` records pre-step
observations plus a final post-step one, so they hold `segment_len + 1` frames
and frame 0 *is* the initial state. Pass ``--kind eval`` for those.

Each extracted frame is annotated from ``glob/rollout_success.csv`` — the group,
task, direction and success of that exact (episode, segment, env) — so the
printed table says what is in the picture, and ``--sheet`` captions it.

Usage::

    cd V0.93/CRONOS
    # what does this run have?
    python tools/extract_run_frames.py <run> --list

    # the motivating case
    python tools/extract_run_frames.py <run> --envs 6 --segments 1,5,9 --frames first

    # first and last frame of every segment of episode 2, for three envs
    python tools/extract_run_frames.py <run> --episodes 2 --envs 0,6,30 \\
        --segments all --frames first,last --sheet figures/ep2_strip.png

    # explicit indices, negatives count from the end
    python tools/extract_run_frames.py <run> --envs 6 --segments 1 --frames 0,39,-1

``<run>`` may be the run directory, the ``wandb/run-*`` directory, or the
``glob/`` directory itself — whichever is convenient; the script walks down to
the ``glob/`` that holds ``train_videos``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

VIDEO_SUBDIR = {"train": "train_videos", "eval": "eval_videos"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("run", type=Path,
                   help="run directory, wandb/run-* directory, or glob/ directory")
    p.add_argument("--kind", choices=("train", "eval"), default="train",
                   help="which recording to read; see the module docstring for how "
                        "their frame indexing differs")
    p.add_argument("--episodes", default="all",
                   help="1-indexed, e.g. '2', '1,3', '1-4', 'all'")
    p.add_argument("--segments", default="all",
                   help="1-indexed, e.g. '1,5,9', '1-16', 'all'")
    p.add_argument("--envs", default="0",
                   help="0-indexed, e.g. '6', '0,6,30', '0-7', 'all'")
    p.add_argument("--frames", default="first",
                   help="'first', 'last', 'mid', 'all', or explicit indices like "
                        "'0,39,-1' (negatives count from the end)")
    p.add_argument("--eval-prefix", default=None,
                   help="--kind eval: which pass, e.g. 'in_domain' / "
                        "'out_of_domain' (default: every one present)")
    p.add_argument("--eval-ep", default="all",
                   help="--kind eval: which eval episode under the prefix, "
                        "1-indexed, e.g. '1' or 'all'")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="where to write the PNGs (default: <run>/glob/extracted_frames)")
    p.add_argument("--sheet", type=Path, default=None,
                   help="also write a labelled contact sheet here")
    p.add_argument("--list", action="store_true",
                   help="print what the run contains and exit without extracting")
    return p.parse_args()


def find_glob_dir(run: Path) -> Path:
    """Accept any of the three paths a person is likely to have on the clipboard.

    A run directory holds `wandb/run-<id>/glob/`, and the interesting files are
    only in the last of those. Requiring the full path would mean pasting an
    opaque wandb id every time, so walk down instead — and fail with the
    candidates listed when the walk is ambiguous, rather than silently picking
    one of several runs.
    """
    run = run.expanduser().resolve()
    if not run.exists():
        raise SystemExit(f"{run} does not exist")
    if (run / "train_videos").exists() or (run / "rollout_success.csv").exists():
        return run
    candidates = sorted(set(run.glob("glob")) | set(run.glob("wandb/run-*/glob")))
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        raise SystemExit(
            f"no glob/ directory under {run}. Pass the run directory, the "
            f"wandb/run-* directory, or the glob/ directory itself.")
    if len(candidates) > 1:
        # Several wandb runs under one run directory is normal after a resume;
        # which one is meant is not something to guess.
        listing = "\n".join(f"  {c}" for c in candidates)
        raise SystemExit(
            f"{run} contains {len(candidates)} runs — name one explicitly:\n{listing}")
    return candidates[0]


def parse_selector(spec: str, available: list[int], what: str) -> list[int]:
    """'all' | '1,5,9' | '1-16' | a mix, intersected with what exists.

    Selecting something the run does not have is an error rather than a silent
    omission: a figure quietly missing segment 9 because the run stopped at 8
    is worse than a failed command.
    """
    spec = spec.strip()
    if spec in ("all", "*"):
        return sorted(available)
    wanted: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok[1:]:  # not a leading minus
            lo, hi = tok.split("-", 1)
            wanted.extend(range(int(lo), int(hi) + 1))
        else:
            wanted.append(int(tok))
    missing = [w for w in wanted if w not in set(available)]
    if missing:
        rng = (f"{min(available)}..{max(available)}" if available else "none")
        raise SystemExit(f"{what} {missing} not in this run (available: {rng})")
    seen, out = set(), []
    for w in wanted:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def resolve_frames(spec: str, n_frames: int) -> list[int]:
    """Keywords and explicit indices -> sorted non-negative indices."""
    spec = spec.strip()
    if spec == "all":
        return list(range(n_frames))
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "first":
            out.append(0)
        elif tok == "last":
            out.append(n_frames - 1)
        elif tok == "mid":
            out.append(n_frames // 2)
        else:
            i = int(tok)
            if i < 0:
                i += n_frames
            if not 0 <= i < n_frames:
                raise SystemExit(f"frame {tok} out of range for a "
                                 f"{n_frames}-frame video")
            out.append(i)
    seen, uniq = set(), []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def scan_train(videos: Path) -> dict[tuple[int, int], set[int]]:
    """{(episode, segment): {env, ...}} from the directory names on disk."""
    found: dict[tuple[int, int], set[int]] = defaultdict(set)
    for d in videos.glob("rollout_ep*_seg*"):
        if not d.is_dir():
            continue
        try:
            ep_part, seg_part = d.name.split("_")[1:3]
            ep, seg = int(ep_part[2:]), int(seg_part[3:])
        except (IndexError, ValueError):
            print(f"[warn] skipping unrecognised directory name: {d.name}")
            continue
        for f in d.glob("env*.mp4"):
            try:
                found[ep, seg].add(int(f.stem[3:]))
            except ValueError:
                print(f"[warn] skipping unrecognised video name: {f.name}")
    return found


def scan_eval(videos: Path) -> dict[tuple[int, str, int], set[int]]:
    """{(outer episode, prefix, eval episode): {env, ...}}."""
    found: dict[tuple[int, str, int], set[int]] = defaultdict(set)
    for ep_dir in videos.glob("ep*"):
        if not ep_dir.is_dir():
            continue
        try:
            ep = int(ep_dir.name[2:])
        except ValueError:
            continue
        for prefix_dir in sorted(ep_dir.iterdir()):
            if not prefix_dir.is_dir():
                continue
            for sub in sorted(prefix_dir.glob("eval_ep*")):
                try:
                    k = int(sub.name[len("eval_ep"):])
                except ValueError:
                    continue
                for f in sub.glob("env*.mp4"):
                    try:
                        found[ep, prefix_dir.name, k].add(int(f.stem[3:]))
                    except ValueError:
                        pass
    return found


def load_rollout_tasks(glob_dir: Path) -> dict[tuple[int, int, int], dict]:
    """{(episode, segment, env): row} from rollout_success.csv, if it is there.

    Purely for annotation. A run whose CSV is missing or truncated still
    extracts frames; the label just says so.
    """
    path = glob_dir / "rollout_success.csv"
    if not path.exists():
        print(f"[warn] no rollout_success.csv in {glob_dir} — frames will be "
              f"extracted, but not labelled with their task")
        return {}
    rows: dict[tuple[int, int, int], dict] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                key = (int(row["episode"]), int(row["segment"]), int(row["env_idx"]))
            except (KeyError, ValueError):
                continue
            rows[key] = row
    return rows


def describe(row: dict | None) -> str:
    if not row:
        return "task unknown"
    bits = [row.get("group", "?"), row.get("task", "?")]
    if row.get("direction") and row["direction"] != "forward":
        bits.append(f"[{row['direction']}]")
    try:
        bits.append(f"success={float(row['success']):.0f}")
    except (KeyError, ValueError, TypeError):
        pass
    return " · ".join(b for b in bits if b)


def main() -> None:
    args = parse_args()
    glob_dir = find_glob_dir(args.run)
    videos = glob_dir / VIDEO_SUBDIR[args.kind]
    if not videos.exists():
        raise SystemExit(f"{videos} does not exist — was this run launched with "
                         f"--record-video?")
    print(f"[run] {glob_dir}")

    import imageio.v3 as iio

    tasks = load_rollout_tasks(glob_dir) if args.kind == "train" else {}

    # --- work out which videos to open ---------------------------------------
    # `jobs` is a flat list of (label parts, video path, key into `tasks`), so
    # the train and eval layouts converge here and everything downstream — frame
    # selection, naming, the sheet — is written once.
    jobs: list[tuple[str, dict, Path, tuple | None]] = []

    if args.kind == "train":
        found = scan_train(videos)
        if not found:
            raise SystemExit(f"no rollout_ep*_seg* directories under {videos}")
        episodes = parse_selector(args.episodes,
                                  sorted({ep for ep, _ in found}), "episodes")
        for ep in episodes:
            segs_here = sorted({s for e, s in found if e == ep})
            segments = parse_selector(args.segments, segs_here, f"segments (ep {ep})")
            for seg in segments:
                envs_here = sorted(found[ep, seg])
                envs = parse_selector(args.envs, envs_here, f"envs (ep {ep} seg {seg})")
                for env in envs:
                    jobs.append((
                        f"ep{ep:02d}__seg{seg:02d}__env{env:02d}",
                        {"ep": ep, "seg": seg, "env": env},
                        videos / f"rollout_ep{ep}_seg{seg}" / f"env{env}.mp4",
                        (ep, seg, env)))
    else:
        found = scan_eval(videos)
        if not found:
            raise SystemExit(f"no ep*/<prefix>/eval_ep* directories under {videos}")
        episodes = parse_selector(args.episodes,
                                  sorted({ep for ep, _, _ in found}), "episodes")
        prefixes = sorted({p for _, p, _ in found})
        if args.eval_prefix:
            if args.eval_prefix not in prefixes:
                raise SystemExit(f"--eval-prefix {args.eval_prefix!r} not in "
                                 f"{prefixes}")
            prefixes = [args.eval_prefix]
        for ep in episodes:
            for prefix in prefixes:
                ks = sorted({k for e, p, k in found if e == ep and p == prefix})
                if not ks:
                    continue
                for k in parse_selector(args.eval_ep, ks, f"--eval-ep (ep {ep})"):
                    envs_here = sorted(found[ep, prefix, k])
                    envs = parse_selector(args.envs, envs_here, "envs")
                    for env in envs:
                        jobs.append((
                            f"ep{ep:02d}__{prefix}__evalep{k:02d}__env{env:02d}",
                            {"ep": ep, "prefix": prefix, "eval_ep": k, "env": env},
                            videos / f"ep{ep}" / prefix / f"eval_ep{k}" / f"env{env}.mp4",
                            None))

    if args.list:
        if args.kind == "train":
            by_ep: dict[int, list[int]] = defaultdict(list)
            for ep, seg in found:
                by_ep[ep].append(seg)
            n_envs = {len(v) for v in found.values()}
            print(f"[list] {len(found)} (episode, segment) recordings, "
                  f"{sorted(n_envs)} envs each")
            for ep in sorted(by_ep):
                segs = sorted(by_ep[ep])
                print(f"        episode {ep:2d}: segments {segs[0]}-{segs[-1]} "
                      f"({len(segs)})")
        else:
            print(f"[list] {len(found)} eval recordings")
            for key in sorted(found):
                ep, prefix, k = key
                print(f"        ep{ep} / {prefix} / eval_ep{k}: "
                      f"{len(found[key])} envs")
        return

    if not jobs:
        raise SystemExit("selection matched no videos")

    out_dir = args.out_dir or glob_dir / "extracted_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- extract -------------------------------------------------------------
    rows_for_sheet: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)
    written = 0
    for label, meta, path, task_key in jobs:
        if not path.exists():
            raise SystemExit(f"{path} is missing, though its directory was scanned")
        clip = iio.imread(path)  # [T, H, W, 3]
        idxs = resolve_frames(args.frames, len(clip))
        note = describe(tasks.get(task_key)) if task_key else ""
        for i in idxs:
            name = f"{label}__f{i:03d}.png"
            iio.imwrite(out_dir / name, clip[i])
            written += 1
            print(f"[frame] {name}  ({len(clip)} frames)"
                  + (f"  {note}" if note else ""))
            rows_for_sheet[meta["env"]].append(
                (f"{label.split('__env')[0]} f{i:03d}", clip[i]))

    print(f"[frames] wrote {written} PNGs to {out_dir}")

    # --- optional contact sheet ---------------------------------------------
    if args.sheet:
        from render_episode_boundaries import label_sheet

        envs_sorted = sorted(rows_for_sheet)
        width = max(len(rows_for_sheet[e]) for e in envs_sorted)
        if any(len(rows_for_sheet[e]) != width for e in envs_sorted):
            raise SystemExit("--sheet needs the same number of frames per env; "
                             "the selection is ragged")
        grid = [[f for _, f in rows_for_sheet[e]] for e in envs_sorted]
        col_labels = [c for c, _ in rows_for_sheet[envs_sorted[0]]]
        row_labels = [f"env {e}\n{VIDEO_SUBDIR[args.kind]}" for e in envs_sorted]
        # Name the run by its RUN_TAG, not by the `wandb` directory that
        # happens to sit two levels up.
        run_name = next((q.name for q in glob_dir.parents
                         if q.name.startswith("CRONOS-")), glob_dir.parents[1].name)
        caption = (f"{run_name} · {glob_dir.parent.name} · "
                   f"{VIDEO_SUBDIR[args.kind]} · --frames {args.frames}")
        sheet = label_sheet(grid, row_labels, col_labels, caption=caption)
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.sheet, sheet)
        print(f"[sheet] {sheet.shape[1]}x{sheet.shape[0]} px → {args.sheet}")


if __name__ == "__main__":
    main()
