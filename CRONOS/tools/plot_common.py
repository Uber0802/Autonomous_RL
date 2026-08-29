"""Shared config loading for the per-run plot tools.

One JSON describes several experiment groups; each group holds several runs, and
a run may itself be a **resume chain** — a list of run dirs that are stitched
into a single continuous series. Same shape as `scripts/plot_config.json`'s
`csv_paths`, so the two config styles stay recognisable.

    {
      "out_dir": "reports/figures/2026-08-26",
      "name": "perturb_ablation",
      "groups": [
        {
          "label": "noep baseline",
          "runs": [
            "/data/runs/A/wandb/run-.../glob",
            ["/data/runs/B-parent/.../glob", "/data/runs/B-child/.../glob"]
          ]
        },
        { "label": "noep + PTBmixed", "runs": ["/data/runs/C/.../glob"] }
      ]
    }

`runs` entries:
  - a string  -> one run, one series
  - a list    -> a resume chain, concatenated in order into ONE series

Within a group each entry is a separate series (typically a seed); the plot
tools aggregate them into a mean ± spread band. Across groups you get one curve
or one panel each.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class Group:
    label: str
    # One entry per series; each entry is the ordered list of run dirs that make
    # up that series (length 1 for a run that was not resumed).
    chains: List[List[Path]] = field(default_factory=list)


@dataclass
class PlotConfig:
    name: str
    out_dir: Path
    groups: List[Group] = field(default_factory=list)


def load_plot_config(path) -> PlotConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a JSON object, got {type(raw).__name__}")

    known = {"out_dir", "name", "groups"}
    # JSON has no comment syntax, so any key starting with "_" is treated as one
    # and ignored — that is what `scripts/plot_runs_example.json` uses to carry
    # its own documentation.
    unknown = {k for k in raw if not k.startswith("_")} - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}; valid: {sorted(known)} "
                         f"(keys starting with '_' are ignored as comments)")

    groups = []
    for i, g in enumerate(raw.get("groups", [])):
        if not isinstance(g, dict):
            raise ValueError(f"groups[{i}] must be an object")
        g_unknown = set(g) - {"label", "runs"}
        if g_unknown:
            raise ValueError(f"groups[{i}] has unknown keys: {sorted(g_unknown)}")
        label = g.get("label") or f"group_{i}"
        entries = g.get("runs", [])
        if not entries:
            raise ValueError(f"groups[{i}] ('{label}') has no runs")
        chains = []
        for j, entry in enumerate(entries):
            if isinstance(entry, str):
                chains.append([Path(entry)])
            elif isinstance(entry, list) and all(isinstance(x, str) for x in entry):
                if not entry:
                    raise ValueError(f"groups[{i}].runs[{j}] is an empty chain")
                chains.append([Path(x) for x in entry])
            else:
                raise ValueError(
                    f"groups[{i}].runs[{j}] must be a run-dir string or a list of "
                    f"them (a resume chain), got {type(entry).__name__}")
        groups.append(Group(label=label, chains=chains))

    if not groups:
        raise ValueError("config has no groups")
    return PlotConfig(name=raw.get("name", path.stem),
                      out_dir=Path(raw.get("out_dir") or path.parent),
                      groups=groups)


def concat_chain(frames: List[pd.DataFrame], x_col: str = "total_steps") -> pd.DataFrame:
    """Stitch a resume chain into one series.

    A resumed run restates the parent's counters, so the child's x values can
    overlap the parent's. At the seam the CHILD wins — it is the run that
    actually produced those steps under the resumed configuration. Same rule
    `scripts/plot.py` applies to its `csv_paths` chains.
    """
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    kept = []
    for idx, f in enumerate(frames):
        later = frames[idx + 1:]
        if later:
            # Drop rows this run contributed that a later run re-covers.
            floor = min(g[x_col].min() for g in later if len(g))
            f = f[f[x_col] < floor]
        kept.append(f)
    return pd.concat(kept, ignore_index=True)


def default_colors(n: int):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def read_run_config(run_dir: Path) -> Optional[dict]:
    """Load a run's `run_config.json` (env_n / env_m / num_envs / obj_set / ...).

    Returns None when absent — an older run, or a glob dir assembled by hand.
    """
    p = Path(run_dir) / "run_config.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None
