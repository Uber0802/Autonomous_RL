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


# Keys that only `scripts/plot.py` acts on. Accepted and ignored here so ONE
# config file drives all three plot tools instead of each needing its own.
_PLOT_PY_TOP_KEYS = {
    "smoothing_window", "n_interp_points", "end_steps", "end_resets",
    "figsize", "eval_kinds", "x_axes",
}
_PLOT_PY_GROUP_KEYS = {"color", "cronos_group_filter", "task_filter"}

# Tool options that may be set in the config instead of on the command line, so
# a comparison is reproducible from one file. Each is the snake_case form of the
# matching CLI flag, and the CLI always wins when both are given. Keys a given
# tool does not understand are simply ignored, exactly like the `scripts/plot.py`
# keys above — one config can carry settings for all three tools.
TOOL_OPTION_KEYS = {
    "actor_kind", "phase", "workspace_scale",   # plot_segment_positions.py
    "direction", "by", "metric", "smooth",      # plot_rollout_success.py
}


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
    # Tool options carried in the config (see TOOL_OPTION_KEYS). Read via
    # `option()`, which lets the CLI override.
    options: dict = field(default_factory=dict)

    def option(self, key, cli_value, default=None):
        """CLI value if given, else the config's, else `default`."""
        if cli_value is not None:
            return cli_value
        return self.options.get(key, default)


def resolve_out_dir(raw_value, config_path) -> Path:
    """Where a config's outputs go — one rule for every plot tool.

    - set        -> resolved to an absolute path, so the result does not depend
                    on the directory the tool happened to be launched from.
    - empty/absent -> **next to the config file**, not the CWD. `scripts/plot.py`
                    used `Path("").resolve()`, i.e. the CWD, and the shipped
                    `plot_config.json` ships `"out_dir": ""` — so figures landed
                    wherever the shell was, which is neither predictable nor
                    discoverable. The config's own directory is both.

    Every caller writes `<out_dir>/<name>_*`, so several configs can share one
    directory without colliding.
    """
    config_path = Path(config_path)
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return config_path.resolve().parent


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
    unknown = ({k for k in raw if not k.startswith("_")}
               - known - _PLOT_PY_TOP_KEYS - TOOL_OPTION_KEYS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}; valid: "
                         f"{sorted(known | _PLOT_PY_TOP_KEYS | TOOL_OPTION_KEYS)} "
                         f"(keys starting with '_' are ignored as comments)")

    groups = []
    for i, g in enumerate(raw.get("groups", [])):
        if not isinstance(g, dict):
            raise ValueError(f"groups[{i}] must be an object")
        g_unknown = set(g) - {"label", "runs", "csv_paths"} - _PLOT_PY_GROUP_KEYS
        if g_unknown:
            raise ValueError(f"groups[{i}] has unknown keys: {sorted(g_unknown)}")
        label = g.get("label") or f"group_{i}"
        if "runs" in g and "csv_paths" in g:
            raise ValueError(
                f"groups[{i}] ('{label}') sets both `runs` and `csv_paths`; use "
                f"one (prefer `runs`, which points at the glob dir)")
        # `csv_paths` is `scripts/plot.py`'s original spelling: paths to
        # individual `eval_success.csv` files. A CSV only names one of the three
        # files these tools read, so it is resolved back to its containing glob
        # dir — which is what `runs` states directly, and why `runs` is the
        # preferred form.
        key = "runs" if "runs" in g else "csv_paths"
        to_dir = (lambda s: Path(s)) if key == "runs" else (lambda s: Path(s).parent)
        entries = g.get(key, [])
        if not entries:
            raise ValueError(f"groups[{i}] ('{label}') has no {key}")
        chains = []
        for j, entry in enumerate(entries):
            if isinstance(entry, str):
                chains.append([to_dir(entry)])
            elif isinstance(entry, list) and all(isinstance(x, str) for x in entry):
                if not entry:
                    raise ValueError(f"groups[{i}].{key}[{j}] is an empty chain")
                chains.append([to_dir(x) for x in entry])
            else:
                raise ValueError(
                    f"groups[{i}].{key}[{j}] must be a path string or a list of "
                    f"them (a resume chain), got {type(entry).__name__}")
        groups.append(Group(label=label, chains=chains))

    if not groups:
        raise ValueError("config has no groups")
    return PlotConfig(name=raw.get("name", path.stem),
                      out_dir=resolve_out_dir(raw.get("out_dir"), path),
                      groups=groups,
                      options={k: raw[k] for k in TOOL_OPTION_KEYS if k in raw})


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
