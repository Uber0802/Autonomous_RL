"""CRONOS — run output directory resolution.

`wandb.init(dir=X)` does **not** guarantee the run lands under `X`. Both wandb
generations silently fall back to the system temp directory instead of raising:

- wandb < ~0.17 (`wandb/old/core.py::wandb_dir`) never creates `X`. It calls
  ``os.access(root_dir, os.W_OK)``, which is False for a path that does not
  exist yet, and then rewrites the path to ``tempfile.gettempdir()/wandb/``.
  A relative, not-yet-created directory — exactly what `scripts/train.sh`
  passes — therefore *always* took this branch.
- wandb >= ~0.17 (`wandb/sdk/wandb_init.py::try_create_root_dir`) does
  ``os.makedirs(exist_ok=True)`` first, but still falls back to
  ``tempfile.gettempdir()`` on any `OSError` or if the result is not R+W.

Both only emit a `termwarn`, which is trivially lost in the SAPIEN /
transformers startup noise. The run then writes every CSV, checkpoint and video
into `/tmp`, where the next reboot removes them.

`setup.sh` installs `wandb` unpinned, so a given machine may have either
generation. This module makes the outcome version-independent:

- `prepare_wandb_dir` creates the directory and returns an **absolute** path, so
  neither the "doesn't exist" nor the "relative to a CWD that moved" failure
  mode can trigger.
- `verify_run_dir` checks *after* `wandb.init` that the run actually landed
  where it was asked to, and raises if wandb fell back anyway.

Shared by `main.py` and `eval_only.py`; both had the same unchecked
`wandb_kwargs["dir"] = args.wandb_dir` assignment.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional


def prepare_wandb_dir(wandb_dir: str) -> Optional[str]:
    """Resolve, create and validate the wandb root directory.

    Returns the absolute path to pass as `wandb.init(dir=...)`, or None when the
    caller did not request a directory (wandb then uses its own default, rooted
    at the CWD).

    Raises instead of letting wandb degrade to `/tmp`: a run that silently
    writes its checkpoints somewhere the user is not looking is worse than a run
    that refuses to start.
    """
    if not wandb_dir:
        return None

    path = Path(wandb_dir).expanduser()
    # Absolute: wandb resolves a relative root against the CWD, and both the
    # buffer mmap dir and `--config-path` already make CRONOS CWD-sensitive.
    # Pinning it here means the run directory does not move if anything later
    # chdir's.
    path = path.resolve()

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(
            f"--wandb-dir {wandb_dir!r} (resolved to {path}) could not be created: {e}. "
            f"wandb would silently fall back to {tempfile.gettempdir()} and the run's "
            f"CSVs, checkpoints and videos would be lost on reboot."
        ) from e

    if not os.access(path, os.W_OK | os.R_OK):
        raise PermissionError(
            f"--wandb-dir {wandb_dir!r} (resolved to {path}) is not readable+writable. "
            f"wandb would silently fall back to {tempfile.gettempdir()}."
        )

    return str(path)


def verify_run_dir(run_dir: Path | str, requested: Optional[str]) -> None:
    """Assert that `wandb.init` honoured `requested`.

    `run_dir` is `wandb.run.dir` (i.e. `<requested>/wandb/<run>/files`). If wandb
    fell back to the temp directory despite `prepare_wandb_dir` succeeding — a
    race, a full disk, a read-only remount between the two calls — this turns a
    warning nobody reads into a hard failure.

    No-op when no directory was requested.
    """
    if not requested:
        return

    run_dir = Path(run_dir).resolve()
    requested_path = Path(requested).resolve()
    if requested_path not in run_dir.parents:
        raise RuntimeError(
            f"wandb ignored dir={requested!r} and placed the run at {run_dir}. "
            f"Everything this run writes (CSVs, checkpoints, videos) would go there "
            f"instead of the requested directory. Check that {requested_path} is on a "
            f"writable, non-full filesystem."
        )
