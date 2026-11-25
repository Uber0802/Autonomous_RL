"""Log a CUDA out-of-memory error with the context needed to act on it.

A bare `torch.cuda.OutOfMemoryError` says how many bytes one allocation wanted,
not where in the run it happened or what the rest of memory was doing. The
entry points (`main.py`, `eval_only.py`) catch it once at the top, call
`write_oom_report`, and re-raise, so the process still exits non-zero.

The report goes to stderr (summary) and to `<out_dir>/oom_report.txt` (full,
including `torch.cuda.memory_summary()` per device and the traceback).
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

# Args that decide peak memory, reported when present on the args object.
MEMORY_ARGS = (
    "policy", "vla_path", "vla_lora_rank", "num_envs", "buffer_inferbatch",
    "buffer_minibatch", "alg_gradient_accum", "alg_ppo_epoch", "alg_name",
    "segment_len", "episode_len", "task_len", "ppo_update_len",
    "num_eval_episode", "record_video", "action_chunk",
)


def is_oom(exc: BaseException) -> bool:
    """True for torch's OOM error, and for the RuntimeError older builds raise."""
    try:
        import torch
    except ImportError:
        return False
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _gb(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GB"


def _device_lines(torch):
    lines, summaries, hints = [], [], set()
    for i in range(torch.cuda.device_count()):
        try:
            name = torch.cuda.get_device_name(i)
            free, total = torch.cuda.mem_get_info(i)
            alloc = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            lines.append(
                f"  cuda:{i} {name}: total {_gb(total)}, free {_gb(free)}, "
                f"allocated {_gb(alloc)} (peak {_gb(torch.cuda.max_memory_allocated(i))}), "
                f"reserved {_gb(reserved)} (peak {_gb(torch.cuda.max_memory_reserved(i))})")
            if reserved - alloc > 2 * 1024 ** 3:
                hints.add("reserved exceeds allocated by >2 GB: fragmentation likely; try "
                          "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
            if total - free > reserved + 2 * 1024 ** 3:
                hints.add("the device holds >2 GB more than this process reserved: another "
                          "process is using the GPU (check nvidia-smi)")
            summaries.append(f"--- torch.cuda.memory_summary(cuda:{i}) ---\n"
                             f"{torch.cuda.memory_summary(i, abbreviated=True)}")
        except Exception as e:  # reporting must never mask the original error
            lines.append(f"  cuda:{i}: <could not query: {type(e).__name__}: {e}>")
    return lines, summaries, sorted(hints)


def write_oom_report(exc: BaseException, out_dir: Optional[Path], context: dict,
                     args=None) -> Optional[Path]:
    """Print an OOM summary to stderr and write the full report; never raises."""
    try:
        import torch
        ctx = dict(context)
        if args is not None:
            ctx["args"] = {k: getattr(args, k) for k in MEMORY_ARGS if hasattr(args, k)}
        ctx["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        ctx["torch"] = torch.__version__
        try:
            import transformers
            ctx["transformers"] = transformers.__version__
        except Exception:
            pass

        dev_lines, summaries, hints = _device_lines(torch)
        hints = hints + [
            "lower buffer_inferbatch (rollout inference) or buffer_minibatch (PPO update), "
            "whichever phase failed; raising alg_gradient_accum keeps the effective batch",
            "OpenVLA-7B PPO on a 48 GB GPU: use the tf440 env (see README, Installation)",
        ]
        summary = "\n".join(
            ["", "=" * 70,
             f"[OOM] CUDA out of memory at {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"  error: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}",
             "  where: " + ", ".join(f"{k}={v}" for k, v in context.items()),
             *dev_lines,
             "  hints:", *[f"    - {h}" for h in hints],
             "=" * 70])
        print(summary, file=sys.stderr, flush=True)

        if out_dir is None:
            return None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "oom_report.txt"
        body = [summary, "", "context:", json.dumps(ctx, indent=2, default=str), "",
                *summaries, "", "traceback:",
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))]
        path.write_text("\n".join(body))
        print(f"[OOM] full report: {path}", file=sys.stderr, flush=True)
        return path
    except Exception as e:
        print(f"[OOM] could not write report: {type(e).__name__}: {e}", file=sys.stderr)
        return None
