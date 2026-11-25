"""Audit a checkpoint tree for cross-stack loadability, before a long run starts.

Why this exists
---------------
CRONOS checkpoints are written by whichever env trained them, and the two envs
do not agree on what `adapter_config.json` contains (see
`SimplerEnv/simpler_env/policies/peft_compat.py` for the full story):

- `tf447` — transformers 4.47 / peft 0.14 — writes three `LoraConfig` fields
  that `tf440` — transformers 4.40.1 / peft 0.11.1 — has no field for.
- `LoraConfig` is a dataclass, so peft 0.11.1 does not ignore them; it raises
  `TypeError: LoraConfig.__init__() got an unexpected keyword argument
  'eva_config'` and the checkpoint is simply unreadable.

`load_peft_adapter` repairs that at load time, but only for keys whose saved
value is provably inert. A key that is genuinely *set* still has to fail, and
finding that out 40 minutes into a sequential eval over 197 checkpoints is a bad
way to find it out. This script answers the same question up front, in a second,
without loading a model — or peft, or torch.

It is read-only: it opens `adapter_config.json` files and nothing else.

Usage
-----
    python tools/check_ckpt_compat.py <dir> [<dir> ...]
    python tools/check_ckpt_compat.py runs/ --target tf440    # the strict reader
    python tools/check_ckpt_compat.py runs/ --verbose         # list every checkpoint

Exit status is 0 when every checkpoint can be loaded by the target stack, 1
otherwise — so it also works as a preflight step in a shell script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPAT_PY = ROOT.parent / "SimplerEnv" / "simpler_env" / "policies" / "peft_compat.py"

# Loaded by path, not imported: this runs in any interpreter, including one with
# no CRONOS env activated and no `simpler_env` on sys.path.
_spec = importlib.util.spec_from_file_location("cronos_peft_compat", COMPAT_PY)
peft_compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(peft_compat)

# `LoraConfig` field names per stack. tf440 is peft 0.11.1 — the strict reader,
# and the only one that can reject anything. tf447 is peft 0.14.0, which is
# 0.11.1's fields plus the three additions; it reads every checkpoint CRONOS
# writes, and is listed here so `--target tf447` can state that positively
# rather than by omission.
PEFT_011_FIELDS = {
    "peft_type", "auto_mapping", "base_model_name_or_path", "revision", "task_type",
    "inference_mode", "r", "target_modules", "lora_alpha", "lora_dropout", "fan_in_fan_out",
    "bias", "use_rslora", "modules_to_save", "init_lora_weights", "layers_to_transform",
    "layers_pattern", "rank_pattern", "alpha_pattern", "megatron_config", "megatron_core",
    "loftq_config", "use_dora", "layer_replication",
}
TARGETS = {
    "tf440": PEFT_011_FIELDS,
    "tf447": PEFT_011_FIELDS | set(peft_compat.INERT_FORWARD_KEYS),
}

OK, REPAIRED, BLOCKED, UNREADABLE = "ok", "repaired", "BLOCKED", "UNREADABLE"


def classify(config_file: Path, accepted: set[str]):
    """Return (status, detail) for one adapter_config.json against one stack."""
    try:
        raw = json.loads(config_file.read_text())
    except (OSError, ValueError) as e:
        return UNREADABLE, str(e)

    try:
        _, dropped = peft_compat.split_forward_keys(raw, accepted)
    except peft_compat.AdapterConfigCompatError as e:
        # Keep the part that names the offending keys; the "Fix by one of" tail
        # is the same for every checkpoint and is printed once in the summary.
        detail = str(e).split("Fix by one of:")[0].strip()
        return BLOCKED, " ".join(detail.split())

    if dropped:
        return REPAIRED, ", ".join(f"{k}={v!r}" for k, v in sorted(dropped.items()))
    return OK, ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("roots", nargs="+", type=Path,
                    help="directories to search recursively for adapter_config.json")
    ap.add_argument("--target", choices=sorted(TARGETS), default="tf440",
                    help="stack that will read these checkpoints (default: tf440, the "
                         "strict one — transformers 4.40.1 / peft 0.11.1)")
    ap.add_argument("--verbose", action="store_true",
                    help="print a line per checkpoint, not just the ones needing repair")
    args = ap.parse_args()

    accepted = TARGETS[args.target]

    configs = sorted(
        {c for root in args.roots for c in root.rglob(peft_compat.CONFIG_NAME)}
    )
    if not configs:
        print(f"No {peft_compat.CONFIG_NAME} found under "
              f"{', '.join(str(r) for r in args.roots)}", file=sys.stderr)
        return 1

    counts = Counter()
    blocked = []
    for config_file in configs:
        status, detail = classify(config_file, accepted)
        counts[status] += 1
        if status in (BLOCKED, UNREADABLE):
            blocked.append((config_file, status, detail))
        # "repaired" is the expected steady state once load_peft_adapter is in
        # place, so it is a count, not 103 lines of scrollback. The failures are
        # reprinted below, where they cannot scroll away.
        if args.verbose:
            print(f"{status:10s} {config_file}" + (f": {detail}" if detail else ""))

    print(f"{len(configs)} checkpoint(s) under target {args.target}: "
          f"{counts[OK]} native, {counts[REPAIRED]} loadable after dropping inert "
          f"forward keys, {counts[BLOCKED]} blocked, {counts[UNREADABLE]} unreadable.")

    if blocked:
        print()
        for config_file, status, detail in blocked:
            print(f"{status:10s} {config_file}: {detail}")
        print(f"\n{len(blocked)} checkpoint(s) cannot be loaded by {args.target}. "
              f"Evaluate them in the stack that wrote them (README.md "
              f"§'Checkpoint portability across envs'), or vet the offending key "
              f"into INERT_FORWARD_KEYS in {COMPAT_PY.name}.")
        return 1

    print("All checkpoints load.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
