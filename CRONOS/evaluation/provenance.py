"""Which config a checkpoint was trained with, and whether eval uses the same scenes.

Standalone eval reads its environment — groups, objects, receptacles,
backgrounds, task pool, fan-out — from the *training* config file. Resolution
order for the file:

  1. `--config-path` on the command line
  2. the snapshot training wrote next to the checkpoint (`experiment_config.yaml`,
     written by `main.py` since this change)
  3. the `config_path` recorded in the checkpoint's `run_config.yaml` /
     `run_config.json`, resolved against the CRONOS directory

Whatever file is used, its scene definition is compared with the training one
(snapshot if present, else the recorded path). A difference is an error unless
`--allow-config-mismatch` is given. Without a snapshot the recorded file is
compared as it is *now*, which cannot detect edits made after training; the
provenance record says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

SNAPSHOT_NAME = "experiment_config.yaml"

# Everything that defines the environment eval rebuilds. `eval:`, policy/VLA
# overrides and `cronos_version` are deliberately not compared.
_TOP_FIELDS = ("fan_out", "scene", "task_order", "env_n", "env_m",
               "obj1_index", "obj2_index", "obj3_index",
               "plate1_index", "plate2_index", "plate3_index")
_GROUP_FIELDS = ("name", "num_envs", "obj", "recep", "table", "background",
                 "task_sequence", "eval_tasks")


def _read_run_config(ckpt: Path) -> Optional[dict]:
    for p in (ckpt / "run_config.yaml", ckpt.parent / "run_config.yaml"):
        if p.exists():
            try:
                import yaml
                data = yaml.safe_load(p.read_text())
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    for p in (ckpt / "run_config.json", ckpt.parent / "run_config.json"):
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None


def find_training_config(ckpt_dir: str, cronos_root: Path) -> dict:
    """{snapshot, recorded, recorded_resolved} for a checkpoint dir (values may be None)."""
    info = dict(snapshot=None, recorded=None, recorded_resolved=None)
    if not ckpt_dir:
        return info
    ckpt = Path(ckpt_dir)
    for p in (ckpt / SNAPSHOT_NAME, ckpt.parent / SNAPSHOT_NAME):
        if p.exists():
            info["snapshot"] = str(p)
            break
    rc = _read_run_config(ckpt)
    rec = (rc or {}).get("config_path") or None
    if rec:
        info["recorded"] = rec
        for cand in (Path(rec), Path(cronos_root) / rec, Path.cwd() / rec):
            if cand.exists():
                info["recorded_resolved"] = str(cand.resolve())
                break
    return info


def resolve_config_path(cli_path: str, ckpt_dir: str, cronos_root: Path) -> Tuple[str, dict]:
    info = find_training_config(ckpt_dir, cronos_root)
    if cli_path:
        path, source = cli_path, "cli"
    elif info["snapshot"]:
        path, source = info["snapshot"], "checkpoint snapshot"
    elif info["recorded_resolved"]:
        path, source = info["recorded_resolved"], "training run_config config_path"
    else:
        raise ValueError(
            "no --config-path given and the checkpoint does not identify its training config "
            f"(looked for {SNAPSHOT_NAME} and run_config.yaml/json config_path under {ckpt_dir!r}). "
            "Pass the training config with --config-path.")
    return path, dict(config_path=str(path), config_source=source, checkpoint=ckpt_dir, **info)


def scene_definition(cfg) -> dict:
    return dict(
        top={k: getattr(cfg, k, None) for k in _TOP_FIELDS},
        groups=[{k: getattr(g, k, None) for k in _GROUP_FIELDS} for g in cfg.groups],
    )


def diff_scene_definitions(a: dict, b: dict) -> List[str]:
    out = []
    for k in _TOP_FIELDS:
        if a["top"].get(k) != b["top"].get(k):
            out.append(f"{k}: eval={a['top'].get(k)!r} training={b['top'].get(k)!r}")
    if len(a["groups"]) != len(b["groups"]):
        out.append(f"number of groups: eval={len(a['groups'])} training={len(b['groups'])}")
    for i, (ga, gb) in enumerate(zip(a["groups"], b["groups"])):
        for k in _GROUP_FIELDS:
            if ga.get(k) != gb.get(k):
                out.append(f"groups[{i}].{k}: eval={ga.get(k)!r} training={gb.get(k)!r}")
    return out


def check_against_training(eval_cfg, provenance: dict, load_fn, allow_mismatch: bool) -> dict:
    """Compare `eval_cfg` with the training config; update and return `provenance`.

    Raises ValueError on a difference unless `allow_mismatch`.
    """
    ref = provenance.get("snapshot") or provenance.get("recorded_resolved")
    if ref is None:
        provenance["check"] = "skipped: checkpoint does not identify its training config"
        return provenance
    same_file = Path(ref).resolve() == Path(provenance["config_path"]).resolve()
    if same_file and provenance.get("snapshot"):
        provenance["check"] = "identical: evaluating the training snapshot itself"
        return provenance
    diffs = diff_scene_definitions(scene_definition(eval_cfg), scene_definition(load_fn(ref)))
    unverified = "" if provenance.get("snapshot") else \
        " (no snapshot: compared with the recorded file as it is now)"
    provenance["reference"] = ref
    provenance["differences"] = diffs
    if not diffs:
        provenance["check"] = "match" + unverified
        return provenance
    provenance["check"] = f"MISMATCH ({len(diffs)} difference(s)){unverified}"
    if not allow_mismatch:
        raise ValueError(
            "eval config does not define the same environment as the checkpoint's training config "
            f"{ref}:\n  " + "\n  ".join(diffs) +
            "\nEval uses the training config by default — drop --config-path, or pass "
            "--allow-config-mismatch to evaluate on a different environment on purpose.")
    return provenance
