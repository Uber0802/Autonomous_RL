"""CRONOS — YAML experiment config loader.

A single YAML file fully describes an experiment: per-group object/receptacle
indices, task sequences with symbolic refs, eval tasks, and global task ordering.

Three task ordering modes:
- ``sequential``:       follow task_sequence in order, repeat when exhausted
- ``pure_random``:      sample randomly each segment (≠ previous), weighted by
                        frequency in task_sequence
- ``sequence_random``:  random permutation covering all unique tasks, then reshuffle

Example config::

    cronos_version: V0.4
    task_order: sequential
    num_envs: 64

    groups:
      - name: "default_scene"
        obj: [7, 2]
        recep: [1, 2]
        task_sequence:
          - "put obj1 on recep1"
          - "put obj1 on recep2"
          - "put obj2 on recep1"
          - "put obj2 on recep2"
        eval_tasks:
          - "put obj1 on recep1"
          - "put obj2 on recep2"
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TASK_ORDERS = {"sequential", "pure_random", "sequence_random"}

_KNOWN_TOP_KEYS = {
    "cronos_version",
    "task_order",
    "num_envs",
    "scene",
    "groups",
    "fan_out",  # sub-group fan-out (default True)
    "unsuitable_detector",  # HSR detector config (name + workspace AABB)
    # Legacy flat keys (still accepted for backward compat / CLI-only mode)
    "env_n", "env_m",
    "obj1_index", "obj2_index", "obj3_index",
    "plate1_index", "plate2_index", "plate3_index",
    "task_filter",
    # policy / VLA overrides (mirrored into Args by main.py's YAML→Args loop).
    "policy", "vla_path", "vla_unnorm_key", "vla_lr", "vla_vhlr",
}

_KNOWN_GROUP_KEYS = {
    "name", "num_envs", "obj", "recep", "table", "background",
    "task_sequence", "eval_tasks",
    # Legacy single-task format
    "task",
}

_SYMBOLIC_RE = re.compile(r'\b(obj\d+|recep\d+)\b')


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GroupSpec:
    """One env group from the YAML ``groups:`` list."""
    name: str = ""
    num_envs: int = 0                                      # envs allocated to this group
    obj: List[int] = field(default_factory=list)           # 1-based object indices
    recep: List[int] = field(default_factory=list)         # 1-based receptacle indices
    table: str = "flat_table"
    background: Union[str, int] = "default"  # "default" = episode_id, int = fixed overlay index
    task_sequence: List[str] = field(default_factory=list) # symbolic or resolved
    eval_tasks: List[str] = field(default_factory=list)    # symbolic or resolved


@dataclass
class CronosConfig:
    """Parsed + validated experiment config."""
    task_order: str = "sequential"
    num_envs: Optional[int] = None
    scene: Optional[str] = None
    fan_out: bool = True              # sub-group fan-out (default True)
    groups: List[GroupSpec] = field(default_factory=list)
    unsuitable_detector: Optional[dict] = None   # raw YAML block, parsed in wrapper

    # Legacy fields (populated when using old flat format or CLI-only)
    env_n: Optional[int] = None
    env_m: Optional[int] = None
    obj1_index: Optional[int] = None
    obj2_index: Optional[int] = None
    obj3_index: Optional[int] = None
    plate1_index: Optional[int] = None
    plate2_index: Optional[int] = None
    plate3_index: Optional[int] = None
    task_filter: Optional[List[Union[int, str]]] = None

    # policy / VLA overrides — optional in YAML, propagated to `Args` by
    # main.py's YAML→Args loop (CLI still wins). Lets a training config like
    # `configs/spatialvla_2x2_train.yaml` pin the SpatialVLA policy / model
    # path / `bridge_orig/1.0.0` unnorm key without forcing every command line
    # to re-state them.
    policy: Optional[str] = None              # "openvla" | "spatialvla"
    vla_path: Optional[str] = None
    vla_unnorm_key: Optional[str] = None
    vla_lr: Optional[float] = None            # LoRA AdamW lr (per-policy split)
    vla_vhlr: Optional[float] = None          # value-head AdamW lr


# ---------------------------------------------------------------------------
# Symbolic task resolution
# ---------------------------------------------------------------------------

def resolve_symbolic_task(
    task_str: str,
    obj_names: Dict[str, str],
    recep_names: Dict[str, str],
) -> str:
    """Replace ``obj1``, ``recep2`` etc. with real object/receptacle names."""
    result = task_str
    for key in sorted(obj_names.keys(), key=len, reverse=True):
        result = result.replace(key, obj_names[key])
    for key in sorted(recep_names.keys(), key=len, reverse=True):
        result = result.replace(key, recep_names[key])
    return result


def has_symbolic_refs(s: str) -> bool:
    """Check if a string contains symbolic refs like obj1, recep2."""
    return bool(_SYMBOLIC_RE.search(s))


def build_obj_recep_name_maps(obj_indices: List[int], recep_indices: List[int],
                               model_db_carrot: dict, model_db_plate: dict
                               ) -> tuple:
    """Build {obj1: name, ...} and {recep1: name, ...} from indices + model_db."""
    obj_names = {}
    for i, idx_1based in enumerate(obj_indices):
        idx_0 = idx_1based - 1
        keys = list(model_db_carrot.keys())
        if 0 <= idx_0 < len(keys):
            obj_names[f"obj{i+1}"] = model_db_carrot[keys[idx_0]]["name"]
    recep_names = {}
    for i, idx_1based in enumerate(recep_indices):
        idx_0 = idx_1based - 1
        keys = list(model_db_plate.keys())
        if 0 <= idx_0 < len(keys):
            recep_names[f"recep{i+1}"] = model_db_plate[keys[idx_0]]["name"]
    return obj_names, recep_names


def get_group_starts(groups: List[GroupSpec]) -> List[int]:
    """Return cumulative env start indices: [0, g0.num_envs, g0+g1, ...]."""
    starts = [0]
    for g in groups:
        starts.append(starts[-1] + g.num_envs)
    return starts


def auto_generate_task_sequence(n_obj: int, n_recep: int) -> List[str]:
    """Generate all NxM task combinations in canonical order."""
    tasks = []
    for i in range(n_obj):
        for j in range(n_recep):
            tasks.append(f"put obj{i+1} on recep{j+1}")
    return tasks


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_config(config: CronosConfig, total_segments: Optional[int] = None):
    """Run all validation checks. Raises ValueError on failure."""
    groups = config.groups
    num_envs = config.num_envs

    # V1: at least one group
    if len(groups) == 0:
        raise ValueError("V1: Config must have at least one group")

    # V19: valid task_order
    if config.task_order not in VALID_TASK_ORDERS:
        raise ValueError(
            f"V19: Unknown task_order '{config.task_order}'. "
            f"Must be one of: {sorted(VALID_TASK_ORDERS)}")

    # V2: every group must have num_envs >= 1
    for i, g in enumerate(groups):
        if g.num_envs < 1:
            raise ValueError(
                f"V2: groups[{i}] ('{g.name}') has num_envs={g.num_envs}. "
                f"Each group must have at least 1 env.")

    # V3: sum of per-group num_envs must equal config.num_envs (if set)
    group_total = sum(g.num_envs for g in groups)
    if num_envs is not None and group_total != num_envs:
        raise ValueError(
            f"V3: sum of per-group num_envs ({group_total}) != config.num_envs ({num_envs}). "
            f"Set num_envs: {group_total} or adjust per-group num_envs.")

    for i, g in enumerate(groups):
        # V4/V5: obj and recep non-empty
        if len(g.obj) == 0:
            raise ValueError(f"V4: groups[{i}] ('{g.name}') must have at least one obj index")
        if len(g.recep) == 0:
            raise ValueError(f"V5: groups[{i}] ('{g.name}') must have at least one recep index")

        # V6: indices >= 1
        for j, v in enumerate(g.obj):
            if v < 1:
                raise ValueError(f"V6: groups[{i}].obj[{j}] = {v}: indices must be >= 1 (1-based)")
        for j, v in enumerate(g.recep):
            if v < 1:
                raise ValueError(f"V6: groups[{i}].recep[{j}] = {v}: indices must be >= 1 (1-based)")

        # V7/V8: no duplicates
        if len(g.obj) != len(set(g.obj)):
            raise ValueError(f"V7: groups[{i}].obj has duplicate indices: {g.obj}")
        if len(g.recep) != len(set(g.recep)):
            raise ValueError(f"V8: groups[{i}].recep has duplicate indices: {g.recep}")

        # V12: symbolic refs valid for group's N, M
        n_obj, n_recep = len(g.obj), len(g.recep)
        for t in g.task_sequence + g.eval_tasks:
            for m in _SYMBOLIC_RE.finditer(t):
                ref = m.group(0)
                if ref.startswith("obj"):
                    idx = int(ref[3:])
                    if idx < 1 or idx > n_obj:
                        raise ValueError(
                            f"V12: groups[{i}] task '{t}' references {ref} "
                            f"but group only has {n_obj} objects (obj1..obj{n_obj})")
                elif ref.startswith("recep"):
                    idx = int(ref[5:])
                    if idx < 1 or idx > n_recep:
                        raise ValueError(
                            f"V12: groups[{i}] task '{t}' references {ref} "
                            f"but group only has {n_recep} receptacles (recep1..recep{n_recep})")

        # V13: task format
        for t in g.task_sequence + g.eval_tasks:
            if not re.match(r'^put .+ on .+$', t):
                raise ValueError(
                    f"V13: groups[{i}] task '{t}' doesn't match 'put ... on ...' format")

        # V15: pure_random with pool size 1
        if config.task_order == "pure_random":
            unique_tasks = set(g.task_sequence) if g.task_sequence else set()
            if len(unique_tasks) == 1:
                warnings.warn(
                    f"V15: groups[{i}] ('{g.name}') has only 1 unique task — "
                    f"pure_random has no effect")

    # V14: sequence length divides total_segments
    if total_segments is not None and config.task_order in ("sequential", "sequence_random"):
        for i, g in enumerate(groups):
            seq_len = len(g.task_sequence)
            if seq_len > 0:
                pool_len = seq_len if config.task_order == "sequential" else len(set(g.task_sequence))
                if total_segments % pool_len != 0:
                    divisors = [d for d in range(1, total_segments + 1) if total_segments % d == 0]
                    raise ValueError(
                        f"V14: groups[{i}] task pool size {pool_len} does not divide "
                        f"total_segments {total_segments}. Valid divisors: {divisors}")

    # V22/V23: fan_out validation — group_envs must accommodate all unique tasks
    if config.fan_out and len(groups) > 0:
        for i, g in enumerate(groups):
            n_unique = len(set(g.task_sequence)) if g.task_sequence else 1
            if n_unique > 1:
                if g.num_envs < n_unique:
                    raise ValueError(
                        f"V22: groups[{i}] has {n_unique} unique tasks but only "
                        f"{g.num_envs} envs. fan_out requires group_envs >= unique_tasks.")
                if g.num_envs % n_unique != 0:
                    raise ValueError(
                        f"V23: groups[{i}] has {n_unique} unique tasks and "
                        f"{g.num_envs} envs. fan_out requires group_envs % unique_tasks == 0.")

    # V16: sequential mode — warn if groups have unequal sequence lengths.
    # This is a warning (not error) because the scheduler handles per-group
    # cursors independently.
    if config.task_order == "sequential" and len(groups) > 1:
        lengths = [len(g.task_sequence) for g in groups if g.task_sequence]
        if lengths and len(set(lengths)) > 1:
            warnings.warn(
                f"V16: sequential mode with unequal task_sequence lengths "
                f"across groups: {lengths}. Groups will desynchronize.")

    # V17: sequence_random — warn if groups have unequal pool sizes
    if config.task_order == "sequence_random" and len(groups) > 1:
        pool_sizes = [len(set(g.task_sequence)) for g in groups if g.task_sequence]
        if pool_sizes and len(set(pool_sizes)) > 1:
            warnings.warn(
                f"V17: sequence_random with unequal unique pool sizes "
                f"across groups: {pool_sizes}. Groups will desynchronize.")

    # V24/V25: eval task divisibility constraints
    for i, g in enumerate(groups):
        n_eval = len(g.eval_tasks) if g.eval_tasks else 0
        n_train = len(set(g.task_sequence)) if g.task_sequence else 1
        if n_eval > 0:
            if g.num_envs % n_eval != 0:
                raise ValueError(
                    f"V24: groups[{i}] has {n_eval} eval tasks but "
                    f"{g.num_envs} envs. Requires group_envs % n_eval_tasks == 0.")
            if n_eval % n_train != 0:
                raise ValueError(
                    f"V25: groups[{i}] has {n_eval} eval tasks and "
                    f"{n_train} unique train tasks. Requires n_eval_tasks % n_train_tasks == 0.")

    # V20: max_reset warning (caller passes this if available)
    # Handled externally in main.py since we don't have max_reset here.

    # V21: groups + task_filter mutually exclusive
    if groups and config.task_filter:
        raise ValueError(
            "V21: Config has both 'groups' and 'task_filter'. This is redundant — "
            "groups define your tasks directly. Remove task_filter.")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_cronos_config(path: Union[str, Path]) -> CronosConfig:
    """Load and validate a YAML experiment config.

    Supports both the new per-group format and the legacy flat format.
    """
    if yaml is None:
        raise ImportError("PyYAML is required for --config_path. pip install pyyaml")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")

    # Reject unknown top-level keys
    unknown = set(raw.keys()) - _KNOWN_TOP_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}. Valid: {sorted(_KNOWN_TOP_KEYS)}")

    # Parse groups
    groups = []
    raw_groups = raw.get("groups", [])
    for i, g in enumerate(raw_groups):
        if not isinstance(g, dict):
            raise ValueError(f"groups[{i}] must be a mapping, got {type(g).__name__}")

        # Check for unknown group keys
        g_unknown = set(g.keys()) - _KNOWN_GROUP_KEYS
        if g_unknown:
            raise ValueError(
                f"groups[{i}] has unknown keys: {g_unknown}. "
                f"Valid: {sorted(_KNOWN_GROUP_KEYS)}")

        # Legacy format: single "task" string → convert to new format
        if "task" in g and "task_sequence" not in g:
            # Old format: groups: [{task: "put obj1 on recep1"}, ...]
            # Infer obj/recep from global config or defaults
            obj = g.get("obj", [raw.get("obj1_index", 7), raw.get("obj2_index", 2)])
            recep = g.get("recep", [raw.get("plate1_index", 1), raw.get("plate2_index", 2)])
            groups.append(GroupSpec(
                name=g.get("name", f"group_{i}"),
                num_envs=g.get("num_envs", 0),
                obj=obj,
                recep=recep,
                table=g.get("table", "flat_table"),
                background=g.get("background", "default"),
                task_sequence=[g["task"]],
                eval_tasks=g.get("eval_tasks", [g["task"]]),
            ))
            continue

        # New format: per-group obj, recep, task_sequence
        obj = g.get("obj", [])
        recep = g.get("recep", [])

        task_sequence = g.get("task_sequence", [])
        if not task_sequence and obj and recep:
            # Auto-generate all NxM combinations
            task_sequence = auto_generate_task_sequence(len(obj), len(recep))

        eval_tasks = g.get("eval_tasks", [])
        if not eval_tasks:
            # Default: eval all unique tasks in the sequence
            seen = set()
            for t in task_sequence:
                if t not in seen:
                    eval_tasks.append(t)
                    seen.add(t)

        groups.append(GroupSpec(
            name=g.get("name", f"group_{i}"),
            num_envs=g.get("num_envs", 0),
            obj=obj,
            recep=recep,
            table=g.get("table", "flat_table"),
            background=g.get("background", "default"),
            task_sequence=task_sequence,
            eval_tasks=eval_tasks,
        ))

    # Resolve per-group num_envs: either from per-group fields or legacy top-level
    group_total = sum(g.num_envs for g in groups)
    top_num_envs = raw.get("num_envs")
    if group_total > 0:
        # Per-group num_envs specified — use their sum
        if top_num_envs is not None and top_num_envs != group_total:
            raise ValueError(
                f"Top-level num_envs ({top_num_envs}) != sum of per-group "
                f"num_envs ({group_total}). Remove top-level num_envs or fix the mismatch.")
        top_num_envs = group_total
    elif top_num_envs is not None and len(groups) > 0:
        # Legacy: top-level num_envs only — split equally among groups
        if top_num_envs % len(groups) != 0:
            raise ValueError(
                f"Top-level num_envs ({top_num_envs}) not divisible by "
                f"{len(groups)} groups. Use per-group num_envs instead.")
        per_group = top_num_envs // len(groups)
        for grp in groups:
            grp.num_envs = per_group

    # Parse task_filter (CLI-only mode, no groups)
    tf_raw = raw.get("task_filter")
    task_filter = None
    if tf_raw is not None:
        if not isinstance(tf_raw, list):
            raise ValueError(f"task_filter must be a list, got {type(tf_raw).__name__}")
        task_filter = tf_raw

    config = CronosConfig(
        task_order=raw.get("task_order", "sequential"),
        num_envs=top_num_envs,
        scene=raw.get("scene"),
        fan_out=raw.get("fan_out", True),
        groups=groups,
        unsuitable_detector=raw.get("unsuitable_detector"),
        # Legacy
        env_n=raw.get("env_n"),
        env_m=raw.get("env_m"),
        obj1_index=raw.get("obj1_index"),
        obj2_index=raw.get("obj2_index"),
        obj3_index=raw.get("obj3_index"),
        plate1_index=raw.get("plate1_index"),
        plate2_index=raw.get("plate2_index"),
        plate3_index=raw.get("plate3_index"),
        task_filter=task_filter,
        # policy / VLA overrides — populated only if the YAML provides
        # them. main.py's YAML→Args loop reads them with `not _cli_provided`
        # so a CLI flag still wins.
        policy=raw.get("policy"),
        vla_path=raw.get("vla_path"),
        vla_unnorm_key=raw.get("vla_unnorm_key"),
        vla_lr=raw.get("vla_lr"),
        vla_vhlr=raw.get("vla_vhlr"),
    )

    # Run validation (total_segments not known at load time; caller validates V14 later)
    validate_config(config)

    return config
