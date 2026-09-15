"""Standalone-eval planning: settings, scenes, rounds, coverage, fingerprints.

Dependency-free on purpose (stdlib + `envs.rng_streams`), so everything that
decides *what* gets evaluated — and how many times each scene is run — can be
checked without a GPU. `evaluation.sequential` executes a plan and
`evaluation.outputs` turns its per-trial rows into every aggregate file.

Vocabulary
----------
scene      one YAML group: its objects, receptacles and background. A config
           without `groups:` has a single scene named "default".
order      a task ordering, labelled by training-order position letters
           (A = first task of the training order): "ABCD", "BCDA", ...
cycle      an ordering up to rotation: ABCD, BCDA, CDAB, DABC are the cycle
           A->B->C->D->A. n tasks have (n-1)! cycles x n rotations = n! orders.
block      a scene's envs split into n equal sub-blocks, one per rotation — the
           layout fan-out training uses. Each block runs its own order.
round      the unit of eval: one reset, then every task slot of the round without
           resets. Rounds are numbered globally and are the only selector
           (`rounds`); everything about a round is a function of its number.
pose set   R consecutive rounds that share one set of start poses. R = (n-1)! in
           sequential mode (every cycle once), 1 in single mode.
pass       one batch of envs stepped together. `parallel` runs every scene in
           one pass on its own env range; `serial` runs one pass per scene with
           every env loaded with that scene.
domain     "in_domain" (the run's `obj_set`) or "out_of_domain" (`rand_ood`).
unit       (domain, pass, round) — what runs, completes, resumes and merges.

Round r (4 tasks, sequential, R = 6):
    pose_set = r // 6, cycle_idx = r % 6
    cycle_idx 0      training round: blocks run ABCD BCDA CDAB DABC
    cycle_idx 1..5   random rounds: blocks run the rotations of the
                     cycle_idx-th untrained cycle (same cycle order in every set)
So rounds 0-5 are all 24 orderings from pose set 0, rounds 6-11 all 24 from
pose set 1 (round 6 is its training round), and so on.

Single mode: round r = pose set r; blocks A, B, C, D each run their task once.

Reproducibility contract (envs/rng_streams.py)
----------------------------------------------
For a fixed seed a round is fully determined by its number, whatever the
checkpoint, the rounds selected, the scene schedule, or what ran before it:

- orders   cycle_idx-th cycle of a prefix-stable shuffle keyed by
           (sequence_seed, n_tasks);
- poses    slot j of a block gets id j of `layout|seed|domain|set=pose_set`,
           independent of round, order and scene;
- policy   torch is reseeded from `policy|seed|domain|round|pass` at the start
           of every unit, so action sampling does not depend on earlier units.

Every key and drawn id is written to `eval_plan.json`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from envs import rng_streams

VALID_MODES = ("sequential", "single")
VALID_SCHEDULES = ("parallel", "serial")
VALID_DOMAINS = ("in_domain", "out_of_domain")
VALID_POSE_PHASES = ("start", "end", "both")
OOD_OBJ_SET = "rand_ood"
MAX_ROUNDS_PER_SET_FOR_ALL = 24   # rounds: all refuses when a pose set is longer than this

# Keys that existed in earlier drafts of the `eval:` block. Rejected with a pointer
# instead of "unknown key", because a config written against them would silently
# run a different design otherwise.
REMOVED_KEYS = {
    "num_sequences": "use `rounds` (e.g. rounds: 0-5)",
    "include_training_sequence": "use `rounds` (training rounds are the multiples of the rounds per pose set)",
    "training_orders": "always rotations",
    "random_orders": "always cycles",
    "layout_rng": "always seeded",
}


class PlanError(ValueError):
    """A config/CLI combination that cannot be evaluated as asked."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# YAML `eval:` key -> (CLI field on EvalArgs / main.Args, default).
# Precedence per field: explicit CLI flag > YAML `eval:` block > default.
EVAL_SETTING_SPEC: Dict[str, Tuple[str, object]] = {
    "mode": ("eval_mode", "sequential"),
    "pose_sets": ("eval_pose_sets", 1),
    "rounds": ("eval_rounds", "all"),
    "sequence_seed": ("eval_sequence_seed", -1),
    "layout_seed": ("eval_layout_seed", -1),
    "policy_seed": ("eval_policy_seed", -1),
    "layout_slots": ("eval_layout_slots", -1),
    "scene_schedule": ("eval_scene_schedule", "parallel"),
    "domains": ("eval_domains", "in_domain,out_of_domain"),
    "record_video": ("record_video", True),
    "video_envs_per_block": ("video_envs_per_block", -1),
    "record_pose": ("record_eval_pose", True),
    "pose_phase": ("eval_pose_phase", "both"),
}

# Settings that do not change what a unit produces; excluded from the fingerprint.
#   rounds, domains  the selection — shards of one eval differ here
#   pose_sets        only sets the design size; round r's orders, poses and policy
#                    seed do not depend on it, so a later `pose_sets: 2, rounds: 6-11`
#                    run merges with an earlier `pose_sets: 1` run
#   record_video, video_envs_per_block
#                    videos do not touch trial results; a resume may turn them off
#   sources          bookkeeping
# record_pose / pose_phase stay in: changing them mid-eval would leave
# eval_segment_pose.csv covering some units and not others.
_SELECTION_KEYS = ("rounds", "domains", "pose_sets", "record_video", "video_envs_per_block", "sources")


@dataclass
class EvalSettings:
    mode: str = "sequential"
    pose_sets: int = 1                     # sets of start poses; each set is R rounds
    rounds: str = "all"                    # "all" | "3" | "3-5" | "3-" | "0,6-11"
    sequence_seed: int = -1                # -1 = the run seed
    layout_seed: int = -1
    policy_seed: int = -1
    layout_slots: int = -1                 # start poses per block; -1 = one per env of the block
    scene_schedule: str = "parallel"
    domains: List[str] = field(default_factory=lambda: list(VALID_DOMAINS))
    record_video: bool = True
    video_envs_per_block: int = -1         # first k envs of each block; -1 = all envs
    record_pose: bool = True
    pose_phase: str = "both"
    sources: Dict[str, str] = field(default_factory=dict)   # field -> cli | yaml | default

    def to_dict(self) -> dict:
        return asdict(self)


def cli_flag_given(field_name: str, argv: Sequence[str]) -> bool:
    """Whether `--field-name`, `--field_name` or their `--no-` form is on argv.

    Token-exact (`--eval-sequences` does not match `--eval-sequences-foo`), and
    recognises tyro's negated boolean flags, which a plain substring test misses.
    """
    snake = field_name
    dash = field_name.replace("_", "-")
    names = {f"--{snake}", f"--{dash}", f"--no-{snake}", f"--no-{dash}",
             f"--no_{snake}"}
    for tok in argv:
        if tok.split("=", 1)[0] in names:
            return True
    return False


def _parse_domains(value) -> List[str]:
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
    else:
        items = [str(v).strip() for v in (value or [])]
    if not items:
        raise PlanError("eval domains must not be empty")
    bad = [d for d in items if d not in VALID_DOMAINS]
    if bad:
        raise PlanError(f"unknown eval domain(s) {bad}; valid: {list(VALID_DOMAINS)}")
    out = []
    for d in items:
        if d not in out:
            out.append(d)
    return out



def _normalize_rounds(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value).strip()


def parse_rounds(spec: str, total: int) -> List[int]:
    """"all" | "3" | "3-5" | "3-" | comma-separated mix  ->  sorted round numbers < total."""
    spec = _normalize_rounds(spec)
    if spec.lower() == "all":
        return list(range(total))
    out = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d*))?", tok)
        if not m:
            raise PlanError(f"rounds: cannot parse {tok!r} (use e.g. 3, 3-5, 3-, 0,6-11, all)")
        lo = int(m.group(1))
        hi = lo if m.group(2) is None else (total - 1 if m.group(2) == "" else int(m.group(2)))
        if hi < lo:
            raise PlanError(f"rounds: {tok!r} is an empty range")
        if hi >= total:
            raise PlanError(f"rounds: {tok!r} goes past the last round {total - 1} "
                            f"(pose_sets x rounds per set = {total}); raise pose_sets or lower the range")
        out.update(range(lo, hi + 1))
    if not out:
        raise PlanError("rounds selects nothing")
    return sorted(out)


def resolve_eval_settings(args, yaml_eval: Optional[dict], argv: Sequence[str]) -> EvalSettings:
    """Merge CLI args, the YAML `eval:` block and defaults into `EvalSettings`.

    `args` is any namespace carrying the CLI field names of `EVAL_SETTING_SPEC`.
    The legacy `eval_ood` flag still works: given explicitly and without
    `--eval-domains`, it selects `in_domain` alone or both domains.
    """
    yaml_eval = dict(yaml_eval or {})
    removed = {k: v for k, v in REMOVED_KEYS.items() if k in yaml_eval}
    if removed:
        raise PlanError("removed key(s) in YAML `eval:` block: " +
                        "; ".join(f"{k}: {v}" for k, v in removed.items()))
    unknown = set(yaml_eval) - set(EVAL_SETTING_SPEC)
    if unknown:
        raise PlanError(f"unknown key(s) in YAML `eval:` block: {sorted(unknown)}; "
                        f"valid: {sorted(EVAL_SETTING_SPEC)}")

    values, sources = {}, {}
    for key, (cli_name, default) in EVAL_SETTING_SPEC.items():
        if hasattr(args, cli_name) and cli_flag_given(cli_name, argv):
            values[key], sources[key] = getattr(args, cli_name), "cli"
        elif key in yaml_eval:
            values[key], sources[key] = yaml_eval[key], "yaml"
        else:
            values[key], sources[key] = default, "default"

    if (sources["domains"] != "cli" and hasattr(args, "eval_ood")
            and cli_flag_given("eval_ood", argv)):
        values["domains"] = "in_domain,out_of_domain" if args.eval_ood else "in_domain"
        sources["domains"] = "cli(eval_ood)"

    def _int(k):
        return -1 if values[k] is None else int(values[k])

    s = EvalSettings(
        mode=str(values["mode"]),
        pose_sets=int(values["pose_sets"]),
        rounds=_normalize_rounds(values["rounds"]),
        sequence_seed=_int("sequence_seed"),
        layout_seed=_int("layout_seed"),
        policy_seed=_int("policy_seed"),
        layout_slots=int(values["layout_slots"]),
        scene_schedule=str(values["scene_schedule"]),
        domains=_parse_domains(values["domains"]),
        record_video=bool(values["record_video"]),
        video_envs_per_block=int(values["video_envs_per_block"]),
        record_pose=bool(values["record_pose"]),
        pose_phase=str(values["pose_phase"]),
        sources=sources,
    )
    validate_eval_settings(s)
    return s


def validate_eval_settings(s: EvalSettings) -> None:
    if s.mode not in VALID_MODES:
        raise PlanError(f"eval mode must be one of {VALID_MODES}, got {s.mode!r}")
    if s.scene_schedule not in VALID_SCHEDULES:
        raise PlanError(f"scene_schedule must be one of {VALID_SCHEDULES}, got {s.scene_schedule!r}")
    if s.pose_phase not in VALID_POSE_PHASES:
        raise PlanError(f"pose_phase must be one of {VALID_POSE_PHASES}, got {s.pose_phase!r}")
    if s.pose_sets < 1:
        raise PlanError(f"pose_sets must be >= 1, got {s.pose_sets}")
    if s.layout_slots == 0 or s.layout_slots < -1:
        raise PlanError(f"layout_slots must be -1 (one per env) or >= 1, got {s.layout_slots}")
    if s.video_envs_per_block < -1 or s.video_envs_per_block == 0 and s.record_video:
        raise PlanError("video_envs_per_block must be -1 (all) or a positive count "
                        f"(use record_video: false to disable), got {s.video_envs_per_block}")
    if s.rounds.lower() != "all":
        for tok in s.rounds.split(","):
            if tok.strip() and not re.fullmatch(r"\d+(?:-\d*)?", tok.strip()):
                raise PlanError(f"rounds: cannot parse {tok.strip()!r} (use e.g. 3, 3-5, 3-, 0,6-11, all)")


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

@dataclass
class EvalScene:
    index: int
    name: str
    env_start: int
    num_envs: int
    eval_tasks: List[str]
    training_order: List[str]              # canonical order; positions A, B, C, ... index into it
    background: str = "default"
    hidden_obj_slots: int = 0              # batch-wide N minus this scene's objects
    hidden_recep_slots: int = 0
    rotations_trained: bool = False        # fan-out training stepped all cyclic rotations of training_order
    rotation_note: str = ""                # why not, when False

    @property
    def env_stop(self) -> int:
        return self.env_start + self.num_envs

    @property
    def n_tasks(self) -> int:
        return len(self.training_order)


def training_order(task_sequence: Sequence[str], eval_tasks: Sequence[str]) -> List[str]:
    """Canonical order for a scene.

    The training order restricted to the eval tasks (first occurrence wins),
    followed by eval tasks the scene never trained on, in their listed order.
    When eval_tasks == unique(task_sequence) — every shipped config but the
    half-train one — this is exactly the training task pool order.
    """
    evals = list(dict.fromkeys(eval_tasks))
    eval_set = set(evals)
    order = [t for t in dict.fromkeys(task_sequence) if t in eval_set]
    order += [t for t in evals if t not in set(order)]
    return order


def rotations_trained(task_sequence: Sequence[str], eval_tasks: Sequence[str], fan_out: bool) -> Tuple[bool, str]:
    """Whether training ran every cyclic rotation of this scene's eval tasks.

    `TaskScheduler.get_next_tasks` with fan_out splits a group into one
    sub-block per unique task and gives sub-block t `pool[(t + offset) % n]`;
    `update_index` adds one to `offset` per forward segment. So the batch runs
    the n cyclic rotations of the pool side by side, regardless of `task_order`.
    """
    pool = list(dict.fromkeys(task_sequence))
    if not fan_out:
        return False, "fan_out is off: training ran one order per group"
    if len(pool) < 2:
        return False, "single training task"
    if set(pool) != set(eval_tasks):
        return False, "eval tasks differ from the trained task pool"
    return True, ""


def build_scenes(group_states, num_envs: int, backgrounds: Optional[Sequence] = None,
                 env_n: Optional[int] = None, env_m: Optional[int] = None,
                 group_specs=None, fan_out: bool = True) -> List[EvalScene]:
    """One `EvalScene` per scheduler group, with contiguous env ranges.

    `group_states` are `envs.scheduler.GroupState`s (resolved task strings).
    `group_specs` (optional `envs.config.GroupSpec`s, same order) supply the
    background label and the object/receptacle counts used to report hidden
    slots. `fan_out` is the config's (TaskScheduler.fan_out).
    """
    scenes, start = [], 0
    for i, gs in enumerate(group_states):
        spec = group_specs[i] if group_specs else None
        bg = spec.background if spec is not None else (backgrounds[i] if backgrounds else "default")
        rot_ok, rot_note = rotations_trained(gs.task_sequence, gs.eval_tasks, fan_out)
        scenes.append(EvalScene(
            index=i,
            name=gs.name,
            env_start=start,
            num_envs=int(gs.num_envs),
            eval_tasks=list(gs.eval_tasks),
            training_order=training_order(gs.task_sequence, gs.eval_tasks),
            background=f"bg{bg}" if isinstance(bg, int) else str(bg),
            hidden_obj_slots=(env_n - len(spec.obj)) if (spec is not None and env_n) else 0,
            hidden_recep_slots=(env_m - len(spec.recep)) if (spec is not None and env_m) else 0,
            rotations_trained=rot_ok,
            rotation_note=rot_note,
        ))
        start += int(gs.num_envs)
    check_env_partition(scenes, num_envs)
    return scenes


def check_env_partition(scenes: Sequence[EvalScene], num_envs: int) -> None:
    """Scene env ranges must tile [0, num_envs) exactly — no padding envs.

    A padded env would run a copy of some other env's task and inflate that
    task's trial count. `TaskScheduler.get_next_tasks` does pad in that case;
    eval refuses instead.
    """
    cursor = 0
    for sc in scenes:
        if sc.num_envs < 1:
            raise PlanError(f"scene {sc.name!r} has num_envs={sc.num_envs}")
        if sc.env_start != cursor:
            raise PlanError(f"scene {sc.name!r} starts at env {sc.env_start}, expected {cursor}")
        cursor = sc.env_stop
    if cursor != num_envs:
        raise PlanError(
            f"scene env ranges cover {cursor} envs but the batch has num_envs={num_envs}. "
            f"Eval does not pad: set num_envs to the sum of per-group num_envs "
            f"(drop --num-envs from the command line, the YAML value is authoritative).")



# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------

def order_label(order: Sequence[int]) -> str:
    """Positions -> letters of the training order: (1, 2, 3, 0) -> "BCDA"."""
    if len(order) <= 26:
        return "".join(chr(ord("A") + i) for i in order)
    return "-".join(str(i) for i in order)



@dataclass
class SequenceSpec:
    """One round."""
    seq_idx: int                           # global round number
    seq_kind: str                          # training | random | single
    orders: List[List[int]]                # one order per block
    pose_set: int = 0
    cycle_idx: int = 0                     # 0 = training cycle; single mode: 0

    @property
    def order(self) -> List[int]:
        return self.orders[0]

    @property
    def n_slots(self) -> int:
        return len(self.orders[0])


def cyclic_rotations(n_tasks: int) -> List[List[int]]:
    return [list(r) for r in rng_streams.excluded_orders(n_tasks, "cyclic")]


def rounds_per_pose_set(n_tasks: int, mode: str) -> int:
    if mode == "single" or n_tasks < 2:
        return 1
    return rng_streams.n_cycles(n_tasks)


def build_rounds(n_tasks: int, mode: str, round_numbers: Sequence[int], seed: int) -> List[SequenceSpec]:
    if n_tasks < 1:
        raise PlanError("a scene has no eval tasks")
    R = rounds_per_pose_set(n_tasks, mode)
    max_cycle = max((r % R for r in round_numbers), default=0)
    cycles = rng_streams.random_cycles(n_tasks, seed, max_cycle) if (mode == "sequential" and max_cycle) else []
    out = []
    for r in round_numbers:
        p, c = divmod(r, R)
        if mode == "single":
            out.append(SequenceSpec(r, "single", [[t] for t in range(n_tasks)], p, 0))
        elif c == 0:
            out.append(SequenceSpec(r, "training", cyclic_rotations(n_tasks), p, 0))
        else:
            out.append(SequenceSpec(r, "random", [list(o) for o in rng_streams.rotations_of(cycles[c - 1])], p, c))
    return out


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass
class EvalPass:
    label: str
    scenes: List[EvalScene]                # scenes stepped together in this pass
    env_ranges: List[Tuple[int, int]]      # (start, stop) per scene, within [0, num_envs)
    group_idx_override: Optional[int]      # serial: load this scene into every env
    sequences: List[SequenceSpec]          # the selected rounds
    mode: str = "sequential"

    @property
    def n_tasks(self) -> int:
        return self.scenes[0].n_tasks

    def splits(self, scene_i: int) -> bool:
        """Whether this scene's envs are divided into per-order blocks.

        Sequential: only when training ran the rotations (fan-out); otherwise every
        env of the scene runs the round's first order. Single mode: always, so the
        blocks run tasks A, B, C, D side by side.
        """
        sc = self.scenes[scene_i]
        return sc.n_tasks > 1 and (sc.rotations_trained or self.mode == "single")

    def block_size(self, scene_i: int) -> int:
        lo, hi = self.env_ranges[scene_i]
        return (hi - lo) // self.scenes[scene_i].n_tasks if self.splits(scene_i) else hi - lo

    def scene_orders(self, seq: SequenceSpec, scene_i: int) -> List[Tuple[int, int, List[int]]]:
        """[(env_lo, env_hi, order)] for one scene in one round."""
        lo, hi = self.env_ranges[scene_i]
        if not self.splits(scene_i):
            return [(lo, hi, seq.orders[0])]
        b = self.block_size(scene_i)
        return [(lo + k * b, lo + (k + 1) * b, seq.orders[k]) for k in range(len(seq.orders))]

    def env_assignment(self, seq: SequenceSpec, task_idx: int, num_envs: int):
        """Per-env (obj, recep, task, scene, order label) for one task slot."""
        from envs.scheduler import TaskScheduler
        objs, receps, tasks, groups, orders = ([None] * num_envs for _ in range(5))
        for si, sc in enumerate(self.scenes):
            for lo, hi, order in self.scene_orders(seq, si):
                task = sc.training_order[order[task_idx]]
                obj, recep = TaskScheduler._extract_obj_recep(task)
                if obj is None:
                    raise PlanError(f"scene {sc.name!r}: task {task!r} is not 'put X on Y'")
                label = order_label(order)
                for e in range(lo, hi):
                    objs[e], receps[e], tasks[e], groups[e], orders[e] = obj, recep, task, sc.name, label
        if any(v is None for v in objs):
            missing = [i for i, v in enumerate(objs) if v is None]
            raise PlanError(f"pass {self.label!r}: envs {missing[:8]} have no task (would need padding)")
        return objs, receps, tasks, groups, orders

    def layout_slots(self, num_envs: int, layout_slots: int) -> List[int]:
        """Per-env layout slot: position within its block, modulo `layout_slots`."""
        slots: List[Optional[int]] = [None] * num_envs
        for si, (lo, hi) in enumerate(self.env_ranges):
            block = self.block_size(si)
            k = block if layout_slots < 0 else min(layout_slots, block)
            for e in range(lo, hi):
                slots[e] = ((e - lo) % block) % k
        return slots  # type: ignore[return-value]

    def layout_ids(self, domain: str, pose_set: int, num_envs: int, layout_seed: int,
                   layout_slots: int) -> List[int]:
        """Per-env layout ids for the reset of any round of `pose_set`."""
        slots = self.layout_slots(num_envs, layout_slots)
        base = rng_streams.layout_ids(layout_seed, domain, max(slots) + 1, pose_set)
        return [base[j] for j in slots]


@dataclass
class EvalPlan:
    settings: EvalSettings
    num_envs: int
    segment_len: int
    seed: int
    sequence_seed: int
    layout_seed: int
    policy_seed: int
    scenes: List[EvalScene]
    passes: List[EvalPass]
    provenance: dict = field(default_factory=dict)   # config file, checkpoint, comparison
    fingerprint: dict = field(default_factory=dict)  # see plan_fingerprint

    # ---- units ------------------------------------------------------------
    def units(self) -> List[Tuple[str, str, int]]:
        """(domain, pass_label, round) in execution order."""
        return [(d, p.label, s.seq_idx) for d in self.settings.domains
                for p in self.passes for s in p.sequences]

    def unit_rows(self) -> List[dict]:
        """One row per unit with everything coverage needs, so shards can be
        combined without their plans: per scene the env count, task slots,
        layout slots, orders and distinct tasks."""
        rows = []
        for d in self.settings.domains:
            for p in self.passes:
                slots = p.layout_slots(self.num_envs, self.settings.layout_slots)
                for q in p.sequences:
                    scenes = {}
                    for si, sc in enumerate(p.scenes):
                        lo, hi = p.env_ranges[si]
                        scenes[sc.name] = dict(
                            envs=hi - lo, layout_slots=len(set(slots[lo:hi])), distinct_tasks=sc.n_tasks,
                            orders=sorted({order_label(o) for _, _, o in p.scene_orders(q, si)}),
                            background=sc.background)
                    rows.append(dict(eval_kind=d, pass_label=p.label, seq_idx=q.seq_idx,
                                     seq_kind=q.seq_kind, pose_set=q.pose_set, cycle_idx=q.cycle_idx,
                                     n_slots=q.n_slots, trials=self.num_envs * q.n_slots, scenes=scenes))
        return rows

    def coverage(self) -> List[dict]:
        return coverage_from_units(self.unit_rows())

    def total_env_steps(self) -> int:
        steps = sum(s.n_slots * self.segment_len * self.num_envs
                    for p in self.passes for s in p.sequences)
        return steps * len(self.settings.domains)

    # ---- serialisation ------------------------------------------------------
    def rng_record(self) -> dict:
        """Every stream key and drawn value — enough to reproduce the schedule offline."""
        s = self.settings
        n_tasks = sorted({sc.n_tasks for sc in self.scenes})
        sets = sorted({q.pose_set for p in self.passes for q in p.sequences})
        n_slots = max(max(p.layout_slots(self.num_envs, s.layout_slots)) + 1 for p in self.passes)
        return dict(
            cycle_streams={str(t): rng_streams.cycle_key(self.sequence_seed, t) for t in n_tasks},
            layouts=[dict(eval_kind=d, pose_set=ps, key=rng_streams.layout_key(self.layout_seed, d, ps),
                          layout_ids=rng_streams.layout_ids(self.layout_seed, d, n_slots, ps))
                     for d in s.domains for ps in sets],
            env_layout_slots={p.label: p.layout_slots(self.num_envs, s.layout_slots) for p in self.passes},
            policy=[dict(eval_kind=d, pass_label=pl, seq_idx=r,
                         key=rng_streams.policy_key(self.policy_seed, d, r, pl),
                         torch_seed=rng_streams.policy_seed(self.policy_seed, d, r, pl))
                    for d, pl, r in self.units()],
        )

    def to_dict(self) -> dict:
        return dict(
            settings=self.settings.to_dict(),
            fingerprint=self.fingerprint,
            num_envs=self.num_envs, segment_len=self.segment_len, seed=self.seed,
            sequence_seed=self.sequence_seed, layout_seed=self.layout_seed, policy_seed=self.policy_seed,
            total_env_steps=self.total_env_steps(),
            provenance=self.provenance,
            scenes=[asdict(sc) for sc in self.scenes],
            passes=[dict(
                label=p.label, group_idx_override=p.group_idx_override,
                env_ranges=[dict(group=sc.name, start=lo, stop=hi)
                            for sc, (lo, hi) in zip(p.scenes, p.env_ranges)],
                rounds=[dict(seq_idx=q.seq_idx, seq_kind=q.seq_kind, pose_set=q.pose_set, cycle_idx=q.cycle_idx,
                             scenes={sc.name: [dict(env_start=lo, env_stop=hi, order=order_label(o),
                                                    tasks=[sc.training_order[i] for i in o])
                                               for lo, hi, o in p.scene_orders(q, si)]
                                     for si, sc in enumerate(p.scenes)})
                        for q in p.sequences],
            ) for p in self.passes],
            units=self.unit_rows(),
            design={p.label: rounds_per_pose_set(p.n_tasks, self.settings.mode) * self.settings.pose_sets
                    for p in self.passes},
            coverage=self.coverage(),
            rng=self.rng_record(),
        )

    def write_json(self, path: Path) -> None:
        tmp = Path(path).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        tmp.replace(path)

    # ---- human-readable -----------------------------------------------------
    def describe(self) -> str:
        s = self.settings
        p0 = self.passes[0]
        R = rounds_per_pose_set(p0.n_tasks, s.mode)
        lines = [
            f"[eval plan] mode={s.mode} schedule={s.scene_schedule} domains={s.domains} "
            f"num_envs={self.num_envs} segment_len={self.segment_len}",
            f"[eval plan] pose_sets={s.pose_sets} x {R} rounds/set; rounds={s.rounds!r} -> "
            f"{len(p0.sequences)} round(s); seeds: sequence={self.sequence_seed} layout={self.layout_seed} "
            f"policy={self.policy_seed}; total env-steps={self.total_env_steps():,}",
        ]
        for key in sorted(s.sources):
            if s.sources[key] != "default":
                lines.append(f"[eval plan]   {key} = {getattr(s, key)!r}  (from {s.sources[key]})")
        if self.provenance:
            lines.append(f"[eval plan] config: {self.provenance.get('config_path')} "
                         f"({self.provenance.get('config_source')}); "
                         f"training config check: {self.provenance.get('check')}")
        for si, sc in enumerate(p0.scenes if len(self.passes) == 1 else self.scenes):
            hidden = ""
            if sc.hidden_obj_slots or sc.hidden_recep_slots:
                hidden = f" hidden slots: obj={sc.hidden_obj_slots} recep={sc.hidden_recep_slots} (NaN in pose CSV)"
            split = "" if sc.rotations_trained else f" [not split into order blocks: {sc.rotation_note}]"
            lines.append(f"[eval plan] scene {sc.name}: envs [{sc.env_start},{sc.env_stop}) "
                         f"background={sc.background} tasks={sc.n_tasks}{hidden}{split}")
            lines.append("[eval plan]   order letters: " +
                         ", ".join(f"{order_label([i])}={t}" for i, t in enumerate(sc.training_order)))
        for q in p0.sequences:
            labels = " ".join(order_label(o) for o in q.orders)
            lines.append(f"[eval plan]   round {q.seq_idx:>3} (set {q.pose_set}, {q.seq_kind}): {labels}")
        if s.mode == "sequential":
            n_orders = len({tuple(o) for q in p0.sequences for o in q.orders})
            lines.append(f"[eval plan]   distinct orders selected: {n_orders} of {math.factorial(p0.n_tasks)}")
        slots = p0.layout_slots(self.num_envs, s.layout_slots)
        lo, hi = p0.env_ranges[0]
        lines.append(f"[eval plan]   block = {p0.block_size(0)} envs, start poses per pose set = "
                     f"{len(set(slots[lo:hi]))} (shared by every round, order and scene of the set)")
        lines.append(format_coverage_table(self.coverage(), title="planned coverage"))
        return "\n".join(lines)


def unit_key(row: dict) -> Tuple[str, str, int]:
    return (row["eval_kind"], row["pass_label"], int(row["seq_idx"]))


def coverage_from_units(units: List[dict]) -> List[dict]:
    """Planned counts per (domain, seq_kind, scene), from unit rows.

    trials    = envs × task slots, summed over rounds
    per_task  = trials / distinct tasks (every task once per block per round)
    resets    = envs × rounds
    layouts   = distinct start poses = pose sets used × layout slots
    """
    acc: Dict[Tuple[str, str, str], dict] = {}
    for u in units:
        for g, info in u["scenes"].items():
            key = (u["eval_kind"], u["seq_kind"], g)
            a = acc.setdefault(key, dict(eval_kind=key[0], seq_kind=key[1], group=g, pass_label=u["pass_label"],
                                         envs=info["envs"], rounds=0, sets=set(), orders=set(),
                                         tasks_per_round=u["n_slots"], distinct_tasks=info["distinct_tasks"],
                                         trials=0, resets=0, layout_slots=info["layout_slots"]))
            a["rounds"] += 1
            a["sets"].add(u["pose_set"])
            a["orders"].update(info["orders"])
            a["trials"] += info["envs"] * u["n_slots"]
            a["resets"] += info["envs"]
    rows = []
    for a in acc.values():
        rows.append(dict(
            eval_kind=a["eval_kind"], seq_kind=a["seq_kind"], group=a["group"], pass_label=a["pass_label"],
            envs=a["envs"], rounds=a["rounds"], pose_sets=len(a["sets"]), orders=len(a["orders"]),
            tasks_per_round=a["tasks_per_round"], distinct_tasks=a["distinct_tasks"],
            trials=a["trials"], trials_per_task=a["trials"] // a["distinct_tasks"],
            resets=a["resets"], layouts=len(a["sets"]) * a["layout_slots"],
        ))
    return rows


def format_coverage_table(rows: List[dict], title: str) -> str:
    hdr = (f"{'eval_kind':<14} {'seq_kind':<9} {'group':<16} {'envs':>5} {'rounds':>6} {'sets':>4} "
           f"{'orders':>6} {'trials':>7} {'per_task':>8} {'resets':>6} {'layouts':>7}")
    out = [f"[eval {title}]", hdr, "-" * len(hdr)]
    for r in rows:
        out.append(f"{r['eval_kind']:<14} {r['seq_kind']:<9} {r['group']:<16} {r['envs']:>5} "
                   f"{r['rounds']:>6} {r['pose_sets']:>4} {r['orders']:>6} {r['trials']:>7} "
                   f"{r['trials_per_task']:>8} {r['resets']:>6} {r['layouts']:>7}")
    return "\n".join(out)


def build_plan(scenes: List[EvalScene], settings: EvalSettings, *, num_envs: int,
               segment_len: int, seed: int) -> EvalPlan:
    check_env_partition(scenes, num_envs)

    def seed_or(v):
        return seed if v < 0 else v

    seq_seed = seed_or(settings.sequence_seed)

    def selected(n_tasks: int) -> List[int]:
        R = rounds_per_pose_set(n_tasks, settings.mode)
        if settings.rounds.lower() == "all" and R > MAX_ROUNDS_PER_SET_FOR_ALL:
            raise PlanError(
                f"rounds: all would run {R} rounds per pose set for {n_tasks} tasks; "
                f"select rounds explicitly (e.g. rounds: 0-{MAX_ROUNDS_PER_SET_FOR_ALL - 1}).")
        return parse_rounds(settings.rounds, R * settings.pose_sets)

    passes: List[EvalPass] = []
    if settings.scene_schedule == "parallel":
        counts = {sc.name: sc.n_tasks for sc in scenes}
        if len(set(counts.values())) != 1:
            raise PlanError(
                f"scene_schedule=parallel steps every scene together, so all scenes need the "
                f"same number of eval tasks; got {counts}. Use scene_schedule: serial.")
        n = scenes[0].n_tasks
        passes.append(EvalPass("all_scenes", list(scenes), [(sc.env_start, sc.env_stop) for sc in scenes],
                               None, build_rounds(n, settings.mode, selected(n), seq_seed), settings.mode))
    else:
        for sc in scenes:
            passes.append(EvalPass(sc.name, [sc], [(0, num_envs)],
                                   sc.index if len(scenes) > 1 else None,
                                   build_rounds(sc.n_tasks, settings.mode, selected(sc.n_tasks), seq_seed),
                                   settings.mode))
    for p in passes:
        for si, sc in enumerate(p.scenes):
            lo, hi = p.env_ranges[si]
            if p.splits(si) and (hi - lo) % sc.n_tasks:
                raise PlanError(f"scene {sc.name!r}: {hi - lo} envs cannot be split evenly into "
                                f"{sc.n_tasks} order blocks (eval does not pad).")
    return EvalPlan(settings=settings, num_envs=num_envs, segment_len=segment_len, seed=seed,
                    sequence_seed=seq_seed, layout_seed=seed_or(settings.layout_seed),
                    policy_seed=seed_or(settings.policy_seed), scenes=list(scenes), passes=passes)


# ---------------------------------------------------------------------------
# Fingerprint: what must match for resumed or merged runs to be one eval
# ---------------------------------------------------------------------------

def plan_fingerprint(plan: EvalPlan, extra: dict) -> dict:
    """Everything that changes what a unit produces, plus a hash of it.

    Excludes the selection (`rounds`, `domains`), so shards of one eval share a
    fingerprint. `extra` carries run inputs outside the plan: checkpoint, policy
    and sampling settings, the config's scene definition.
    """
    settings = {k: v for k, v in plan.settings.to_dict().items() if k not in _SELECTION_KEYS}
    body = dict(settings=settings, num_envs=plan.num_envs, segment_len=plan.segment_len,
                seed=plan.seed, sequence_seed=plan.sequence_seed, layout_seed=plan.layout_seed,
                policy_seed=plan.policy_seed, scenes=[asdict(sc) for sc in plan.scenes], **extra)
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    return dict(hash=digest, body=body)


def fingerprint_differences(a: dict, b: dict, prefix: str = "") -> List[str]:
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += fingerprint_differences(a.get(k), b.get(k), f"{prefix}{k}.")
    elif a != b:
        out.append(f"{prefix.rstrip('.')}: {a!r} != {b!r}")
    return out


# ---------------------------------------------------------------------------
# Checkpoint progress (for the x-axis columns of the eval CSVs)
# ---------------------------------------------------------------------------

def checkpoint_progress(ckpt_dir: str) -> Tuple[int, int, str]:
    """(episode, total_steps, source) for a `.../glob/episode_XXXX` checkpoint.

    `episode` comes from the directory name. `total_steps` is
    episode × episode_len × num_envs from the checkpoint's `run_config.yaml` —
    the same fallback `main.py::_restore_training_state` uses for old
    checkpoints; exact unless episode_len changed across a resume chain.
    Unknown values are 0 and the source says so.
    """
    if not ckpt_dir:
        return 0, 0, "none"
    m = re.search(r"episode_(\d+)", str(ckpt_dir))
    if not m:
        return 0, 0, "unknown (checkpoint dir not named episode_XXXX)"
    episode = int(m.group(1))
    cfg = Path(ckpt_dir) / "run_config.yaml"
    try:
        import yaml  # optional
        raw = yaml.safe_load(cfg.read_text()) if cfg.exists() else None
    except Exception:
        raw = None
    if isinstance(raw, dict) and raw.get("episode_len") and raw.get("num_envs"):
        return episode, episode * int(raw["episode_len"]) * int(raw["num_envs"]), "run_config.yaml estimate"
    return episode, 0, "episode from dir name; total_steps unknown"

