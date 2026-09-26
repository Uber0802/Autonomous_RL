"""Named, recordable RNG streams. Stdlib only.

Every random quantity is drawn from a `random.Random` whose seed is derived from
a readable key — e.g. ``layout|seed=0|domain=in_domain|set=1`` — by SHA-256.
Consequences:

- A value depends only on its key, never on how much of some shared generator
  was consumed before it. The env's own `torch.randint` layout draw shares the
  global CUDA generator with the policy's action sampling, so the layout an env
  received used to depend on the checkpoint being evaluated.
- The key *is* the record: logging it (plus the drawn integers) is enough to
  reproduce or audit a draw, with no generator state to serialise.
- Adding sequences, domains or scenes does not shift existing draws: streams
  are consumed in a fixed order (layout slot 0, 1, ...; cycle 0, 1, ...), so a
  longer draw extends a shorter one instead of replacing it.

Used by standalone eval (`evaluation/plan.py`) and by training (`main.py`):
training draws every scene quantity — episode layouts, HSR respawn poses and the
layouts of training-time eval — from `scene_ids`, and the task schedule from
`task_rng`, so neither the task order nor the policy's sampling on the GPU can
move a scene. `--legacy-rng` restores the old global-generator draws.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from typing import List, Tuple

LAYOUT_ID_BITS = 62          # fits int64; `% ltt` in the env is unbiased to ~2^-40
_PERM_MATERIALIZE_LIMIT = 50_000


def stream_key(namespace: str, **fields) -> str:
    """Canonical key string, fields in the order given."""
    return "|".join([namespace] + [f"{k}={v}" for k, v in fields.items()])


def stream_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def stream(key: str) -> random.Random:
    return random.Random(stream_seed(key))


def layout_key(seed: int, domain: str, pose_set: int = 0) -> str:
    return stream_key("layout", seed=seed, domain=domain, set=pose_set)


def layout_ids(seed: int, domain: str, n_slots: int, pose_set: int = 0) -> List[int]:
    """Layout ids for slots 0..n_slots-1 of one pose set. Prefix-stable in `n_slots`.

    Deliberately not keyed by round, order or scene: within a pose set, slot j is
    the same initial placement whenever it is used, so every sequence of the set
    starts from the same poses.
    """
    rng = stream(layout_key(seed, domain, pose_set))
    return [rng.getrandbits(LAYOUT_ID_BITS) for _ in range(n_slots)]


def scene_key(seed: int, kind: str, **fields) -> str:
    return stream_key("scene", seed=seed, kind=kind, **fields)


def scene_ids(seed: int, n: int, kind: str, **fields) -> List[int]:
    """`n` per-env scene ids for one training-side draw.

    `kind` names what the draw places — ``episode`` (layout at an episode's
    `env.reset()`), ``respawn`` (HSR re-placement at a segment boundary) or
    ``eval`` (training-time eval) — and `fields` say which one, e.g.
    ``episode=12, segment=3``. Ids are uniform 62-bit integers; the env maps
    them onto its config table with `% ltt`, the same distribution as its own
    `torch.randint` / `np.random.choice` draws, but computed on the CPU from
    the key alone: no GPU or global generator is consumed, and a resumed run
    redraws exactly what the uninterrupted run drew.
    """
    rng = stream(scene_key(seed, kind, **fields))
    return [rng.getrandbits(LAYOUT_ID_BITS) for _ in range(n)]


def task_rng(seed: int) -> random.Random:
    """Dedicated generator for the task schedule (`TaskScheduler`).

    Separate from every scene draw, so changing the task order (mode, pool,
    fan-out) never changes a layout, and the reverse. Stateful: the scheduler
    checkpoints its state in `scheduler_state.json`.
    """
    return stream(stream_key("task", seed=seed))


def policy_key(seed: int, domain: str, round_idx: int, pass_label: str) -> str:
    return stream_key("policy", seed=seed, domain=domain, round=round_idx, **{"pass": pass_label})


def policy_seed(seed: int, domain: str, round_idx: int, pass_label: str) -> int:
    """Seed for torch's generators at the start of one eval unit.

    Action sampling (temperature > 0) consumes the global CUDA generator. Reseeding
    per (domain, round, pass) makes a unit's samples independent of which units ran
    before it, so a resumed or sharded eval reproduces an uninterrupted one.
    """
    return stream_seed(policy_key(seed, domain, round_idx, pass_label)) % (2 ** 63 - 1)


EXCLUDE_MODES = ("identity", "cyclic")


def excluded_orders(n_tasks: int, exclude: str) -> List[Tuple[int, ...]]:
    """Orderings a random round must not reproduce.

    identity  the canonical order only
    cyclic    all n cyclic rotations (ABCD, BCDA, CDAB, DABC) — exactly the orders a
              fan-out training run steps through (TaskScheduler rotates each
              sub-block's offset by one task per segment)
    """
    if exclude == "identity":
        return [tuple(range(n_tasks))]
    if exclude == "cyclic":
        return [tuple((r + k) % n_tasks for k in range(n_tasks)) for r in range(n_tasks)]
    raise ValueError(f"exclude must be one of {EXCLUDE_MODES}, got {exclude!r}")


def sequence_key(seed: int, n_tasks: int, exclude: str = "identity") -> str:
    return stream_key("sequence", seed=seed, n_tasks=n_tasks, exclude=exclude)


def random_orders(n_tasks: int, seed: int, count: int, exclude: str = "identity") -> List[Tuple[int, ...]]:
    """`count` distinct permutations of range(n_tasks), none in `excluded_orders`.

    Prefix-stable: the r-th ordering (0-based) is the same for any count > r, so
    round N's task order does not depend on how many rounds were requested.
    Small pools shuffle the full candidate list once; large pools draw shuffles
    in order and skip repeats. Raises if fewer than `count` orderings qualify.
    """
    excluded = set(excluded_orders(n_tasks, exclude))
    available = math.factorial(n_tasks) - len(excluded)
    if count > available:
        raise ValueError(f"{count} distinct random orderings of {n_tasks} tasks requested, "
                         f"only {available} exist outside the {len(excluded)} excluded ({exclude})")
    if count <= 0:
        return []
    rng = stream(sequence_key(seed, n_tasks, exclude))
    if available + len(excluded) <= _PERM_MATERIALIZE_LIMIT:
        candidates = [p for p in itertools.permutations(range(n_tasks)) if p not in excluded]
        rng.shuffle(candidates)
        return candidates[:count]
    seen, out = set(excluded), []
    while len(out) < count:
        cand = list(range(n_tasks))
        rng.shuffle(cand)
        t = tuple(cand)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Cycles: an ordering up to rotation
# ---------------------------------------------------------------------------
#
# ABCD, BCDA, CDAB and DABC are one cycle, A->B->C->D->A, entered at different
# tasks. n tasks have (n-1)! cycles and each has n rotations, so the cycles
# partition all n! orderings. Fan-out training runs the rotations of the
# identity cycle; a random round draws another cycle and runs all its rotations.

def rotations_of(order) -> List[Tuple[int, ...]]:
    o = tuple(order)
    return [o[r:] + o[:r] for r in range(len(o))]


def n_cycles(n_tasks: int) -> int:
    return math.factorial(max(n_tasks, 1) - 1)


def cycle_key(seed: int, n_tasks: int) -> str:
    return stream_key("cycle", seed=seed, n_tasks=n_tasks)


def random_cycles(n_tasks: int, seed: int, count: int) -> List[Tuple[int, ...]]:
    """`count` distinct cycles other than the identity, each written from task 0.

    Prefix-stable like `random_orders`. Raises if fewer than `count` exist.
    """
    identity = tuple(range(n_tasks))
    available = n_cycles(n_tasks) - 1
    if count > available:
        raise ValueError(f"{count} random cycles of {n_tasks} tasks requested, only {available} "
                         f"exist besides the training cycle")
    if count <= 0:
        return []
    rng = stream(cycle_key(seed, n_tasks))
    if available + 1 <= _PERM_MATERIALIZE_LIMIT:
        candidates = [(0,) + p for p in itertools.permutations(range(1, n_tasks))]
        candidates = [c for c in candidates if c != identity]
        rng.shuffle(candidates)
        return candidates[:count]
    seen, out = {identity}, []
    while len(out) < count:
        rest = list(range(1, n_tasks))
        rng.shuffle(rest)
        c = (0,) + tuple(rest)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
