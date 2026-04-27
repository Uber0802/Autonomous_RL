"""CRONOS V0.3 — Task scheduler with per-group state and sub-group fan-out.

Supports three ordering modes:
- ``sequential``:       cycle through task_sequence in order
- ``pure_random``:      sample randomly each segment (≠ previous),
                        weighted by frequency in task_sequence
- ``sequence_random``:  random permutation of unique tasks, reshuffle when exhausted

V0.3 M3: With ``fan_out=True`` (default, AutoRL behavior), each group's envs
are further split into sub-groups, one per unique task in the pool. All tasks
run simultaneously within each group. The assignment rotates each segment.

Each group maintains its own cursor, permutation, and last-task state via
``GroupState``.  The scheduler returns flat ``(objects, receptacles)`` lists
of length ``num_envs`` where each group's envs are contiguous.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Per-group state
# ---------------------------------------------------------------------------

@dataclass
class GroupState:
    """Mutable scheduling state for one group."""
    name: str
    num_envs: int                     # envs allocated to this group
    task_sequence: List[str]          # resolved task strings (may have repeats)
    eval_tasks: List[str]             # resolved eval task strings
    task_pool: List[str]              # unique tasks (deduplicated from sequence)
    weights: List[float]              # sampling weights for pure_random (from frequency)
    cursor: int = 0                   # current position in sequence (sequential)
    shuffle_perm: List[int] = field(default_factory=list)  # current permutation (sequence_random)
    shuffle_cursor: int = 0           # position in current permutation
    last_task_idx: int = -1           # previous task pool index (pure_random: avoid repeat)

    @staticmethod
    def from_sequence(name: str, task_sequence: List[str],
                      eval_tasks: List[str],
                      num_envs: int = 0) -> "GroupState":
        """Create a GroupState from a resolved task_sequence."""
        # Build deduplicated pool preserving first-occurrence order
        seen = set()
        pool = []
        for t in task_sequence:
            if t not in seen:
                seen.add(t)
                pool.append(t)

        # Compute weights from frequency in task_sequence
        counts = {t: 0.0 for t in pool}
        for t in task_sequence:
            counts[t] += 1.0
        total = sum(counts.values())
        weights = [counts[t] / total for t in pool]

        return GroupState(
            name=name,
            num_envs=num_envs,
            task_sequence=task_sequence,
            eval_tasks=eval_tasks,
            task_pool=pool,
            weights=weights,
        )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Per-group task scheduler supporting 3 ordering modes and sub-group fan-out."""

    def __init__(
        self,
        group_states: List[GroupState],
        mode: str = "sequential",
        num_envs: int = 64,
        fan_out: bool = True,
    ):
        self.group_states = group_states
        self.mode = mode
        self.num_envs = num_envs
        self.fan_out = fan_out

        # Flat task_pool for backward compat (union of all groups' pools)
        seen = set()
        self.task_pool: List[str] = []
        for gs in group_states:
            for t in gs.task_pool:
                if t not in seen:
                    seen.add(t)
                    self.task_pool.append(t)

        # V0.3 M3: per-group fan-out cursor (rotates sub-group assignment)
        self._fan_out_offsets: List[int] = [0] * len(group_states)

        # Legacy compat
        self.current_task_idx = 0

    # ------------------------------------------------------------------
    # Class methods for backward-compatible construction
    # ------------------------------------------------------------------

    @classmethod
    def from_flat_pool(
        cls,
        task_pool: List[str],
        mode: str = "sequential",
        num_envs: int = 64,
        task_filter: Optional[List[Union[int, str]]] = None,
        fan_out: bool = True,
    ) -> "TaskScheduler":
        """Legacy constructor: single flat task pool → one group with all tasks."""
        if task_filter is not None:
            filtered = cls._apply_filter(task_pool, task_filter)
        else:
            filtered = list(task_pool)
        if not filtered:
            raise ValueError("task_pool is empty after applying task_filter")

        gs = GroupState.from_sequence(
            name="default",
            task_sequence=filtered,
            eval_tasks=filtered,
            num_envs=num_envs,
        )
        return cls(group_states=[gs], mode=mode, num_envs=num_envs, fan_out=fan_out)

    @staticmethod
    def _apply_filter(pool: List[str], filt: List[Union[int, str]]) -> List[str]:
        """Restrict pool to entries matching filt (indices or task strings)."""
        result = []
        for entry in filt:
            if isinstance(entry, int):
                if 0 <= entry < len(pool):
                    result.append(pool[entry])
                else:
                    raise ValueError(
                        f"task_filter index {entry} out of range "
                        f"(pool has {len(pool)} tasks)")
            elif isinstance(entry, str):
                matches = [t for t in pool if t == entry]
                if not matches:
                    raise ValueError(
                        f"task_filter string '{entry}' not found in pool: {pool}")
                result.extend(matches)
            else:
                raise ValueError(
                    f"task_filter entries must be int or str, got {type(entry)}")
        seen = set()
        return [t for t in result if not (t in seen or seen.add(t))]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_obj_recep(task_str: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts object and receptacle from a task string."""
        m = re.search(r"put (.*?) on (.*)", task_str)
        if m:
            return m.group(1), m.group(2)
        return None, None

    def _pick_task_for_group(self, gs: GroupState) -> str:
        """Pick the next task for a group based on the current mode."""
        if self.mode == "sequential":
            task = gs.task_sequence[gs.cursor % len(gs.task_sequence)]
            return task

        elif self.mode == "pure_random":
            pool = gs.task_pool
            if len(pool) == 1:
                return pool[0]
            # Weighted sample, different from last
            attempts = 0
            while attempts < 100:
                idx = random.choices(range(len(pool)), weights=gs.weights, k=1)[0]
                if idx != gs.last_task_idx or len(pool) == 1:
                    gs.last_task_idx = idx
                    return pool[idx]
                attempts += 1
            # Fallback (shouldn't happen)
            gs.last_task_idx = (gs.last_task_idx + 1) % len(pool)
            return pool[gs.last_task_idx]

        elif self.mode == "sequence_random":
            pool = gs.task_pool
            # Generate new permutation if needed
            if not gs.shuffle_perm or gs.shuffle_cursor >= len(gs.shuffle_perm):
                gs.shuffle_perm = list(range(len(pool)))
                random.shuffle(gs.shuffle_perm)
                gs.shuffle_cursor = 0
            idx = gs.shuffle_perm[gs.shuffle_cursor]
            return pool[idx]

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_next_tasks(self, num_groups: Optional[int] = None
                       ) -> Tuple[List[str], List[str]]:
        """Returns (objects, receptacles) for all envs.

        Each group's envs are contiguous: envs[0:group_size] = group 0, etc.

        With fan_out=True (V0.3 M3, AutoRL default), each group's envs are
        further split into sub-groups, one per unique task in the pool.
        This matches AutoRL's behavior where all tasks run simultaneously.
        """
        n_groups = len(self.group_states)
        if num_groups is not None:
            n_groups = min(num_groups, n_groups)

        objects: List[str] = []
        receptacles: List[str] = []

        for g_idx in range(n_groups):
            gs = self.group_states[g_idx % len(self.group_states)]
            g_size = gs.num_envs

            if self.fan_out and len(gs.task_pool) > 1:
                # V0.3 M3: fan out all unique tasks across sub-groups
                n_tasks = len(gs.task_pool)
                sub_size = g_size // n_tasks
                offset = self._fan_out_offsets[g_idx % len(self._fan_out_offsets)]
                for t_idx in range(n_tasks):
                    task = gs.task_pool[(t_idx + offset) % n_tasks]
                    obj, recep = self._extract_obj_recep(task)
                    objects.extend([obj] * sub_size)
                    receptacles.extend([recep] * sub_size)
                # Handle remainder envs within group
                remainder_in_group = g_size - sub_size * n_tasks
                if remainder_in_group > 0:
                    objects.extend([objects[-1]] * remainder_in_group)
                    receptacles.extend([receptacles[-1]] * remainder_in_group)
            else:
                # fan_out=False or single task: all envs get same task (V0.2 behavior)
                task = self._pick_task_for_group(gs)
                obj, recep = self._extract_obj_recep(task)
                objects.extend([obj] * g_size)
                receptacles.extend([recep] * g_size)

        # Pad to num_envs if needed (shouldn't happen with correct per-group sizes)
        remainder = self.num_envs - len(objects)
        if remainder > 0:
            objects.extend([objects[-1]] * remainder)
            receptacles.extend([receptacles[-1]] * remainder)

        return objects, receptacles

    def update_index(self):
        """Advance all groups' cursors by one step.

        With fan_out=True: rotates the sub-group assignment offset.
        With fan_out=False: advances per-group task cursor (V0.2 behavior).
        """
        for g_idx, gs in enumerate(self.group_states):
            if self.fan_out and len(gs.task_pool) > 1:
                # V0.3 M3: rotate sub-group assignment
                self._fan_out_offsets[g_idx] = (
                    self._fan_out_offsets[g_idx] + 1) % len(gs.task_pool)
            else:
                # V0.2: advance single-task cursor
                if self.mode == "sequential":
                    gs.cursor = (gs.cursor + 1) % len(gs.task_sequence)
                elif self.mode == "sequence_random":
                    gs.shuffle_cursor += 1

        # Legacy compat
        if self.task_pool:
            self.current_task_idx = (self.current_task_idx + 1) % len(self.task_pool)

    def get_eval_tasks_per_group(self) -> List[Tuple[str, int, List[str]]]:
        """Returns [(group_name, group_num_envs, [eval_task_strings]), ...] for all groups."""
        return [(gs.name, gs.num_envs, gs.eval_tasks) for gs in self.group_states]

    def get_max_eval_rounds(self) -> int:
        """Max eval tasks across all groups (determines number of eval rounds)."""
        if not self.group_states:
            return 0
        return max(len(gs.eval_tasks) for gs in self.group_states)

    # ------------------------------------------------------------------
    # State save/restore (for checkpoint resume)
    # ------------------------------------------------------------------

    def get_state(self) -> Dict:
        """Serialize scheduler state for checkpoint."""
        return {
            "mode": self.mode,
            "fan_out": self.fan_out,
            "current_task_idx": self.current_task_idx,
            "fan_out_offsets": self._fan_out_offsets,
            "groups": [
                {
                    "name": gs.name,
                    "cursor": gs.cursor,
                    "shuffle_perm": gs.shuffle_perm,
                    "shuffle_cursor": gs.shuffle_cursor,
                    "last_task_idx": gs.last_task_idx,
                }
                for gs in self.group_states
            ],
        }

    def load_state(self, state: Dict):
        """Restore scheduler state from checkpoint."""
        self.current_task_idx = state.get("current_task_idx", 0)
        saved_offsets = state.get("fan_out_offsets", [])
        for i, off in enumerate(saved_offsets):
            if i < len(self._fan_out_offsets):
                self._fan_out_offsets[i] = off
        for gs, gs_state in zip(self.group_states, state.get("groups", [])):
            gs.cursor = gs_state.get("cursor", 0)
            gs.shuffle_perm = gs_state.get("shuffle_perm", [])
            gs.shuffle_cursor = gs_state.get("shuffle_cursor", 0)
            gs.last_task_idx = gs_state.get("last_task_idx", -1)
