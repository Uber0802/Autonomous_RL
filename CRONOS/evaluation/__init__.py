"""CRONOS — standalone evaluation (single-task and sequential).

`plan` and `records` are dependency-free (no torch, no ManiSkill) so the
schedule, the per-scene coverage and the CSV writers can be unit-tested and
inspected on a machine without a GPU. `sequential` holds the rollout loop and is
the only module that touches the env and the policy.
"""
