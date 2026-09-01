# Reset modes

Index of these documents: [`README.md`](README.md).

What `scripts/train.sh`'s 4th positional argument means, what each mode is made
of, and which of them are scientifically load-bearing rather than merely
different. Read this before adding a mode or comparing two of them.

## The four dimensions are orthogonal

`main.py` has no notion of a "reset mode". It has four independent flags, and
every mode name is just a preset over them:

| Dimension | Flag | Default | What it does |
|---|---|---|---|
| **HSR** — High-level State Reset | `--reset-unsuitable` | off | At every task boundary, respawn actors the detector flags as *unsuitable* — fallen (`low_z`, z < 0.7) or out of the box (`workspace_aabb`) |
| **LSR** — Low-level State Reset | `--enable-backward --backward-interval N` | off | Alternate the forward task with a *reset goal* segment ("put X on table"), so the policy learns to undo its own work |
| **Episodic reset** | `--reset-mode per_episode\|none` | `per_episode` | Whether `env.reset()` runs between episodes at all |
| **EER** — End-Effector Reset | `--reset-robot` / `--no-reset-robot` | on | Whether the gripper returns to its initial pose at every segment boundary |

They are genuinely independent in the code, not just nominally:
`envs/reset.py::reset_unsuitable_envs` does only the object-side respawn (the M3
fix; before it, `--reset-unsuitable --no-reset-robot` silently behaved like
LSR+HSR), and `main.py`'s train loop applies HSR and EER as two separate `if`s,
matching AutoRL's `train_ms3_ppo.py:637-643`.

So any combination is expressible on the command line. The mode names exist to
make the combinations we actually run reproducible and taggable, not to restrict
them.

## The modes

| `reset` argument | Also accepted as | Flags it expands to | `RUN_TAG` fragment |
|---|---|---|---|
| `normal` | — | (none) | `normal` |
| `LSR` | — | `--enable-backward --backward-interval 1` | `LSR` |
| `HSR` | — | `--reset-unsuitable` | `HSR` |
| `HSR+LSR` | `LSR+HSR` | `--enable-backward --backward-interval 1 --reset-unsuitable` | `HSRLSR` |
| `noep` | — | `--reset-unsuitable --reset-mode none` | `HSRnoep` |
| `noep+LSR` | — | `--enable-backward --backward-interval 1 --reset-unsuitable --reset-mode none` | `HSRLSRnoep` |

`noep` means **HSR with no episodic reset**. It does *not* include LSR — that is
`noep+LSR`. The three-way comparison `HSR+LSR` / `noep` / `noep+LSR` is the
point: it separates "does removing the episodic reset hurt" from "does learning
a reset policy pay for it".

### Deriving the two dependent settings

Two things downstream of the mode must follow HSR and LSR rather than the mode
*name*, or a new mode silently gets the wrong one:

- **`MAX_RESET`** — with HSR on, soft resets can fire at every segment boundary
  on every env, so the budget is `max_episodes × segments_per_episode ×
  num_envs`. Without HSR it is exactly `max_episodes × num_envs`. Getting this
  wrong does not produce a warning: the run stops early on
  "max_reset exceeded", mid-training.
- **Perturbation** (`perturb ≠ off`) — perturbs the LSR reset goal, so it
  requires LSR. `main.py` rejects `--backward-goal recep|mixed` without
  `--enable-backward`, but only after the VLA has loaded, so `train.sh` checks
  it up front too.

`train.sh` therefore sets `_hsr_on` / `_lsr_on` inside each mode branch and keys
both off those, instead of re-listing mode names in three separate `case`
statements — which is how the earlier name-matching lists came to disagree with
each other.

## What changed, and what it invalidates

`noep` used to expand to LSR+HSR+`--reset-mode none` — i.e. what is now
`noep+LSR`. Runs whose directory name contains `-noep-` and predate this change
are **`noep+LSR` runs under the current definition.**

The `RUN_TAG` fragments for all three affected modes were renamed
(`LSR+HSR`→`HSRLSR`, `noep`→`HSRnoep`, new `noepLSR`→`HSRLSRnoep`) precisely so
this is unambiguous going forward: no new run directory can collide with a
historical one, and a tag now states which of HSR / LSR / noep are on rather
than requiring the reader to know which revision produced it. The old spellings
stay accepted as *input* (`LSR+HSR` for `HSR+LSR`) so existing launch scripts
keep working, but nothing emits them into a tag any more.

Consequences to be aware of:

- A `plot_config.json` mixing `-noep-` runs with new `-HSRnoep-` runs is
  comparing two different conditions. Old `-noep-` belongs in a `noep+LSR`
  group.
- `CKPT=` resume paths pointing into old `-noep-` directories still work — the
  checkpoint is unaffected — but the resumed segment will be tagged with the new
  scheme, so a resume chain can span both spellings. That is expected; state it
  in the plot config's group label.

## Caveat: `noep` without LSR drifts toward already-satisfied start states

**Recorded, not fixed.** No diagnostic column is written for this; it is here so
the curves are not misread.

Under `noep` (HSR + no episodic reset, no LSR) nothing ever returns a
*successfully placed* object to its initial state:

- HSR only respawns actors the detector flags. Both detectors test "fallen" or
  "outside the box" — an object resting on its target receptacle is neither, so
  it is never respawned.
- `--reset-mode none` means there is no `env.reset()` after the first episode.

So the batch drifts one-way toward "the objects are already on receptacles". Two
things follow:

1. **The start-state distribution is the previous segment's end state.** When
   the scheduler moves to a different receptacle this is a legitimate continuous
   rearrangement setting — arguably the point of the mode. When it draws the
   same `(obj, recep)` pair again, the task is satisfied before the policy acts.
2. **The metrics inflate in that case.** `reward_old` is zeroed at every task
   switch (`envs/wrapper.py::set_task`), so a segment that begins already
   satisfied pays out the full +1.0 potential term as soon as the object is
   grasped, and `rollout_success.csv`'s `success` reads 1.0 for a segment in
   which the policy did nothing.

This is almost certainly why `noep` originally bundled LSR: the backward
segment's "put X on table" goal is the only mechanism in the system that
restores the initial condition. Which means the `noep` vs `noep+LSR` comparison
is not only "is a reset policy worth the steps" but also "how much of `noep`'s
apparent success rate is this drift" — and with no `success_at_start` column
recorded, that second part has to be argued from the position plots
(`tools/plot_segment_positions.py --phase start`) rather than measured directly.

Use `--phase start` and look for the object cloud collapsing onto the receptacle
positions over training. If the two modes' curves differ mainly in the
already-satisfied fraction, add the diagnostic before drawing conclusions.
