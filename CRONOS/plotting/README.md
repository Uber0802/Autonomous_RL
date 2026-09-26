# CRONOS plotting and statistics tools

Offline tools that read a run's CSVs (`rollout_success.csv`, `segment_pose.csv`,
`eval_success.csv`, `eval_per_trial.csv`) and compare several runs from one
`--config` JSON, plus statistics (`mcnemar_pair.py`, `parse_autorl_eval.py`) and
GPU figure renderers (`render_*.py`, see [`figures/README.md`](figures/README.md)).

**Status:** not part of the V0.99 release (tracked on the internal branch,
gitignored on the release branch); planned for a later release. Run every tool from `CRONOS/`, e.g.
`python plotting/plot_eval_success.py --config plotting/configs/plot_config.json`.
Dependencies: `scripts/requirements_plot.txt` (installed by `setup.sh`).

### Visualization

CRONOS ships a per-run live dashboard plus tools that read a run's CSVs directly
and compare several runs from one `--config` JSON:

| Question | Tool | Reads |
|---|---|---|
| Is this run healthy right now? | `../tools/plot_run_trends.py` (ships with the release; see the top-level README) | wandb + `eval_success.csv` |
| How does training success evolve, segment by segment? | `plot_rollout_success.py` | `rollout_success.csv` |
| Where do objects start / end up? | `plot_segment_positions.py` | `segment_pose.csv` |
| Does success depend on a task's position in the sequence? | `plot_sequence_eval.py` | `eval_per_trial.csv` |
| Eval success curves across runs | `plot_eval_success.py` | `eval_success.csv` |

#### Per-segment training curves — `plotting/plot_rollout_success.py`

Per-80-step success rate straight from the run's `rollout_success.csv`. That file
holds one row per (episode, segment, env) written at every `task_len` boundary,
so "per-80" is its native granularity — each point is the mean over one
segment's `num_envs` rows, with no resampling. Needs no wandb access.

```bash
python plotting/plot_rollout_success.py --run-dir <RUN_OUT_DIR>/wandb/run-*/glob
```

| Flag | Default | Description |
|---|---|---|
| `--config` | — | JSON with several groups of runs; one curve per group, mean ± 1 std band across that group's series. See [Comparing runs](#comparing-several-runs----config) |
| `--direction` | `forward` | `forward` \| `backward` \| `backward_recep` \| `all`. Reset segments score `success` against a different goal, so mixing them in reads as a ~50% collapse that is pure alternation artifact — see [`doc/data_schemas.md`](../doc/data_schemas.md). `all` draws one series per direction. |
| `--by` | `none` | Add a second panel split by `task` \| `group` \| `obj` \| `recep` |
| `--x-axis` | `total_steps` | `total_steps` \| `segment` \| `episode` |
| `--smooth` | 5 | Rolling-mean window in segments (1 disables) |
| `--no-per-group` | off | `--config` mode: write only the all-group main figure, skipping the per-group ones |
| `--no-reset-split` | off | Draw every per-group curve whole instead of splitting it at its resets |

The top panel overlays success, `consecutive_grasp` and `is_src_obj_grasped`;
the success-vs-grasp gap is the placement-collapse diagnostic. Empty cells (an
env that did not report at a boundary) are read as NaN and excluded from the
means rather than as zeros.

##### `--config` mode writes a figure set, split at the resets

    <name>_rollout_success.png            all groups, one hue each, drawn whole
    <name>_rollout_success_<group>.png    one group, split at ITS OWN resets

A training curve is not one continuous experiment: every reset re-randomizes the
batch, so the rate the policy reaches *within* an inter-reset stretch and the
trend *across* stretches are different quantities. One unbroken line hides the
difference — a within-stretch climb followed by the reset's drop reads as noise.
Each per-group figure therefore splits the curve at that group's own reset
boundaries and gives **each piece its own colour** (the `tab10` head — blue,
orange, green, red — so four pieces are unmistakable and eight are still
separable). A dotted rule marks each reset.

The hue is free to carry the reset index here because a per-group figure holds
exactly one group and the legend's title names it. An earlier version shaded the
pieces light → dark within the group's hue, which kept the tie to the main
figure's colour but left adjacent pieces differing only in lightness — not
enough to tell apart at a glance, which is the whole point of the split.

The split stays out of the main figure deliberately: there the hue is what tells
the conditions apart, and spending it on the reset index would leave nothing to
identify a group by.

**T1280 and up only.** At `segment_len = 80` an episode holds `episode_len / 80`
segments, so T1280 is the first horizon whose 16 make a shape — T320's 4 and
T80's 1 do not. A shorter run is drawn whole and says so on stderr, so one
figure set can mix split and unsplit groups without either being mistaken for
the other. A resume chain is only as splittable as its coarsest leg, and a run
whose resets outpace its horizon (HSR fires soft resets at segment boundaries)
is caught by a second guard on the measured piece length.

```
[rollout] noep+LSR T2560: split into 4 inter-reset pieces (episode_len=2560)
[rollout] baseline T320: drawn whole — episode_len=320 < 1280 (T1280+ only)
```

#### Per-segment position distribution — `plotting/plot_segment_positions.py`

Top-down xy positions of the objects (`obj`) and receptacles (`recep`) from
`segment_pose.csv`. Each boundary is recorded twice — once **before** its
HSR/EER resets (`phase=end`, the steady state the policy produced) and once
**after** (`phase=start`, the initial state the next segment begins from, and
after a full `env.reset()` at an episode boundary). `--phase` defaults to
`start`, which is the distribution the forward policy actually faces.

```bash
python plotting/plot_segment_positions.py --run-dir <RUN_OUT_DIR>/wandb/run-*/glob
python plotting/plot_segment_positions.py --config my_runs.json            # several groups
```

**One PNG per figure, never a grid** (`<...>_obj.png`, `<...>_recep.png`, one pair
per group in `--config` mode). There are no px / py axis labels — every figure
is the same top-down table view — and no title; the counts that used to be in
the title (points, synthetic share, fraction below `low_z = 0.7`) are printed on
stderr. Every figure one invocation writes shares **one view range and one count
scale**, which is what makes separate PNGs comparable.

**Density.** The spawn lattice stacks thousands of poses on one xy, so points
that land close together are merged into `--bin-size` cells:

| `--density` | Look |
|---|---|
| `emphasis` (default) | every point as a plain scatter, plus an enlarged, darker marker on each cell holding ≥ `--dense-min` points — the figure reads like a scatter and only the stacks (spawn / reset sites an untouched object never left) stand out. The key bottom-left gives the counts |
| `size` | one marker per cell, area ∝ count |
| `shade` | one filled cell per `--bin-size` square, darker = more (log scale) |
| `scatter` | every point, unmerged; coloured by episode for a single run |

`--dense-min` defaults to 0.6 % of a figure's points (at least 5): about 50 for a
whole-run T2560 figure, 5 for a per-task one.

**What to draw.**

| Flag | Output |
|---|---|
| *(none)* | one figure per actor kind (per group) |
| `--per-task` | one figure per (task, kind) in a `..._per_task/` directory, showing only the task's own object / receptacle (matched by model name). Add `--all-slots` to keep every actor of the envs running the task. Rows without a task (synthetic start draws) are left out, so on a pre-`phase` run use `--phase end` |
| `--color-by item` | `--config` only. Objects only, one colour per (object, step range): object 1 / range 1, object 1 / range 2, object 2 / range 1, … One figure per group (`..._obj_by_item.png`); combines with `--per-task` |
| `--color-by scene` | `--config` only. For every scene (YAML group): one figure per object, one per receptacle, and one of the whole scene — 5 for a 2×2 scene — each with one colour per step range. Written to `..._by_scene/<scene>_<n>_<actor>.png` |

`--color-by` takes the actor order from the **scene config**: object *k* of a
scene is the *k*-th entry of its `obj:` list (receptacles: `recep:`), mapped to
`model_name` through ManiSkill's `assets/carrot/more_{carrot,plate}/model_db.json`,
and scenes own env ranges in `num_envs` order. The config is the run's
`experiment_config.yaml` snapshot, or else `run_config.json`'s `config_path` as it
is in this checkout now (stderr says which). PyYAML is used when installed;
otherwise a small parser reads the `groups:` keys it needs. Fallbacks, each
announced with `[warn]`: without the config, scenes come from the run's
`rollout_success.csv` (`group` per env) and the actor order from the slot index;
without the model tables, models are matched by their number prefix
(`007_ketchup bottle_1` → 7). Rows no source can place are drawn as scene
`unknown` — if a figure is named `unknown_…`, read the warnings.

**Comparing stretches of training.** In `--config` mode `--step-range` takes
several comma-separated ranges; each becomes its own figure (or colour, with
`--color-by`), all on one view and count scale, with `_steps<LO>-<HI>` in the
filename:

```bash
python plotting/plot_segment_positions.py --config q2.json --phase end \
    --step-range 0:163840,163841:327680 --color-by scene
```

| Flag | Description |
|---|---|
| `--config` | JSON with several groups of runs. See [Comparing runs](#comparing-several-runs----config) |
| `--phase` | `start` (default) — the state each segment *begins* from, after that boundary's HSR/EER resets and after `env.reset()` at an episode boundary. `end` — the steady state the policy produced, before them. `all` — both |
| `--actor-kind` / `--slot` / `--model` / `--task` | Narrow to one actor class, logical slot, model-name substring, or task substring |
| `--step-range LO:HI[,LO:HI...]` | Keep only boundaries whose `total_steps` falls in the range. **The default is not the whole run** (`DEFAULT_STEP_RANGE` in the tool), so a longer run is cropped unless you widen it or pass `--step-range all`. Either side may be left open (`:HI`, `LO:`). Several ranges: `--config` only. What it kept is always reported on stderr |
| `--segment` / `--episode-range LO:HI` / `--last-episodes N` | Narrow in time |
| `--forward-only` | Join `rollout_success.csv` on (episode, segment, env) and keep only forward segments — worth using under a mode with LSR (`LSR`, `HSR+LSR`, `noep+LSR`), where half the segment ends are reset-goal states. A no-op without LSR |
| `--density` / `--bin-size` / `--dense-min` | See *Density* above |
| `--per-task` / `--all-slots` / `--color-by` | See *What to draw* above |
| `--hexbin` | viridis hexbin (overrides `--density`) |
| `--workspace-scale` | View size as a multiple of the preset's spawn region (default 3) |
| `--no-clip` | Plot the full coordinate range instead of a view robust to escaped actors |
| `--workspace=X0,X1,Y0,Y1` | Overlay a rectangle, e.g. `workspace_aabb` bounds being validated. Use the `=` form — the bounds are negative and argparse would read them as a flag |

Config keys (CLI wins): `actor_kind`, `phase`, `step_range` (string, comma list
allowed), `workspace_scale`, `density`, `bin_size`, `dense_min`, `per_task`,
`color_by`.

Hidden slots (a group declaring fewer objects than the batch-wide N) are written
as NaN by design and are dropped, with the count reported. The gripper is
recorded but not plotted (EER pins it).

#### Sequence-eval success by position — `plotting/plot_sequence_eval.py`

Bar charts from a standalone sequential eval's `eval_per_trial.csv`. Two
series: **by position** (x = the task's position in the round, 1–4) and **by
task** (x = the task, pooled over positions), one bar per group. A pooled
position figure holds one domain (solid bars); per-task and by-task figures
hold both domains in one bar — **out-of-domain solid, in-domain hatched over
it** — in the group's colour. Bars are the mean over a group's series (seeds);
no spread is drawn.

```bash
python plotting/plot_sequence_eval.py --run-dir <EVAL_OUT_DIR>/wandb/run-*/glob
python plotting/plot_sequence_eval.py --config plotting/configs/plot_sequence_example.json --seq-kind random
```

A ready-to-edit config is [`plotting/configs/plot_sequence_example.json`](configs/plot_sequence_example.json);
it also shows a group whose seed was evaluated as two round shards.

| Output | Contents |
|---|---|
| `<name>_seq_position_<metric>_<kind>_{in_domain,out_of_domain}.png` | by position, pooled over tasks, one figure per domain |
| `<name>_seq_position_<metric>_<kind>_per_task/…_<task>.png` | by position, one per task, both domains (`--no-per-task` to skip) |
| `<name>_seq_position.csv` | every plotted position number (`metric`, `seq_kind` columns), with `n_trials` |
| `<name>_seq_task_<metric>_<kind>.png` | by task, both domains |
| `<name>_seq_task_<metric>_<kind>_per_scene/…_<scene>.png` | by task, one per scene (`--no-per-scene` to skip) |
| `<name>_seq_task.csv` | the by-task numbers, pooled rows under `scene = __all__` |

| Flag | Default | Description |
|---|---|---|
| `--metric` | `success success_chained` | one or more of `success` (each task on its own) \| `success_chained` (AND along the round) \| `grasp` \| `obj_grasped` |
| `--seq-kind` | `each` | `each` (seen / training and unseen / random orders get their own figures) \| `pooled` (one set, kinds averaged) \| `seen` / `training` \| `unseen` / `random` |
| `--no-final-eval` | off | skip the figures of the checkpoint's own last training eval |
| `--out-dir` / `--name` | run dir / `eval` | where and under which prefix to write |

Training runs have no `eval_per_trial.csv`; point it at an eval glob dir. In
`--config` mode the `runs` entries are eval glob dirs, and a list entry is a set
of round shards of one eval.

#### Comparing several runs — `--config`

The per-run tools take a `--config` JSON describing several experiment
groups. A group is one curve (success), one set of figures (positions) or one
bar colour (sequence eval); each entry in
its `runs` list is one series, typically a seed. **A `runs` entry that is itself
a list is a resume chain** — those run dirs are stitched into one continuous
series, with the child winning at any overlapping `total_steps`.

```bash
python plotting/plot_rollout_success.py   --config plotting/configs/plot_runs_example.json
python plotting/plot_segment_positions.py --config plotting/configs/plot_runs_example.json --actor-kind obj
```

```json
{
  "out_dir": "reports/figures/2026-08-26",
  "name": "perturb_ablation",
  "groups": [
    { "label": "noep baseline",
      "runs": [ ["/data/runs/T320-seed0/.../glob", "/data/runs/T1280-seed0/.../glob"],
                "/data/runs/T320-seed1/.../glob" ] },
    { "label": "noep + PTBmixed",
      "runs": [ "/data/runs/T320-PTBmixed0.5-seed0/.../glob" ] }
  ]
}
```

Schema and full docs in [`plotting/plot_common.py`](plot_common.py);
a ready-to-edit copy is [`plotting/configs/plot_runs_example.json`](configs/plot_runs_example.json).
Top-level keys starting with `_` are ignored, so the example carries its own notes.

##### A run with no data is skipped, not fatal

A config routinely points at runs that cannot answer the question asked of them:
an eval-only run writes no `rollout_success.csv`, `--no-record-segment-pose`
writes no `segment_pose.csv`, a run still in its first episode has files that
exist but are empty, and an older run may predate a column the current code
reads. In `--config` mode every plot tool names each such run on stderr and
skip it, so one of them does not cost every other group its figure:

```
[warn] group 'noep baseline': /data/runs/.../glob/rollout_success.csv: no rollout_success.csv
[warn] group 'noep baseline': /data/runs/.../glob/rollout_success.csv: rollout_success.csv is missing column(s) ['total_steps'] — written by an older version?
```

A tool fails only when **no** group produced anything — that is a configuration
error rather than a missing file, and the error names the usual causes. The
single-run paths (`--run-dir` / `--csv`) still fail loudly: there you named one
specific file, so "not found" is the answer to the question you asked.

##### Runs recorded before the `phase` split

An older run has no `phase=start` rows, and they cannot be recovered — the env
draws initial poses with `torch.randint` on the global CUDA generator (which the
VLA's action sampling also consumes) and HSR with `np.random.choice` (which the
PPO minibatch shuffle also consumes), and neither index is logged. A same-seed
replay would have to reproduce the whole training bit-for-bit.

The *distribution* those draws came from is recoverable, though: the sampler is
uniform over `xyz_configs`, a deterministic table `envs/suite.py` builds from the
(N, M) preset with no randomness. `plot_segment_positions.py` therefore
reconstructs the start cloud by drawing uniformly from that same table **as many
times as the run actually reset** — `total_resets` counts exactly one draw per
per-env respawn, so a T80 run (128 episodes × 64 envs = 8,192 draws) and a T2560
one (4 × 64 = 256) produce clouds of the right relative density instead of a
misleading uniform one.

Only the boundaries that really are an `env.reset()` draw are synthesized; every
other start is carried over from the previous segment's recorded end when the
boundary provably moved nothing (no HSR, no EER). With EER on, every start is
synthesized. Synthetic points are drawn like recorded ones and their share is
reported on stderr; they carry no task, so `--per-task` leaves them out.
`--no-synth` skips them instead, `--synth-seed` makes the draw reproducible.
Synthesis needs `envs.suite` importable (numpy + transforms3d).

For the shipped 2×2 preset that table is 432 ordered configs = 18 distinct
four-point geometries × 4! slot permutations, occupying just 16 xy positions
(a 4×4 corner sub-grid) — the workspace is 0.15 × 0.15 m and the spacing
constraint is 0.12 m, so the points are pushed into the corners.

#### Cross-run aggregator — `plotting/plot_eval_success.py`

Reads multiple `eval_success.csv` files (one per seed × config × condition), aggregates mean ± std, and writes 4 main PNGs (ID/OOD × Steps/Resets) plus 2 gap PNGs (success vs grasp).

```bash
# 1. Edit plotting/configs/plot_config.json — list run groups (label → list of CSV paths)
# 2. Run the aggregator
python plotting/plot_eval_success.py --config plotting/configs/plot_config.json
```

`plot_config.json` schema (one entry per logical comparison curve):

```json
{
  "out_dir": "reports/aggregated/2026-06-18",
  "name": "four_group_T320_vs_T1280",
  "groups": [
    {
      "label": "T320 normal (3 seeds)",
      "csv_paths": [
        "/path/to/CRONOS-openvla-…-T320-normal-seed0/glob/eval_success.csv",
        "/path/to/CRONOS-openvla-…-T320-normal-seed1/glob/eval_success.csv",
        "/path/to/CRONOS-openvla-…-T320-normal-seed2/glob/eval_success.csv"
      ]
    },
    {
      "label": "T1280 noep (resumed from T320 seed1)",
      "csv_paths": [
        [
          "/path/to/CRONOS-openvla-…-T320-normal-seed1/glob/eval_success.csv",
          "/path/to/CRONOS-openvla-…-T1280-noep-seed1/glob/eval_success.csv"
        ]
      ]
    }
  ]
}
```

A `csv_paths` entry that is a **list** (chain) is the **resume chain**: parent CSV + child CSV; the aggregator dedupes overlapping `(total_steps, eval_kind, group, task)` rows and keeps the child at the seam. Adding a new run = append a path; no code changes.

Outputs:

| File | Contents |
|---|---|
| `<name>_aggregated.csv` | long-form per-group/eval_kind/x_axis mean ± std |
| `<name>_summary.csv` | final-value mean ± std at the rightmost eval per group × eval_kind |
| `<name>_<eval_kind>_<x_axis>.png` | 4 main curves (ID/OOD × total_steps/total_resets) |
| `<name>_gap_<eval_kind>.png` | success-vs-grasp overlay (placement-collapse diagnostic) |

The `plot_*.py` tools require `pandas`, `numpy`, `matplotlib`; pinned versions are in `scripts/requirements_plot.txt` and pulled in by `setup.sh` automatically.

