"""One-off (V0.99): doc/ is not released; make README and code self-contained."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "CRONOS"

p = ROOT / "README.md"
s = p.read_text()
R = [
    ("> **V0.99 — early release.** What changed in each version: [`CRONOS/doc/CHANGELOG.md`](CRONOS/doc/CHANGELOG.md). All other documentation: [`CRONOS/doc/`](CRONOS/doc/README.md).",
     "> **V0.99 — early release.** See [Version](#version) for known issues."),
    (" (**experimental** — it currently collapses, see [`doc/results/grpo.md`](CRONOS/doc/results/grpo.md));",
     " (**experimental** — see [Version](#version));"),
    (", documented column by column in [`CRONOS/doc/data_schemas.md`](CRONOS/doc/data_schemas.md).",
     " (see [Training-time outputs](#training-time-outputs))."),
    ("""> Memory figures are historical reports; re-measure with `tools/bench_rollout.py`
> on your hardware. Why there are four envs, and what "bit-exact" means here:
> [`CRONOS/doc/environments.md`](CRONOS/doc/environments.md).""",
     """> Memory figures are approximate; re-measure with `tools/bench_rollout.py` on your
> hardware. For bit-exact comparisons, run every arm of an ablation in the same env."""),
    ("""(handled
by `SimplerEnv/simpler_env/policies/peft_compat.py`; details in
[`CRONOS/doc/environments.md`](CRONOS/doc/environments.md)). """,
     """(handled
by `SimplerEnv/simpler_env/policies/peft_compat.py`). """),
    ("**Reset-mode legend** — full account in [`CRONOS/doc/reset_modes.md`](CRONOS/doc/reset_modes.md):",
     "**Reset-mode legend:**"),
    ("""> `main.py` warns at startup; [`doc/reset_modes.md`](CRONOS/doc/reset_modes.md) says how to
> check for it. Run directories containing `-noep-` (before V0.93g) are `noep+LSR` runs.""",
     """> `main.py` warns at startup. To check a run, compare the start-of-segment object
> positions (`segment_pose.csv`, `phase=start`) with the spawn positions."""),
    ("""term. `off` is numerically identical to not having the option. Design notes:
[`doc/reset_modes.md`](CRONOS/doc/reset_modes.md).""",
     """term. `off` is numerically identical to not having the option."""),
    (" See [`doc/results/grpo.md`](CRONOS/doc/results/grpo.md) (Part 1 §9) |", " |"),
    ("Sequential-eval numbers from before V0.91 are not comparable to later ones\n([`doc/results/bug_reports.md`](CRONOS/doc/results/bug_reports.md)).\n\n", ""),
    ("- `doc/` — All documentation, indexed by [`doc/README.md`](CRONOS/doc/README.md): changelog, experiment results, bug reports, design references\n", ""),
    ("""release; see [`CRONOS/doc/README.md`](CRONOS/doc/README.md#what-is-not-in-the-release).""",
     """release."""),
]
for a, b in R:
    assert a in s, a[:80]
    s = s.replace(a, b)

# remaining "see/Full design/Details/columns in [doc/...]" pointers
s = re.sub(r"\n?Full design: \[`doc/eval_sequential\.md`\]\(CRONOS/doc/eval_sequential\.md\)\.", "", s)
s = re.sub(r"\n?Details: \[`doc/eval_sequential\.md`\]\(CRONOS/doc/eval_sequential\.md\)\.\n", "\n", s)
s = s.replace("; columns in [`doc/data_schemas.md`](CRONOS/doc/data_schemas.md))", ")")
s = re.sub(r"Written to the run's `glob/` on every run; full column specs in\n\[`doc/data_schemas\.md`\]\(CRONOS/doc/data_schemas\.md\)\.",
           "Written to the run's `glob/` on every run.", s)

i = s.index("## Version and changelog")
s = s[:i] + """## Version

This is **CRONOS V0.99**, an early release. The version is defined in
[`CRONOS/version.py`](CRONOS/version.py) and stamped into every run's
`run_config.json`; compare runs only within one version.

Known issues:

- **GRPO is experimental.** `--alg-name grpo` degrades the SFT policy instead of
  improving it (the objective favours inaction). Use PPO, the default.
- **Training RNG is not yet isolated.** Standalone eval is reproducible per round,
  but training shares random generators between layout draws and action sampling,
  so two training runs that differ only on the policy side do not see the same
  start states.
"""
assert "doc/" not in s, [l for l in s.split("\n") if "doc/" in l]
p.write_text(s)

# ---- code / config comments and messages ----
CODE = [
    ("configs/four_group_sequential_2x2.yaml",
     "# requires multiple gym.make calls and is deferred to V0.5 per task_list.md.",
     "# requires multiple gym.make calls and is not supported yet."),
    ("configs/four_group_sequential_2x2.yaml",
     "Keys: evaluation/plan.py::EVAL_SETTING_SPEC, doc/eval_sequential.md.",
     "Keys: evaluation/plan.py::EVAL_SETTING_SPEC; README \"Evaluation (standalone)\"."),
    ("configs/two_group_sequential_2x2.yaml",
     "Keys: evaluation/plan.py::EVAL_SETTING_SPEC, doc/eval_sequential.md.",
     "Keys: evaluation/plan.py::EVAL_SETTING_SPEC; README \"Evaluation (standalone)\"."),
    ("evaluation/records.py",
     "Why derived files are rebuilt rather than appended: see doc/rng_and_io_notes.md.",
     "Derived files are rebuilt whole rather than appended, so a crash or a rerun can\nnever leave duplicate or partial aggregate rows."),
    ("evaluation/sequential.py", "semantics; doc/eval_audit.md)", "semantics)"),
    ("main.py", "    # and \"global\" are the same thing. See `doc/grpo_autorl.md` §\"是否需要 /std\".",
     "    # and \"global\" are the same thing."),
    ("main.py", "            # the policy did nothing. See doc/reset_modes.md.", "            # the policy did nothing."),
    ("main.py", "condition; see doc/reset_modes.md.", "condition; see README \"Reset-mode legend\"."),
    ("scripts/train.sh", "# Reset modes (see doc/reset_modes.md for the full account):", "# Reset modes (see README \"Reset-mode legend\"):"),
    ("scripts/train.sh", "    # startup; doc/reset_modes.md explains what it does to the metrics.",
     "    # startup: start states drift toward already-satisfied tasks, which inflates metrics."),
    ("scripts/train.sh", "(see doc/reset_modes.md).", "(see README \"Reset-mode legend\")."),
    ("training/buffer.py", "See `doc/grpo_autorl.md` for the trade-off; the short version is",
     "The short version of the trade-off is"),
]
for f, a, b in CODE:
    q = C / f
    t = q.read_text()
    assert a in t, (f, a)
    q.write_text(t.replace(a, b))
print("ok")
