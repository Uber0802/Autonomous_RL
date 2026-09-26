"""One-off (V0.99): make README usage-only; move rationale into doc/environments.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = ROOT / "README.md"
s = p.read_text()


def cut(start, end):
    """Return s[start:end] (end exclusive, by anchor text) and remove it from s."""
    global s
    i = s.index(start)
    j = s.index(end, i) if end else len(s)
    block = s[i:j]
    s = s[:i] + s[j:]
    return block


# ---------- Installation: pull rationale out ----------
rename_note = cut("> Renamed from `cronos_env`", "| Env | `setup.sh` args |")
bitexact = cut("> **Bit-exact note:**", "### 3. Run the setup script")
memory = cut("> **Memory budget", "`setup.sh` installs CRONOS")
blackwell_detail = cut("Both variants pin `torch==2.7.0+cu128`", "### 6. (Optional)")
ada_intro = cut("For Ada-class GPUs (L40S", "```bash\nconda create -n cronos_tf440_cu121")
tradeoffs = cut("Tradeoffs:\n", "### 7. Checkpoint portability")
portability = cut("A checkpoint is written by whichever env trained it", "Audit a checkpoint tree")
weights = cut("The LoRA *weights*", "## Quick Start")

s = s.replace("### 3. Run the setup script", """> Memory figures are historical reports; re-measure with `tools/bench_rollout.py`
> on your hardware. Why there are four envs, and what "bit-exact" means here:
> [`CRONOS/doc/environments.md`](CRONOS/doc/environments.md).

### 3. Run the setup script""", 1)
s = s.replace("### 6. (Optional) Lightweight OpenVLA-only env for Ada-class GPUs\n\n",
              "### 6. (Optional) Lightweight OpenVLA-only env for Ada-class GPUs\n\n"
              "For 48 GB Ada-class GPUs (L40S, RTX 6000 Ada, A6000), where the dual-VLA stack's\n"
              "OpenVLA-7B PPO does not fit. OpenVLA only — `--policy spatialvla` is unavailable.\n\n", 1)
s = s.replace("### 7. Checkpoint portability across envs\n\n",
              "### 7. Checkpoint portability across envs\n\n"
              "Checkpoints trained in a `tf447` env load in a `tf440` env and vice versa (handled\n"
              "by `SimplerEnv/simpler_env/policies/peft_compat.py`; details in\n"
              "[`CRONOS/doc/environments.md`](CRONOS/doc/environments.md)). "
              "Audit a checkpoint tree before a long eval, without loading a model:\n\n", 1)
s = s.replace("Audit a checkpoint tree before committing to a long eval, without loading a model\n(or peft, or torch):\n\n", "", 1)

# ---------- Training: history notes ----------
s = s.replace("""Output directory: defaults to `./$RUN_TAG`, created before launch and passed as an
**absolute** `--wandb-dir`. Override with `RUN_OUT_DIR=/data/runs/my-run`. Passing a
relative or not-yet-existing directory used to make wandb silently redirect the whole
run — every CSV, checkpoint and video — into `$TMPDIR`; `run_paths.py` now creates and
validates the directory up front and fails loudly if wandb ignores it.""",
"""Output directory: defaults to `./$RUN_TAG`, created before launch and passed as an
**absolute** `--wandb-dir`. Override with `RUN_OUT_DIR=/data/runs/my-run`. The run
fails at startup if wandb would write anywhere else.""")
noep_hist = cut("> ⚠️ **`noep` changed meaning, and the tags were renamed.**", "> ⚠️ Bare `noep`")
s = s.replace("""`main.py` warns at startup; `doc/reset_modes.md` says how to check for it.""",
"""`main.py` warns at startup; [`doc/reset_modes.md`](CRONOS/doc/reset_modes.md) says how to
> check for it. Run directories containing `-noep-` (before V0.93g) are `noep+LSR` runs.""")
motivation = cut("Both goals reuse tasks that already exist in the pool", "**EER (End-Effector Reset)**")
s = s.replace("**EER (End-Effector Reset)**", """Both goals reuse existing `put <obj> on <recep>` tasks — no new task string or reward
term. `off` is numerically identical to not having the option. Design notes:
[`doc/reset_modes.md`](CRONOS/doc/reset_modes.md).

**EER (End-Effector Reset)**""", 1)
eer_note = cut("> `reset_robot()` is also the only thing that zeroes", "Key training flags:")
s = s.replace("""`eer=on` emits a command line byte-identical to before the option existed, so
prior runs, resume paths and wandb dirs are unaffected.""", "`eer=on` adds no flag and no tag.")

# ---------- Eval: removed flags, accounting history ----------
removed = cut("Removed: `--eval-sequences`", "**Note:** the per-env rotation eval")
accounting = cut("#### Accounting fix (affects numbers from prior runs)", "## Tools")
s = s.replace("## Tools", """Sequential-eval numbers from before V0.91 are not comparable to later ones
([`doc/results/bug_reports.md`](CRONOS/doc/results/bug_reports.md)).

## Tools""", 1)

# ---------- link fixes ----------
s = s.replace("(CRONOS/doc/reports/eval_audit.md)", "(CRONOS/doc/results/bug_reports.md)")
s = s.replace("[`doc/eval_audit.md`](CRONOS/doc/results/bug_reports.md)", "[`doc/results/bug_reports.md`](CRONOS/doc/results/bug_reports.md)")
s = s.replace("See [`doc/grpo_autorl.md` §9](CRONOS/doc/reports/grpo_autorl.md)", "See [`doc/results/grpo.md`](CRONOS/doc/results/grpo.md) (Part 1 §9)")
s = s.replace("[`grpo_failure.md`](CRONOS/doc/reports/grpo_failure.md)", "[`doc/results/grpo.md`](CRONOS/doc/results/grpo.md)")
s = s.replace("see [`doc/reports/grpo_failure.md`](CRONOS/doc/reports/grpo_failure.md)", "see [`doc/results/grpo.md`](CRONOS/doc/results/grpo.md)")
p.write_text(s)

# ---------- doc/environments.md ----------
env = f"""# Conda environments and checkpoint portability

Index of the documentation: [`README.md`](README.md). How to install is in the
top-level README (*Installation*); this document says why there are four
environments, what they do and do not reproduce, and how checkpoints move between
them.

## Why four environments

{memory.replace('> ', '').replace('>', '').replace('See [Lightweight env](#6-optional-lightweight-openvla-only-env-for-ada-class-gpus).', 'See the lightweight env below.').strip()}

## Bit-exactness

{bitexact.replace('> ', '').strip()}

## Blackwell

{blackwell_detail.strip()}

## Lightweight OpenVLA-only env (Ada)

{ada_intro.strip()}

{tradeoffs.strip()}

## Checkpoint portability

{portability.strip()}

{weights.strip()}

## Env names

{rename_note.replace('> ', '').replace('>', '').strip()}
"""
env = env.replace("[main.py:385](CRONOS/main.py#L385)", "[main.py:385](../main.py#L385)")
env = env.replace("[eval_only.py:210](CRONOS/eval_only.py#L210)", "[eval_only.py:210](../eval_only.py#L210)")
env = env.replace("[`SimplerEnv/simpler_env/policies/peft_compat.py`](SimplerEnv/simpler_env/policies/peft_compat.py)",
                  "[`SimplerEnv/simpler_env/policies/peft_compat.py`](../../SimplerEnv/simpler_env/policies/peft_compat.py)")
(ROOT / "CRONOS/doc/environments.md").write_text(env)
print("ok")
