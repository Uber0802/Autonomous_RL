"""One-off: merge doc/reports/*.md into doc/results/{grpo,bug_reports}.md (V0.99)."""
import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc"


def demote(text, drop_first_lines=0):
    """Push every Markdown heading one level down, leaving fenced code alone."""
    out, fence = [], False
    for line in text.split("\n")[drop_first_lines:]:
        if line.lstrip().startswith("```"):
            fence = not fence
        elif not fence and re.match(r"#{1,5} ", line):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


def body(name):
    s = (DOC / "reports" / name).read_text()
    # drop the title line and the "Index of these documents" / version pointer lines
    lines = s.split("\n")
    title = lines[0].lstrip("# ").strip()
    rest = [l for l in lines[1:] if not l.startswith(("Index of these documents",
                                                      "Describes the current tree"))]
    return title, "\n".join(rest)


# ---- grpo.md ----
t1, b1 = body("grpo_autorl.md")
t2, b2 = body("grpo_failure.md")
link_fix = lambda s: (s.replace("[`grpo_autorl.md`](grpo_autorl.md)", "[Part 1](#part-1--autorl-grpo-review-v093)")
                       .replace("[`grpo_failure.md`](grpo_failure.md)", "[Part 2](#part-2--why-grpo-collapses-v095)"))
grpo = f"""# GRPO in CRONOS — review and failure analysis

Index of the documentation: [`../README.md`](../README.md).

**Summary.** `--alg-name grpo` is implemented to match AutoRL bit-for-bit (Part 1),
and it collapses: in every measured run grasp and success fall to ≈ 0 while the PPO
controls improve (Part 2, and Q5 in [`paper_experiments.md`](paper_experiments.md)).
The cause is the objective itself — per-step normalization with no group baseline,
and `alg_grpo_fix` leaving idle trajectories at zero advantage — not a CRONOS port
error. The trajectory-level fix proposed in Part 2 §6 is **not implemented in V0.99**;
treat GRPO as experimental and do not cite GRPO-vs-PPO numbers.

Part 2 §5 corrects three claims of Part 1 (§4.1, §5, §6); Part 1 is kept unedited
as the record of what was believed at V0.93.

---

## Part 1 — AutoRL GRPO review (V0.93)

*Originally `grpo_autorl.md`: {t1}.*
{demote(link_fix(b1))}

---

## Part 2 — Why GRPO collapses (V0.95)

*Originally `grpo_failure.md`: {t2}.*
{demote(link_fix(b2))}
"""
(DOC / "results").mkdir(exist_ok=True)
(DOC / "results" / "grpo.md").write_text(grpo)

# ---- bug_reports.md ----
t3, b3 = body("eval_audit.md")
bugs = f"""# Bug reports

Index of the documentation: [`../README.md`](../README.md).

Every defect found so far whose fix **changed numbers produced by earlier
versions**, newest first, then the full sequential-eval audit. Details of each fix
are in [`../CHANGELOG.md`](../CHANGELOG.md); the GRPO failure is in
[`grpo.md`](grpo.md). Before comparing two runs, check that no row below falls
between their versions (the version a run was produced with is in its
`run_config.json`; see the stamp caveat at the top of the changelog).

| Fixed in | Area | Defect | What it invalidates |
|---|---|---|---|
| V0.95 | GRPO | Objective ranks inaction above attempting — [`grpo.md`](grpo.md) (open, not fixed) | Every GRPO result |
| V0.94d | Standalone eval | SpatialVLA checkpoints were evaluated with sampling (T = 0.6) instead of greedy | SpatialVLA standalone eval before V0.94d |
| V0.94 | Standalone eval | Multi-group eval broadcast group 0's objects to every env; orders, layouts and seeds redesigned | All standalone eval before V0.94 |
| V0.93g | Reset modes | `noep` silently included LSR; tags renamed | Grouping of old `-noep-` runs with new `-HSRnoep-` runs |
| V0.93a | wandb logging | Backward (reset) segments were averaged into `rollout/<task>/*` | wandb per-task rollout curves of LSR runs |
| V0.93 | Output dirs | Relative `--wandb-dir` could put a whole run under `/tmp` | (data loss, not numbers) |
| V0.92 | Training, EER off | `_elapsed_steps` never reset: masks 0, GAE degenerated to `returns = reward` | Every `--no-reset-robot` run before V0.92 |
| V0.91 | Sequential eval | Tasks after the first ran permanently truncated: time-averaged success, leaked grasp flags | Sequential-eval numbers before V0.91 (below) |
| V0.4.2 | SpatialVLA eval | Sampled instead of deterministic eval in `train.sh` | SpatialVLA eval before V0.4.2 |
| V0.4.1 | wandb logging | Only the last minibatch of each PPO update was logged | Logged loss / grad curves before V0.4.1 |
| V0.4c | Reset modes | `LSR` / `HSR` meant different flags; HSR `MAX_RESET` ~3× too small at T1280 | LSR / HSR runs before V0.4c |
| V0.4a | HSR | HSR-only silently reset the robot; default respawn scope changed to `per_env` | HSR runs before V0.4a |
| V0.3a | HSR, init | Respawn used hard-coded slots and unrotated carrots; stale first observation; eval clobbered non-episodic state | HSR and `reset_mode=none` runs before V0.3a |

---

## Sequential-eval accounting defect (V0.91)

*Originally `eval_audit.md`: {t3}.*
{demote(b3)}
"""
(DOC / "results" / "bug_reports.md").write_text(bugs)
print("ok")
