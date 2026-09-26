"""Execute an `EvalPlan`: the standalone single-task / sequential eval loop.

Shared by `eval_only.py` and `main.py --eval-single / --eval-sequential`.

The unit of work is (domain, pass, round). Per unit:
  1. env.reset with the pose set's layout ids, set_task      (same start poses for every
                                                               round of the set)
  2. torch reseeded from `policy|seed|domain|round|pass`       (sampling independent of
                                                               what ran before)
  3. for each task slot: set_task (+ begin_segment after the first), `segment_len`
     steps, terminal success/grasp per env                     (AutoRL render_seq
                                                               semantics)
  4. rows appended to eval_per_trial / eval_layouts / eval_segment_pose, then
     eval_status.json updated — so a crash loses at most the unit in progress.

In `parallel` schedule every scene steps in the same batch on its own env range,
reset with its own objects/background (the per-env options `set_group_specs`
builds, same as training). In `serial` schedule each scene gets its own pass
with the scene broadcast to every env (`group_idx_override`).

Resume (`resume=True`): the directory's existing `eval_plan.json` must have the
same fingerprint and the same units; completed units are skipped, a partial unit's
rows are dropped and the unit is rerun.
"""

from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from tqdm import tqdm

from envs import rng_streams
from evaluation import outputs as O
from evaluation import records as R
from evaluation.plan import OOD_OBJ_SET, EvalPlan, fingerprint_differences, unit_key

TERMINAL_KEYS = ("success", "consecutive_grasp", "is_src_obj_grasped")


class SequentialEvaluator:
    def __init__(self, *, plan: EvalPlan, env, glob_dir: Path,
                 act_fn: Callable, chunk_fn: Optional[Callable] = None, action_chunk: int = 1,
                 prep_rollout: Optional[Callable] = None,
                 obj_set: str = "rand", episode: int = 0, total_steps: int = 0,
                 video_workers: int = 4, report_name: str = "eval_report.txt", resume: bool = False):
        """
        act_fn(obs, instructions) -> action ids [num_envs, A]
        chunk_fn(obs, instructions, K) -> action ids [num_envs, K, A]   (action_chunk > 1 only)
        episode / total_steps: checkpoint progress, written to every CSV row
        """
        self.plan = plan
        self.settings = plan.settings
        self.env = env
        self.glob_dir = Path(glob_dir)
        self.act_fn = act_fn
        self.chunk_fn = chunk_fn
        self.action_chunk = max(1, int(action_chunk))
        if self.action_chunk > 1 and chunk_fn is None:
            raise ValueError("action_chunk > 1 needs a chunk_fn (SpatialVLA only)")
        self.prep_rollout = prep_rollout or (lambda: None)
        self.obj_set = obj_set
        self.episode = episode
        self.total_steps = total_steps
        self.num_envs = plan.num_envs
        self.segment_len = plan.segment_len
        self.video_workers = video_workers
        self.report_name = report_name
        self.resume = resume
        self.pose_writer = (R.SegmentPoseWriter(self.glob_dir / R.POSES, R.EVAL_POSE_EXTRA)
                            if self.settings.record_pose else None)

    # ------------------------------------------------------------------ setup
    def _plan_dict(self) -> dict:
        d = self.plan.to_dict()
        d["checkpoint_progress"] = dict(episode=self.episode, total_steps=self.total_steps)
        return d

    def _prepare_directory(self) -> set:
        """Write or check eval_plan.json; return the units already done."""
        O.refuse_training_dir(self.glob_dir)
        plan_path = self.glob_dir / "eval_plan.json"
        new = self._plan_dict()
        if not self.resume:
            if (self.glob_dir / R.PER_TRIAL).exists():
                raise ValueError(f"{self.glob_dir} already holds eval results; use --eval-resume to continue it")
            R.atomic_write_text(plan_path, json.dumps(new, indent=2) + "\n")
            return set()

        if not plan_path.exists():
            raise ValueError(f"--eval-resume: {plan_path} not found")
        old = json.loads(plan_path.read_text())
        if old["fingerprint"].get("hash") != new["fingerprint"].get("hash"):
            diffs = fingerprint_differences(old["fingerprint"].get("body", {}), new["fingerprint"].get("body", {}))
            raise ValueError("--eval-resume: this command is not the same eval as the one in the directory:\n  " +
                             "\n  ".join(diffs or ["fingerprint hash differs"]))
        if sorted(map(unit_key, old["units"])) != sorted(map(unit_key, new["units"])):
            raise ValueError("--eval-resume: rounds/domains differ from the original run. Resume with the "
                             f"original selection (rounds={old['settings']['rounds']!r}, domains="
                             f"{old['settings']['domains']}); to add rounds, run them into a new directory "
                             f"and merge with tools/rebuild_eval_outputs.py")
        units = O.planned_units([old])
        trials = R.read_rows(self.glob_dir / R.PER_TRIAL)
        done, partial, overfull = O.completed_units(trials, units)
        if overfull:
            raise ValueError(f"--eval-resume: units with more rows than planned {overfull}; "
                             f"the directory was written by more than one run")
        dropped = O.truncate_to_units(self.glob_dir, done)
        old.setdefault("resumes", []).append(dict(
            at=datetime.now(timezone.utc).isoformat(), done_units=len(done),
            dropped_partial_units=[list(k) for k in sorted(partial)], dropped_rows=dropped,
            provenance=new.get("provenance")))
        R.atomic_write_text(plan_path, json.dumps(old, indent=2) + "\n")
        print(f"[eval resume] {len(done)}/{len(units)} units done; rerunning partial {sorted(partial)}; "
              f"dropped rows {dropped}")
        return done

    # ------------------------------------------------------------------ helpers
    def _video_envs(self, p, seq) -> List[int]:
        """First k envs of every order block of every scene (-1 = all)."""
        k = self.settings.video_envs_per_block
        envs = []
        for si in range(len(p.scenes)):
            for lo, hi, _ in p.scene_orders(seq, si):
                envs.extend(range(lo, hi if k < 0 else min(hi, lo + k)))
        return envs

    def _layout_rows(self, p, seq, eval_kind, ids, orders):
        """What the env applied at this reset, next to the id that asked for it."""
        u = self.env.env.unwrapped
        applied = getattr(u, "last_layout_rand_ids", None)
        if applied is None:
            raise RuntimeError("layout_ids were passed to reset but the env did not apply them "
                               "(obj_set 'fixed', or an env without layout_ids support)")
        pos, quat = u.select_pos_ids.tolist(), u.select_quat_ids.tolist()
        overlay, applied = u.select_overlay_ids.tolist(), applied.tolist()
        key = rng_streams.layout_key(self.plan.layout_seed, eval_kind, seq.pose_set)
        slots = p.layout_slots(self.num_envs, self.settings.layout_slots)
        rows = []
        for sc, (lo, hi) in zip(p.scenes, p.env_ranges):
            for e in range(lo, hi):
                rows.append(dict(
                    eval_kind=eval_kind, seq_idx=seq.seq_idx, seq_kind=seq.seq_kind, pose_set=seq.pose_set,
                    pass_label=p.label, group=sc.name, env_idx=e, scene_env_idx=e - lo,
                    order=orders[e], layout_slot=slots[e], layout_key=key, layout_id=ids[e],
                    rand_id=applied[e], pos_id=pos[e], quat_id=quat[e], overlay_id=overlay[e],
                ))
        return rows

    def _record_pose(self, phase, instructions, *, eval_kind, p, seq, task_idx, groups, obj_set):
        want = self.settings.pose_phase
        if self.pose_writer is None or (want != "both" and want != phase):
            return
        unwrapped = self.env.env.unwrapped
        if not hasattr(unwrapped, "get_all_slot_poses"):
            raise AttributeError(
                f"record_pose needs an env exposing get_all_slot_poses() (PickPlaceNxM-v1); "
                f"got {type(unwrapped).__name__}. Set eval.record_pose: false.")
        n = self.num_envs
        self.pose_writer.write(
            unwrapped.get_all_slot_poses(), instructions, n,
            episode=seq.seq_idx + 1, segment=task_idx + 1, phase=phase, total_steps=self.total_steps,
            extra_values=[[eval_kind] * n, [seq.seq_kind] * n, [seq.seq_idx] * n, [seq.pose_set] * n,
                          [task_idx] * n, groups, [obj_set] * n, [p.label] * n],
        )

    def _rollout(self, obs, instructions, video_envs):
        """Step `segment_len` times; return (terminal info dict, frames)."""
        frames: Dict[int, list] = {i: [] for i in video_envs}
        subset = len(video_envs) < self.num_envs
        terminal, events, last_event_step = None, 0, -1
        K = self.action_chunk
        step_i = 0
        pbar = tqdm(total=self.segment_len, leave=False, desc="eval")
        while step_i < self.segment_len:
            if K == 1:
                chunk = self.act_fn(obs, instructions).unsqueeze(1)
            else:
                chunk = self.chunk_fn(obs, instructions, K)
            for k in range(K):
                if step_i >= self.segment_len:
                    break
                if video_envs:
                    # One device→host copy per step, not one per env.
                    arr = obs.cpu().numpy()
                    for e in video_envs:
                        frames[e].append(arr[e].copy() if subset else arr[e])
                obs, _reward, _truncated, info = self.env.step(chunk[:, k])
                if "episode" in info:
                    events += 1
                    terminal, last_event_step = info["episode"], step_i
                step_i += 1
                pbar.update(1)
        pbar.close()

        # Contract: exactly one terminal report, on the final step. Anything else
        # means `_elapsed_steps` was not reset at the segment start, and the
        # values below would come from the wrong timestep. Fail rather than pad.
        if events != 1 or last_event_step != self.segment_len - 1:
            raise RuntimeError(
                f"eval segment produced {events} terminal report(s), last at step "
                f"{last_event_step} (expected exactly 1, at step {self.segment_len - 1}). "
                f"The env's elapsed-step counter was not reset at the segment start; "
                f"see CronosWrapper.begin_segment.")
        for key in TERMINAL_KEYS:
            if len(terminal.get(key, ())) != self.num_envs:
                raise RuntimeError(f"terminal '{key}' has {len(terminal.get(key, ()))} values "
                                   f"for {self.num_envs} envs")
        if video_envs:
            arr = obs.cpu().numpy()
            for e in video_envs:
                frames[e].append(arr[e].copy() if subset else arr[e])
        return terminal, frames

    @staticmethod
    def _reseed_policy(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ---------------------------------------------------------------------- run
    def _run_unit(self, eval_kind, obj_set, p, seq, submit_video):
        n = self.num_envs
        video_envs = self._video_envs(p, seq) if self.settings.record_video else []
        chain = [1.0] * n
        for task_idx in range(seq.n_slots):
            objs, receps, tasks, groups, orders = p.env_assignment(seq, task_idx, n)
            if task_idx == 0:
                self.prep_rollout()
                ids = p.layout_ids(eval_kind, seq.pose_set, n, self.plan.layout_seed, self.settings.layout_slots)
                obs, _, _ = self.env.reset(obj_set_override=obj_set, group_idx_override=p.group_idx_override,
                                           skip_scheduler=True, layout_ids=ids)
                self.env.set_task(objs, receps)
                layout_rows = self._layout_rows(p, seq, eval_kind, ids, orders)
                self._reseed_policy(rng_streams.policy_seed(self.plan.policy_seed, eval_kind, seq.seq_idx, p.label))
                unit_trials = []
            else:
                self.env.set_task(objs, receps)
                self.env.begin_segment()
                obs = self.env.get_obs_image()
            instructions = self.env.get_language_instructions()
            pose_kw = dict(eval_kind=eval_kind, p=p, seq=seq, task_idx=task_idx, groups=groups, obj_set=obj_set)
            self._record_pose("start", instructions, **pose_kw)
            terminal, frames = self._rollout(obs, instructions, video_envs)
            self._record_pose("end", self.env.get_language_instructions(), **pose_kw)

            prefix = f"{eval_kind}_seq{seq.seq_idx}_task{task_idx}"
            rows = []
            for e in range(n):
                s = float(terminal["success"][e])
                chain[e] *= s
                rows.append(dict(
                    seq_idx=seq.seq_idx, task_idx=task_idx, obj_set=obj_set, task=tasks[e], env_idx=e,
                    success=s, success_chained=chain[e],
                    grasp=float(terminal["consecutive_grasp"][e]),
                    obj_grasped=float(terminal["is_src_obj_grasped"][e]),
                    prefix=prefix, eval_kind=eval_kind, seq_kind=seq.seq_kind, group=groups[e],
                    obj=objs[e], recep=receps[e], order=orders[e], pose_set=seq.pose_set,
                    cycle_idx=seq.cycle_idx, pass_label=p.label,
                    episode=self.episode, total_steps=self.total_steps,
                ))
            unit_trials.extend(rows)
            self._print_slot(p, seq, task_idx, rows)
            if video_envs:
                vdir = (self.glob_dir / "eval_videos" / eval_kind / seq.seq_kind
                        / f"round{seq.seq_idx}" / f"task{task_idx}")
                vdir.mkdir(parents=True, exist_ok=True)
                for e in video_envs:
                    name = (f"{groups[e]}-{orders[e]}-env{e}-{objs[e]}_{receps[e]}"
                            f"-s{int(rows[e]['success'])}").replace(" ", "_")
                    submit_video(frames[e], vdir, name, len(video_envs))
        # Source rows of a unit are written together, after its last slot, so a
        # crash mid-unit leaves no per-trial rows for it (pose rows are written as
        # they happen and are truncated on resume).
        R.append_rows(self.glob_dir / R.LAYOUTS, R.LAYOUT_FIELDS, layout_rows)
        R.append_rows(self.glob_dir / R.PER_TRIAL, R.PER_TRIAL_FIELDS, (R.format_trial(r) for r in unit_trials))

    @torch.no_grad()
    def run(self) -> dict:
        plan = self.plan
        done = self._prepare_directory()
        plan_dict = json.loads((self.glob_dir / "eval_plan.json").read_text())
        print(plan.describe())

        executor = ThreadPoolExecutor(max_workers=self.video_workers) if self.settings.record_video else None
        futures = []

        def submit_video(frames, vdir, name, per_slot):
            from mani_skill.utils.visualization.misc import images_to_video
            futures.append(executor.submit(images_to_video, frames, str(vdir), name, fps=10, verbose=False))
            # Queued jobs pin their frames in memory; bound the queue.
            if len(futures) > 2 * per_slot:
                wait(futures, return_when=FIRST_COMPLETED)
            for f in [f for f in futures if f.done()]:
                f.result()          # surface writer exceptions now
                futures.remove(f)

        try:
            for eval_kind in self.settings.domains:
                obj_set = self.obj_set if eval_kind == "in_domain" else OOD_OBJ_SET
                print(f"\n[eval] {eval_kind} (obj_set={obj_set})")
                for p in plan.passes:
                    for seq in p.sequences:
                        if (eval_kind, p.label, seq.seq_idx) in done:
                            continue
                        self._run_unit(eval_kind, obj_set, p, seq, submit_video)
                        O.write_status(self.glob_dir, [plan_dict],
                                       R.read_rows(self.glob_dir / R.PER_TRIAL))
        finally:
            if executor is not None:
                for f in futures:
                    f.result()
                executor.shutdown(wait=True)

        result = O.rebuild_outputs(
            self.glob_dir, [plan_dict], report_name=self.report_name,
            header=f"Standalone eval — mode={self.settings.mode} schedule={self.settings.scene_schedule} "
                   f"episode={self.episode} total_steps={self.total_steps}\n")
        print("\n" + result["report"])
        if not (result["status"]["complete"] and result["status"]["coverage_match"]):
            raise RuntimeError(f"eval finished its loop but is not complete: {result['status']}")
        return result

    def _print_slot(self, p, seq, task_idx, rows):
        by_group: Dict[str, List[dict]] = {}
        for r in rows:
            by_group.setdefault(r["group"], []).append(r)
        parts = []
        for g, rs in by_group.items():
            succ = sum(r["success"] for r in rs) / len(rs)
            chn = sum(r["success_chained"] for r in rs) / len(rs)
            tasks = sorted({r["task"] for r in rs})
            what = tasks[0] if len(tasks) == 1 else "/".join(sorted({r["order"] for r in rs}))
            parts.append(f"{g}: {what} s={succ:.3f} c={chn:.3f}")
        print(f"  [{p.label}] round{seq.seq_idx}(set{seq.pose_set},{seq.seq_kind}) task{task_idx}  "
              + " | ".join(parts))
