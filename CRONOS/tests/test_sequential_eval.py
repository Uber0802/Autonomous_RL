"""`SequentialEvaluator` against a fake env. Needs torch (CPU is enough), no GPU, no ManiSkill.

The fake env reproduces the two properties the loop depends on:
- ManiSkill's TimeLimit semantics: `truncated = elapsed >= max_episode_steps`,
  recomputed every step, cleared only by reset / begin_segment;
- explicit `layout_ids` applied at reset;
- an outcome that depends on the sampled actions, so resume/shard equality
  also proves the per-unit policy reseed.
"""

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from envs.scheduler import GroupState
import json

from evaluation.plan import EvalSettings, build_plan, build_scenes, plan_fingerprint

L1, L2 = 432, 4
LTT = 16 * 16 * 4 * L1 * L2


class FakeUnwrapped:
    def __init__(self, n):
        self.num_envs = n
        self._elapsed_steps = torch.zeros(n, dtype=torch.long)
        self.select_pos_ids = torch.zeros(n, dtype=torch.long)
        self.select_quat_ids = torch.zeros(n, dtype=torch.long)
        self.select_overlay_ids = torch.zeros(n, dtype=torch.long)
        self.last_layout_rand_ids = None

    def get_all_slot_poses(self):
        n = self.num_envs
        p = self.select_pos_ids.float().reshape(n, 1).repeat(1, 3)
        q = self.select_quat_ids.float().reshape(n, 1).repeat(1, 4)
        return {"obj": [(p, q), (p, q)], "recep": [(p, q), (p, q)], "gripper": (p, q),
                "obj_names": [["o"] * n] * 2, "recep_names": [["r"] * n] * 2}


class FakeWrapper:
    def __init__(self, n, max_steps, reset_counter_on_begin=True, crash_at_step=None):
        self.num_envs = n
        self.max_steps = max_steps
        self.env = types.SimpleNamespace(unwrapped=FakeUnwrapped(n))
        self.reset_calls = []
        self.tasks = None
        self.reset_counter_on_begin = reset_counter_on_begin
        self.crash_at_step = crash_at_step
        self.total_steps = 0
        self.acc = torch.zeros(n, dtype=torch.long)

    def _obs(self):
        return torch.zeros(self.num_envs, 4, 4, 3, dtype=torch.uint8)

    def reset(self, same_init=True, obj_set_override=None, group_idx_override=None,
              skip_scheduler=False, layout_ids=None):
        u = self.env.unwrapped
        self.reset_calls.append(dict(obj_set=obj_set_override, group_idx_override=group_idx_override,
                                     layout_ids=list(layout_ids)))
        rand = torch.tensor([x % LTT for x in layout_ids], dtype=torch.long)
        u.last_layout_rand_ids = rand
        u.select_pos_ids = (rand // L2) % L1
        u.select_quat_ids = rand % L2
        u._elapsed_steps[:] = 0
        self.acc.zero_()
        return self._obs(), None, None

    def set_task(self, objs, receps):
        self.tasks = [f"put {o} on {r}" for o, r in zip(objs, receps)]

    def begin_segment(self):
        if self.reset_counter_on_begin:
            self.env.unwrapped._elapsed_steps[:] = 0
        self.acc.zero_()

    def get_obs_image(self):
        return self._obs()

    def get_language_instructions(self):
        return list(self.tasks)

    def step(self, action):
        self.total_steps += 1
        if self.crash_at_step is not None and self.total_steps == self.crash_at_step:
            raise RuntimeError("simulated crash")
        u = self.env.unwrapped
        u._elapsed_steps += 1
        self.acc += action.long().sum(1)
        truncated = u._elapsed_steps >= self.max_steps
        info = {}
        if truncated.any():
            # Outcome depends on the start pose, the task and the sampled actions,
            # so reproducing it needs the same layouts AND the same sampling stream.
            succ = [bool((int(u.select_pos_ids[i]) + len(self.tasks[i]) + int(self.acc[i])) % 3 == 0)
                    for i in range(self.num_envs)]
            info["episode"] = {"success": succ, "consecutive_grasp": succ, "is_src_obj_grasped": succ}
        return self._obs(), torch.zeros(self.num_envs, 1), truncated.reshape(-1, 1), info


def groups(n_groups=2, envs=4):
    out = []
    for g in range(n_groups):
        seq = [f"put g{g}o{t // 2} on g{g}r{t % 2}" for t in range(4)]
        out.append(GroupState.from_sequence(f"group_{g}", seq, list(seq), envs))
    return out


def sampling_policy(extra_draws=0):
    def act(obs, instr):
        if extra_draws:
            torch.rand(extra_draws)
        return torch.randint(0, 255, (obs.shape[0], 7))
    return act


@unittest.skipIf(torch is None, "torch not installed")
class TestSequentialEvaluator(unittest.TestCase):
    SEG = 3

    def setUp(self):
        # images_to_video is imported lazily from ManiSkill; stub it.
        written = self.written = []
        mod = types.ModuleType("mani_skill.utils.visualization.misc")
        mod.images_to_video = lambda frames, d, name, fps=10, verbose=False: written.append((d, name, len(frames)))
        for name in ("mani_skill", "mani_skill.utils", "mani_skill.utils.visualization"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["mani_skill.utils.visualization.misc"] = mod

    def make(self, d, schedule="parallel", rounds="all", pose_sets=1, seed=0, record_video=False,
             begin_ok=True, crash_at_step=None, act=None, resume=False):
        from evaluation.sequential import SequentialEvaluator
        settings = EvalSettings(pose_sets=pose_sets, rounds=rounds, scene_schedule=schedule,
                                record_video=record_video, video_envs_per_block=1)
        plan = build_plan(build_scenes(groups(), 8), settings, num_envs=8, segment_len=self.SEG, seed=seed)
        plan.fingerprint = plan_fingerprint(plan, dict(checkpoint="ckpt"))
        env = FakeWrapper(8, self.SEG, reset_counter_on_begin=begin_ok, crash_at_step=crash_at_step)
        ev = SequentialEvaluator(plan=plan, env=env, glob_dir=Path(d), act_fn=act or sampling_policy(),
                                 obj_set="rand", resume=resume)
        return ev, env, plan

    def run_eval(self, d, **kw):
        ev, env, plan = self.make(d, **kw)
        return ev.run(), env, plan

    def read(self, d, name):
        with open(Path(d) / name) as f:
            return list(csv.DictReader(f))

    def test_parallel_run_counts_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            res, env, plan = self.run_eval(d)
            self.assertTrue(res["status"]["complete"] and res["status"]["coverage_match"])
            trials = self.read(d, "eval_per_trial.csv")
            self.assertEqual(len(trials), 2 * 6 * 4 * 8)             # domains x rounds x slots x envs
            self.assertEqual(len(env.reset_calls), 2 * 6)
            self.assertEqual([c["obj_set"] for c in env.reset_calls], ["rand"] * 6 + ["rand_ood"] * 6)
            layouts = self.read(d, "eval_layouts.csv")
            self.assertEqual(len(layouts), 2 * 6 * 8)
            for r in layouts:
                self.assertEqual(int(r["rand_id"]), int(r["layout_id"]) % LTT)
            self.assertEqual(len(self.read(d, "eval_segment_pose.csv")), 2 * 6 * 4 * 2 * 8 * 5)
            self.assertTrue((Path(d) / "eval_success.csv").exists())
            self.assertIn("status: COMPLETE", (Path(d) / "eval_report.txt").read_text())
            t0 = [r for r in trials if r["seq_idx"] == "0" and r["task_idx"] == "0" and r["eval_kind"] == "in_domain"]
            self.assertEqual([r["order"] for r in t0], ["ABCD", "BCDA", "CDAB", "DABC"] * 2)
            orders = {r["order"] for r in trials if r["group"] == "group_0" and r["eval_kind"] == "in_domain"}
            self.assertEqual(len(orders), 24)
            for r in trials:
                self.assertEqual(r["group"], f"group_{int(r['env_idx']) // 4}")

    def test_selected_rounds_reproduce_the_full_run(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            self.run_eval(d1, pose_sets=2)
            self.run_eval(d2, pose_sets=2, rounds="3-4,9")
            full = {(r["eval_kind"], r["seq_idx"], r["task_idx"], r["env_idx"]): r for r in self.read(d1, "eval_per_trial.csv")}
            part = self.read(d2, "eval_per_trial.csv")
            self.assertEqual(len(part), 2 * 3 * 4 * 8)
            for r in part:
                self.assertEqual(r, full[(r["eval_kind"], r["seq_idx"], r["task_idx"], r["env_idx"])])

    def test_layouts_do_not_depend_on_the_policy(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            self.run_eval(d1)
            self.run_eval(d2, act=sampling_policy(extra_draws=5))
            self.assertEqual((Path(d1) / "eval_layouts.csv").read_text(), (Path(d2) / "eval_layouts.csv").read_text())

    def test_crash_then_resume_equals_uninterrupted(self):
        with tempfile.TemporaryDirectory() as ref, tempfile.TemporaryDirectory() as d:
            self.run_eval(ref, pose_sets=2)
            # 4 slots x SEG steps per round; crash in the middle of round 2 (0-based) of in_domain
            crash = 2 * 4 * self.SEG + 5
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                self.run_eval(d, pose_sets=2, crash_at_step=crash)
            status = json.loads((Path(d) / "eval_status.json").read_text())
            self.assertEqual((status["complete"], status["done_units"]), (False, 2))
            self.assertFalse((Path(d) / "eval_success.csv").exists())
            # a fresh (non-resume) run into the same directory is refused
            with self.assertRaisesRegex(ValueError, "--eval-resume"):
                self.run_eval(d, pose_sets=2)
            # resuming with another selection or another seed is refused
            with self.assertRaisesRegex(ValueError, "rounds/domains differ"):
                self.run_eval(d, pose_sets=2, rounds="0-5", resume=True)
            with self.assertRaisesRegex(ValueError, "not the same eval"):
                self.run_eval(d, pose_sets=2, seed=1, resume=True)
            # videos are not part of the fingerprint: turning them on for the resume is allowed
            res, env, _ = self.run_eval(d, pose_sets=2, resume=True, record_video=True)
            self.assertTrue(res["status"]["complete"])
            self.assertEqual(len(env.reset_calls), 2 * 12 - 2)       # the 2 done units were skipped
            self.assertTrue(self.written)
            for name in ("eval_per_trial.csv", "eval_layouts.csv", "eval_segment_pose.csv",
                         "eval_sequence_summary.csv", "eval_coverage.csv", "eval_success.csv"):
                self.assertEqual((Path(ref) / name).read_text(), (Path(d) / name).read_text(), name)
            plan = json.loads((Path(d) / "eval_plan.json").read_text())
            self.assertEqual(len(plan["resumes"]), 1)

    def test_serial_schedule(self):
        with tempfile.TemporaryDirectory() as d:
            res, env, _ = self.run_eval(d, schedule="serial")
            self.assertEqual({c["group_idx_override"] for c in env.reset_calls}, {0, 1})
            self.assertEqual(len(self.read(d, "eval_per_trial.csv")), 2 * 2 * 6 * 4 * 8)
            self.assertTrue(res["status"]["complete"] and res["status"]["coverage_match"])

    def test_same_poses_within_a_set_new_poses_per_set(self):
        with tempfile.TemporaryDirectory() as d:
            _, env, _ = self.run_eval(d, pose_sets=2, rounds="0-11")
            ids = [c["layout_ids"] for c in env.reset_calls[:12]]       # in_domain
            self.assertTrue(all(x == ids[0] for x in ids[:6]))
            self.assertTrue(all(x == ids[6] for x in ids[6:12]))
            self.assertNotEqual(ids[0], ids[6])
            by_slot = {}
            for r in self.read(d, "eval_layouts.csv"):
                by_slot.setdefault((r["eval_kind"], r["pose_set"], r["layout_slot"]), set()).add((r["pos_id"], r["quat_id"]))
            self.assertTrue(all(len(v) == 1 for v in by_slot.values()))

    def test_missing_begin_segment_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "terminal report"):
                self.run_eval(d, begin_ok=False)

    def test_video_one_per_block(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_eval(d, record_video=True, rounds="0")
            # 2 domains x 4 slots x (1 env per block: 4 blocks x 2 scenes)
            self.assertEqual(len(self.written), 2 * 4 * 8)
            self.assertTrue(all(n == self.SEG + 1 for _, _, n in self.written))
            self.assertTrue(any("/in_domain/training/round0/task3" in d_ for d_, _, _ in self.written))


if __name__ == "__main__":
    unittest.main()
