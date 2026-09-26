"""Standalone-eval planning, RNG streams, settings and records. No GPU, no torch.

Run from the CRONOS directory:
    python -m pytest tests/ -q
"""

import csv
import io
import json
import itertools
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from envs import rng_streams
from envs.config import load_cronos_config, resolve_symbolic_task
from envs.scheduler import GroupState
from evaluation import outputs as O
from evaluation import records as R
from evaluation.plan import (EvalSettings, PlanError, build_plan, build_scenes, cli_flag_given,
                             coverage_from_units, fingerprint_differences, order_label, parse_rounds,
                             plan_fingerprint, resolve_eval_settings, training_order, unit_key)
from evaluation.records import SegmentPoseWriter

ROOT = Path(__file__).resolve().parents[1]


def fake_groups(n_groups=4, envs=16, n_tasks=4):
    """GroupStates with readable, globally unique task names."""
    out = []
    for g in range(n_groups):
        name = "ABCDEFGH"[g]
        seq = [f"put {name}obj{t // 2 + 1} on {name}recep{t % 2 + 1}" for t in range(n_tasks)]
        out.append(GroupState.from_sequence(f"group_{name}", seq, list(seq), envs))
    return out


def config_groups(path):
    """Resolve a real YAML config's groups with placeholder names (no model_db needed)."""
    cfg = load_cronos_config(ROOT / path)
    states = []
    for gi, g in enumerate(cfg.groups):
        on = {f"obj{i+1}": f"g{gi}_obj{idx}" for i, idx in enumerate(g.obj)}
        rn = {f"recep{i+1}": f"g{gi}_recep{idx}" for i, idx in enumerate(g.recep)}
        seq = [resolve_symbolic_task(t, on, rn) for t in g.task_sequence]
        ev = [resolve_symbolic_task(t, on, rn) for t in g.eval_tasks]
        states.append(GroupState.from_sequence(g.name, seq, ev, g.num_envs))
    env_n = max(len(g.obj) for g in cfg.groups)
    env_m = max(len(g.recep) for g in cfg.groups)
    return cfg, states, env_n, env_m


class TestRngStreams(unittest.TestCase):
    def test_random_orders_prefix_stable_distinct_non_identity(self):
        for n_tasks in (2, 3, 4):
            full = rng_streams.random_orders(n_tasks, seed=7, count=math.factorial(n_tasks) - 1)
            self.assertEqual(len(set(full)), len(full))
            self.assertNotIn(tuple(range(n_tasks)), full)
            for k in range(len(full)):
                self.assertEqual(rng_streams.random_orders(n_tasks, 7, k), full[:k])

    def test_random_orders_large_pool_prefix_stable(self):
        a = rng_streams.random_orders(9, seed=1, count=6)
        b = rng_streams.random_orders(9, seed=1, count=3)
        self.assertEqual(a[:3], b)
        self.assertEqual(len(set(a)), 6)

    def test_random_orders_too_many(self):
        with self.assertRaises(ValueError):
            rng_streams.random_orders(2, 0, 2)

    def test_layout_ids_prefix_stable_and_keyed(self):
        a = rng_streams.layout_ids(0, "in_domain", 64)
        self.assertEqual(rng_streams.layout_ids(0, "in_domain", 16), a[:16])
        self.assertTrue(all(0 <= x < 2 ** rng_streams.LAYOUT_ID_BITS for x in a))
        self.assertNotEqual(rng_streams.layout_ids(1, "in_domain", 16), a[:16])
        self.assertNotEqual(rng_streams.layout_ids(0, "out_of_domain", 16), a[:16])

    def test_layout_ids_uniform_enough_over_configs(self):
        # The env maps id -> id % ltt -> (pos, quat); check pos/quat marginals are flat.
        l1, l2, ltt = 432, 4, 16 * 16 * 4 * 432 * 4
        ids = []
        for seed in range(400):
            ids += rng_streams.layout_ids(seed, "in_domain", 64)
        rand = [x % ltt for x in ids]
        quat = np.bincount(np.array([x % l2 for x in rand]), minlength=l2)
        self.assertLess(quat.max() / quat.min(), 1.1)
        self.assertEqual(len({(x // l2) % l1 for x in rand}), l1)

    def test_cycles_partition_all_orders(self):
        for n in (3, 4, 5):
            cyc = rng_streams.random_cycles(n, 0, rng_streams.n_cycles(n) - 1)
            self.assertNotIn(tuple(range(n)), cyc)
            self.assertTrue(all(c[0] == 0 for c in cyc))
            every = [o for c in [tuple(range(n))] + cyc for o in rng_streams.rotations_of(c)]
            self.assertEqual(sorted(every), sorted(itertools.permutations(range(n))))
            for k in range(len(cyc)):
                self.assertEqual(rng_streams.random_cycles(n, 0, k), cyc[:k])
        with self.assertRaises(ValueError):
            rng_streams.random_cycles(4, 0, 6)
        self.assertEqual(len(set(rng_streams.random_cycles(10, 3, 8))), 8)   # rejection path


class TestSettings(unittest.TestCase):
    def args(self, **kw):
        base = dict(eval_mode="sequential", eval_pose_sets=1, eval_rounds="all", eval_sequence_seed=-1,
                    eval_layout_seed=-1, eval_policy_seed=-1, eval_layout_slots=-1,
                    eval_scene_schedule="parallel", eval_domains="in_domain,out_of_domain",
                    record_video=True, video_envs_per_block=-1, record_eval_pose=True,
                    eval_pose_phase="both", eval_ood=True)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_precedence_cli_over_yaml_over_default(self):
        s = resolve_eval_settings(self.args(eval_rounds="3-5"), {"rounds": "0-1", "scene_schedule": "serial"},
                                  ["--eval-rounds", "3-5"])
        self.assertEqual((s.rounds, s.sources["rounds"]), ("3-5", "cli"))
        self.assertEqual((s.scene_schedule, s.sources["scene_schedule"]), ("serial", "yaml"))
        self.assertEqual((s.pose_phase, s.sources["pose_phase"]), ("both", "default"))

    def test_yaml_rounds_forms(self):
        for value, expected in ((3, "3"), ("3-5", "3-5"), ([0, "6-11"], "0,6-11"), ("all", "all")):
            self.assertEqual(resolve_eval_settings(self.args(), {"rounds": value}, []).rounds, expected)

    def test_negated_bool_flag_and_legacy_eval_ood(self):
        s = resolve_eval_settings(self.args(record_video=False, eval_ood=False),
                                  {"record_video": True}, ["--no-record-video", "--no-eval-ood"])
        self.assertFalse(s.record_video)
        self.assertEqual(s.domains, ["in_domain"])

    def test_flag_match_is_token_exact(self):
        self.assertTrue(cli_flag_given("eval_rounds", ["--eval-rounds=4"]))
        self.assertFalse(cli_flag_given("eval_rounds", ["--eval-rounds-foo", "4"]))

    def test_invalid_values(self):
        for bad in ({"mode": "x"}, {"scene_schedule": "x"}, {"domains": ["x"]}, {"pose_sets": 0},
                    {"pose_phase": "x"}, {"layout_slots": 0}, {"rounds": "a-b"}, {"nope": 1}):
            with self.assertRaises(PlanError, msg=str(bad)):
                resolve_eval_settings(object(), bad, [])

    def test_removed_keys_point_to_rounds(self):
        for key in ("num_sequences", "include_training_sequence", "training_orders", "random_orders", "layout_rng"):
            with self.assertRaisesRegex(PlanError, "removed"):
                resolve_eval_settings(object(), {key: 1}, [])

    def test_parse_rounds(self):
        self.assertEqual(parse_rounds("all", 6), [0, 1, 2, 3, 4, 5])
        self.assertEqual(parse_rounds("0,3-4,10-", 12), [0, 3, 4, 10, 11])
        with self.assertRaisesRegex(PlanError, "past the last round 5"):
            parse_rounds("6", 6)
        with self.assertRaisesRegex(PlanError, "empty range"):
            parse_rounds("4-2", 6)


class TestConfigs(unittest.TestCase):
    def test_all_shipped_configs_load(self):
        for p in sorted((ROOT / "configs").rglob("*.yaml")):
            load_cronos_config(p)

    def test_four_group_eval_block(self):
        cfg = load_cronos_config(ROOT / "configs/four_group_sequential_2x2.yaml")
        s = resolve_eval_settings(object(), cfg.eval, [])
        self.assertEqual((s.pose_sets, s.rounds, s.scene_schedule, s.video_envs_per_block, s.layout_slots),
                         (1, "all", "parallel", 1, -1))

    def test_bad_eval_block_rejected_at_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text("eval:\n  pose_sets: 0\ngroups:\n  - name: a\n    num_envs: 4\n"
                         "    obj: [7, 2]\n    recep: [1, 2]\n")
            with self.assertRaisesRegex(ValueError, "V30"):
                load_cronos_config(p)

    def test_training_order_half_train(self):
        self.assertEqual(training_order(["b", "a", "b"], ["a", "b", "c", "d"]), ["b", "a", "c", "d"])
        self.assertEqual(training_order(["a", "b", "c", "d"], ["d", "c", "b", "a"]), ["a", "b", "c", "d"])


ROT = ["ABCD", "BCDA", "CDAB", "DABC"]
ALL24 = sorted("".join(chr(65 + i) for i in q) for q in itertools.permutations(range(4)))


def settings(**kw):
    return EvalSettings(**kw)


class TestPlan(unittest.TestCase):
    def plan(self, schedule="parallel", groups=None, num_envs=64, seed=0, fan_out=True, **kw):
        groups = groups or fake_groups()
        return build_plan(build_scenes(groups, num_envs, fan_out=fan_out), settings(scene_schedule=schedule, **kw),
                          num_envs=num_envs, segment_len=80, seed=seed)

    def labels(self, seq):
        return [order_label(o) for o in seq.orders]

    # ---- rounds ------------------------------------------------------------
    def test_default_is_six_rounds_all_24_orders(self):
        p = self.plan()
        seqs = p.passes[0].sequences
        self.assertEqual([s.seq_idx for s in seqs], [0, 1, 2, 3, 4, 5])
        self.assertEqual([s.seq_kind for s in seqs], ["training"] + ["random"] * 5)
        self.assertEqual(self.labels(seqs[0]), ROT)
        self.assertEqual(sorted(o for s in seqs for o in self.labels(s)), ALL24)

    def test_pose_sets_repeat_the_same_cycles(self):
        p = self.plan(pose_sets=3)
        seqs = p.passes[0].sequences
        self.assertEqual(len(seqs), 18)
        self.assertEqual([s.pose_set for s in seqs], [r // 6 for r in range(18)])
        self.assertEqual([s.seq_kind for s in seqs[6:12]], ["training"] + ["random"] * 5)
        for r in range(6):
            self.assertEqual(self.labels(seqs[r]), self.labels(seqs[r + 6]))
            self.assertEqual(self.labels(seqs[r]), self.labels(seqs[r + 12]))

    def test_round_is_a_function_of_its_number(self):
        full = {s.seq_idx: (s.seq_kind, s.pose_set, self.labels(s)) for s in self.plan(pose_sets=2).passes[0].sequences}
        for spec in ("3-5", "0,7", "8-", "11"):
            part = self.plan(pose_sets=2, rounds=spec).passes[0].sequences
            for s in part:
                self.assertEqual((s.seq_kind, s.pose_set, self.labels(s)), full[s.seq_idx])
        with self.assertRaisesRegex(PlanError, "past the last round 11"):
            self.plan(pose_sets=2, rounds="12")

    def test_each_round_is_position_balanced(self):
        p = self.plan()
        for seq in p.passes[0].sequences:
            for t in range(4):
                _, _, tasks, _, _ = p.passes[0].env_assignment(seq, t, 64)
                for g in range(4):
                    self.assertEqual(len(set(tasks[16 * g:16 * g + 16:4])), 4)

    def test_random_rounds_never_train_orders(self):
        for seed in range(5):
            for s in self.plan(seed=seed).passes[0].sequences[1:]:
                self.assertFalse(set(ROT) & set(self.labels(s)))

    def test_seed_changes_random_cycles_only(self):
        a = [self.labels(s) for s in self.plan(seed=0).passes[0].sequences]
        b = [self.labels(s) for s in self.plan(seed=1).passes[0].sequences]
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[1:], b[1:])

    def test_training_round_matches_scheduler_fan_out(self):
        """Round 0 gives every env exactly the task sequence TaskScheduler hands it
        over one training episode (4 segments)."""
        from envs.scheduler import TaskScheduler
        sched = TaskScheduler(fake_groups(), mode="sequential", num_envs=64, fan_out=True)
        trained = [[] for _ in range(64)]
        for _ in range(4):
            objs, receps = sched.get_next_tasks()
            for e in range(64):
                trained[e].append(f"put {objs[e]} on {receps[e]}")
            sched.update_index()
        p = self.plan()
        seq0 = p.passes[0].sequences[0]
        evaluated = [[] for _ in range(64)]
        for t in range(4):
            _, _, tasks, _, _ = p.passes[0].env_assignment(seq0, t, 64)
            for e in range(64):
                evaluated[e].append(tasks[e])
        self.assertEqual(evaluated, trained)

    def test_every_env_gets_its_own_scene_task(self):
        p = self.plan()
        for seq in p.passes[0].sequences:
            for t in range(4):
                objs, receps, tasks, groups, orders = p.passes[0].env_assignment(seq, t, 64)
                for e in range(64):
                    letter = "ABCD"[e // 16]
                    self.assertEqual(groups[e], f"group_{letter}")
                    self.assertTrue(objs[e].startswith(letter) and receps[e].startswith(letter))
                    self.assertEqual(p.scenes[e // 16].training_order.index(tasks[e]), ord(orders[e][t]) - 65)

    # ---- poses -------------------------------------------------------------
    def test_start_poses_shared_within_a_set_and_new_per_set(self):
        par, ser = self.plan(pose_sets=2), self.plan("serial", pose_sets=2)
        p = par.passes[0]
        self.assertEqual(p.layout_slots(64, -1)[:16], [0, 1, 2, 3] * 4)
        ids0 = p.layout_ids("in_domain", 0, 64, par.layout_seed, -1)
        ids1 = p.layout_ids("in_domain", 1, 64, par.layout_seed, -1)
        self.assertEqual(len(set(ids0)), 4)
        self.assertEqual(ids0[:16], ids0[48:64])               # same poses in every scene
        self.assertFalse(set(ids0) & set(ids1))                # set 1 is a new set of poses
        sids = ser.passes[0].layout_ids("in_domain", 0, 64, ser.layout_seed, -1)
        self.assertEqual(sids[:4], ids0[:4])                   # serial slots 0-3 = parallel poses
        one = self.plan(layout_slots=1).passes[0].layout_ids("in_domain", 0, 64, 0, 1)
        self.assertEqual(len(set(one)), 1)

    # ---- single mode ---------------------------------------------------------
    def test_single_mode_runs_tasks_side_by_side(self):
        p = self.plan(mode="single", pose_sets=2)
        seqs = p.passes[0].sequences
        self.assertEqual([(s.seq_idx, s.seq_kind, s.pose_set, s.n_slots) for s in seqs],
                         [(0, "single", 0, 1), (1, "single", 1, 1)])
        _, _, tasks, _, orders = p.passes[0].env_assignment(seqs[0], 0, 64)
        self.assertEqual(orders[:16], ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4)
        cov = [r for r in p.coverage() if r["eval_kind"] == "in_domain" and r["group"] == "group_A"][0]
        self.assertEqual((cov["trials"], cov["trials_per_task"], cov["resets"], cov["layouts"]), (32, 8, 32, 8))
        # a non-fan-out scene is still split in single mode
        q = self.plan(mode="single", fan_out=False)
        _, _, _, _, orders = q.passes[0].env_assignment(q.passes[0].sequences[0], 0, 64)
        self.assertEqual(len(set(orders[:16])), 4)

    # ---- coverage ------------------------------------------------------------
    def test_coverage_parallel_and_serial(self):
        cov = {(r["seq_kind"], r["group"]): r for r in self.plan(pose_sets=2).coverage() if r["eval_kind"] == "in_domain"}
        t, r = cov[("training", "group_A")], cov[("random", "group_A")]
        self.assertEqual((t["rounds"], t["pose_sets"], t["orders"], t["trials"], t["trials_per_task"], t["resets"], t["layouts"]),
                         (2, 2, 4, 128, 32, 32, 8))
        self.assertEqual((r["rounds"], r["pose_sets"], r["orders"], r["trials"], r["trials_per_task"], r["resets"], r["layouts"]),
                         (10, 2, 20, 640, 160, 160, 8))
        ser = {(r["seq_kind"], r["group"]): r for r in self.plan("serial").coverage() if r["eval_kind"] == "in_domain"}
        self.assertEqual((ser[("random", "group_C")]["trials"], ser[("random", "group_C")]["layouts"]), (1280, 16))
        self.assertEqual(self.plan().total_env_steps(), 6 * 4 * 80 * 64 * 2)
        self.assertEqual(self.plan("serial").total_env_steps(), 4 * self.plan().total_env_steps())

    def test_coverage_from_units_adds_up_across_shards(self):
        full = self.plan(pose_sets=2).unit_rows()
        a = self.plan(pose_sets=2, rounds="0-4").unit_rows()
        b = self.plan(pose_sets=2, rounds="5-").unit_rows()
        key = lambda rows: sorted(map(unit_key, rows))
        self.assertEqual(key(a + b), key(full))
        norm = lambda rows: sorted((r["eval_kind"], r["seq_kind"], r["group"], r["trials"], r["resets"], r["layouts"])
                                   for r in rows)
        self.assertEqual(norm(coverage_from_units(a + b)), norm(coverage_from_units(full)))

    # ---- refusals ------------------------------------------------------------
    def test_no_padding_partition(self):
        with self.assertRaisesRegex(PlanError, "does not pad"):
            build_scenes(fake_groups(envs=16), 60)

    def test_parallel_requires_equal_task_counts(self):
        groups = fake_groups(2, 8, 4)
        groups[1] = GroupState.from_sequence("group_B", groups[1].task_sequence[:2], groups[1].task_sequence[:2], 8)
        with self.assertRaisesRegex(PlanError, "serial"):
            self.plan(groups=groups, num_envs=16)
        p = self.plan("serial", groups=groups, num_envs=16)
        self.assertEqual([len(q.sequences) for q in p.passes], [6, 1])   # 2 tasks: one cycle

    def test_all_rounds_refused_for_large_pools(self):
        tasks = [f"put o{i} on r" for i in range(9)]
        g = [GroupState.from_sequence("g", tasks, tasks, 9)]
        with self.assertRaisesRegex(PlanError, "select rounds explicitly"):
            self.plan(groups=g, num_envs=9)
        self.assertEqual(len(self.plan(rounds="0-3", groups=g, num_envs=9).passes[0].sequences), 4)

    def test_block_split_must_be_even(self):
        groups = [GroupState.from_sequence("g", fake_groups(1, 6, 4)[0].task_sequence,
                                           fake_groups(1, 6, 4)[0].task_sequence, 6)]
        with self.assertRaisesRegex(PlanError, "split evenly"):
            self.plan(groups=groups, num_envs=6)

    def test_unsplit_scene_when_training_did_not_rotate(self):
        p = self.plan(fan_out=False)
        _, _, _, _, orders = p.passes[0].env_assignment(p.passes[0].sequences[0], 0, 64)
        self.assertEqual(set(orders), {"ABCD"})
        self.assertIn("fan_out is off", p.describe())

    def test_real_four_group_config_and_fingerprint(self):
        cfg, states, n, m = config_groups("configs/four_group_sequential_2x2.yaml")
        s = resolve_eval_settings(object(), cfg.eval, [])
        scenes = build_scenes(states, 64, env_n=n, env_m=m, group_specs=cfg.groups, fan_out=cfg.fan_out)
        plan = build_plan(scenes, s, num_envs=64, segment_len=80, seed=0)
        self.assertEqual([sc.background for sc in plan.scenes], ["bg0", "bg1", "bg2", "bg3"])
        d = plan.to_dict()
        self.assertEqual(d["rng"]["layouts"][0]["key"], "layout|seed=0|domain=in_domain|set=0")
        self.assertEqual(len(d["units"]), 2 * 6)
        self.assertEqual(d["rng"]["policy"][0]["key"], "policy|seed=0|domain=in_domain|round=0|pass=all_scenes")
        fp = plan_fingerprint(plan, dict(checkpoint="x"))
        shard = build_plan(scenes, resolve_eval_settings(object(), dict(cfg.eval, rounds="3-5"), []),
                           num_envs=64, segment_len=80, seed=0)
        self.assertEqual(plan_fingerprint(shard, dict(checkpoint="x"))["hash"], fp["hash"])
        for change in (dict(pose_sets=3, rounds="12-17"), dict(record_video=False), dict(video_envs_per_block=-1)):
            other_sel = build_plan(scenes, resolve_eval_settings(object(), dict(cfg.eval, **change), []),
                                   num_envs=64, segment_len=80, seed=0)
            self.assertEqual(plan_fingerprint(other_sel, dict(checkpoint="x"))["hash"], fp["hash"], change)
        for change in (dict(layout_slots=2), dict(record_pose=False), dict(scene_schedule="serial")):
            other_def = build_plan(scenes, resolve_eval_settings(object(), dict(cfg.eval, **change), []),
                                   num_envs=64, segment_len=80, seed=0)
            self.assertNotEqual(plan_fingerprint(other_def, dict(checkpoint="x"))["hash"], fp["hash"], change)
        other = build_plan(scenes, s, num_envs=64, segment_len=80, seed=1)
        self.assertNotEqual(plan_fingerprint(other, dict(checkpoint="x"))["hash"], fp["hash"])
        self.assertTrue(fingerprint_differences(fp["body"], plan_fingerprint(other, dict(checkpoint="x"))["body"]))
        print("\n" + plan.describe())


class TestRecordsAndOutputs(unittest.TestCase):
    def simulate(self, d, plan, rounds=None, rng_seed=0):
        """Write source rows for the plan's units (optionally a subset) as the evaluator would."""
        rng = np.random.default_rng(rng_seed)
        d = Path(d)
        for domain in plan.settings.domains:
            p = plan.passes[0]
            slots = p.layout_slots(64, -1)
            for seq in p.sequences:
                if rounds is not None and seq.seq_idx not in rounds:
                    continue
                chain = [1.0] * 64
                trials = []
                for t in range(seq.n_slots):
                    objs, receps, tasks, groups, orders = p.env_assignment(seq, t, 64)
                    for e in range(64):
                        s = float(rng.random() < 0.7)
                        chain[e] *= s
                        trials.append(dict(seq_idx=seq.seq_idx, task_idx=t, obj_set="rand", task=tasks[e], env_idx=e,
                                           success=s, success_chained=chain[e], grasp=1.0, obj_grasped=1.0,
                                           prefix="x", eval_kind=domain, seq_kind=seq.seq_kind, group=groups[e],
                                           obj=objs[e], recep=receps[e], order=orders[e], pose_set=seq.pose_set,
                                           cycle_idx=seq.cycle_idx, pass_label=p.label))
                _, _, _, groups, orders = p.env_assignment(seq, 0, 64)
                R.append_rows(d / R.LAYOUTS, R.LAYOUT_FIELDS,
                              [dict(eval_kind=domain, seq_idx=seq.seq_idx, seq_kind=seq.seq_kind, pose_set=seq.pose_set,
                                    pass_label=p.label, group=groups[e], env_idx=e, layout_slot=slots[e])
                               for e in range(64)])
                R.append_rows(d / R.PER_TRIAL, R.PER_TRIAL_FIELDS, (R.format_trial(r) for r in trials))

    def plan_dict(self, **kw):
        plan = TestPlan().plan(**kw)
        plan.fingerprint = plan_fingerprint(plan, {})
        return plan, plan.to_dict()

    def test_complete_run_writes_everything(self):
        plan, pd = self.plan_dict(pose_sets=2)
        with tempfile.TemporaryDirectory() as d:
            self.simulate(d, plan)
            res = O.rebuild_outputs(Path(d), [pd])
            self.assertTrue(res["status"]["complete"] and res["status"]["coverage_match"])
            rows = R.read_rows(Path(d) / R.SUCCESS)
            self.assertEqual({r["eval_kind"] for r in rows},
                             {f"{k}_{s}" for k in ("in_domain", "out_of_domain") for s in ("training", "random")})
            self.assertEqual(len(rows), 2 * 2 * 4 * 4)               # domains x kinds x scenes x tasks
            summary = res["summary"]
            orders = [r for r in summary if r["level"] == "order" and r["eval_kind"] == "in_domain"
                      and r["group"] == "group_A" and r["seq_kind"] == "random"]
            self.assertEqual(len(orders), 2 * 20)
            self.assertEqual(len([r for r in summary if r["level"] == "pose_set"]), 2 * 2 * 4 * 2)
            report = (Path(d) / "eval_report.txt").read_text()
            self.assertIn("status: COMPLETE", report)
            self.assertIn("in_domain · training rounds (rounds 0, 6; pose sets 0, 1)", report)
            self.assertIn("by pose set", report)
            self.assertTrue(res["wandb"])

    def test_incomplete_run_has_status_but_no_eval_success(self):
        plan, pd = self.plan_dict()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / R.SUCCESS).write_text("episode,total_steps\n")     # header-only leftover
            self.simulate(d, plan, rounds={0, 1})
            res = O.rebuild_outputs(Path(d), [pd])
            self.assertFalse(res["status"]["complete"])
            self.assertFalse((Path(d) / R.SUCCESS).exists())
            self.assertEqual(len(res["status"]["missing"]), 2 * 6 - 2 * 2)
            self.assertEqual(res["wandb"], {})
            self.assertIn("status: INCOMPLETE (4/12 units)", (Path(d) / "eval_report.txt").read_text())
            status = json.loads((Path(d) / R.STATUS).read_text())
            self.assertFalse(status["complete"])

    def test_partial_unit_detected_and_truncated(self):
        plan, pd = self.plan_dict()
        with tempfile.TemporaryDirectory() as d:
            self.simulate(d, plan, rounds={0, 1, 2})
            # crash in the middle of round 2 of out_of_domain: drop its last 100 per-trial rows
            path = Path(d) / R.PER_TRIAL
            lines = path.read_text().splitlines(keepends=True)
            path.write_text("".join(lines[:-100]))
            units = O.planned_units([pd])
            done, partial, overfull = O.completed_units(R.read_rows(path), units)
            self.assertEqual(partial, {("out_of_domain", "all_scenes", 2)})
            dropped = O.truncate_to_units(Path(d), done)
            self.assertEqual(dropped[R.PER_TRIAL], 256 - 100)
            self.assertEqual(dropped[R.LAYOUTS], 64)
            done2, partial2, _ = O.completed_units(R.read_rows(path), units)
            self.assertEqual((len(done2), partial2), (5, set()))

    def test_training_directory_is_refused(self):
        plan, pd = self.plan_dict()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "rollout_success.csv").write_text("x\n")
            with self.assertRaisesRegex(ValueError, "training run directory"):
                O.rebuild_outputs(Path(d), [pd])

    def test_merge_shards(self):
        import tools.rebuild_eval_outputs as tool
        plan_a, pa = self.plan_dict(rounds="0-2")
        plan_b, pb = self.plan_dict(rounds="3-")
        with tempfile.TemporaryDirectory() as root:
            a, b, out = Path(root) / "a", Path(root) / "b", Path(root) / "out"
            for g, plan, pd in ((a, plan_a, pa), (b, plan_b, pb)):
                g.mkdir()
                (g / "eval_plan.json").write_text(json.dumps(pd))
                self.simulate(g, plan)
                st = O.rebuild_outputs(g, [pd])["status"]
                self.assertEqual((st["complete"], st["full_design"]), (True, False))  # shard: complete selection
                self.assertFalse((g / R.SUCCESS).exists())                              # ...but no eval_success
            res = tool.merge(out, [a, b], "eval_report.txt")
            self.assertTrue(res["status"]["complete"] and res["status"]["coverage_match"] and res["status"]["full_design"])
            self.assertTrue((out / R.SUCCESS).exists())
            self.assertEqual(len(R.read_rows(out / R.PER_TRIAL)), 2 * 6 * 4 * 64)
            # overlapping shards are refused
            with self.assertRaisesRegex(SystemExit, "complete in both"):
                tool.merge(Path(root) / "out2", [a, a], "eval_report.txt")
            # adding a pose set later: a pose_sets=2 shard of rounds 6-11 extends the design
            plan_d, pd2 = self.plan_dict(pose_sets=2, rounds="6-11")
            e = Path(root) / "e"
            e.mkdir()
            (e / "eval_plan.json").write_text(json.dumps(pd2))
            self.simulate(e, plan_d)
            res2 = tool.merge(Path(root) / "out_sets", [out, e], "eval_report.txt")
            st2 = res2["status"]
            self.assertEqual((st2["complete"], st2["full_design"], st2["planned_units"]), (True, True, 24))
            self.assertIn("pose sets 0, 1", res2["report"])
            # without the pose-set-0 rounds the pose_sets=2 design is not covered
            st3 = tool.merge(Path(root) / "out_sets_only1", [e], "eval_report.txt")["status"]
            self.assertEqual((st3["complete"], st3["full_design"]), (True, False))
            # a different eval is refused
            plan_c, pc = self.plan_dict(rounds="3-", seed=1)
            c = Path(root) / "c"
            c.mkdir()
            (c / "eval_plan.json").write_text(json.dumps(pc))
            with self.assertRaisesRegex(ValueError, "fingerprints differ"):
                tool.merge(Path(root) / "out3", [a, c], "eval_report.txt")

    def test_mcnemar_pairs_every_round(self):
        # mcnemar_pair.py is an offline analysis tool, not shipped in the release.
        try:
            import analysis.mcnemar_pair as mc
        except ImportError:
            self.skipTest("analysis/mcnemar_pair.py not present")
        plan, pd = self.plan_dict(domains=["in_domain"])
        with tempfile.TemporaryDirectory() as d:
            self.simulate(d, plan)
            path = Path(d) / R.PER_TRIAL
            rows = R.read_rows(path)
            for r in rows:
                r["prefix"] = f"in_domain_seq{r['seq_idx']}_task{r['task_idx']}"
            R.atomic_write_csv(path, R.PER_TRIAL_FIELDS, rows)
            self.assertEqual(len(mc.read_per_trial(path)), 6 * 4 * 64)
            self.assertEqual(len(mc.read_per_trial(path, {"training"})), 4 * 64)


class TestProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "configs").mkdir()
        self.train_cfg = self.root / "configs" / "train.yaml"
        self.train_cfg.write_text((ROOT / "configs/two_group_sequential_2x2.yaml").read_text())
        self.ckpt = self.root / "run" / "glob" / "episode_0004"
        self.ckpt.mkdir(parents=True)
        (self.ckpt / "run_config.yaml").write_text("config_path: configs/train.yaml\nepisode_len: 320\nnum_envs: 32\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_to_recorded_training_config(self):
        from evaluation.provenance import resolve_config_path, check_against_training
        path, prov = resolve_config_path("", str(self.ckpt), self.root)
        self.assertEqual(Path(path), self.train_cfg.resolve())
        self.assertEqual(prov["config_source"], "training run_config config_path")
        check_against_training(load_cronos_config(path), prov, load_cronos_config, False)
        self.assertTrue(prov["check"].startswith("match"))
        self.assertIn("no snapshot", prov["check"])

    def test_snapshot_wins_and_detects_edits(self):
        from evaluation.provenance import resolve_config_path, check_against_training
        (self.ckpt / "experiment_config.yaml").write_text(self.train_cfg.read_text())
        path, prov = resolve_config_path("", str(self.ckpt), self.root)
        self.assertEqual(prov["config_source"], "checkpoint snapshot")
        # training file edited after training: evaluating it explicitly must fail
        self.train_cfg.write_text(self.train_cfg.read_text().replace("background: 1", "background: 2"))
        path, prov = resolve_config_path(str(self.train_cfg), str(self.ckpt), self.root)
        with self.assertRaisesRegex(ValueError, "background"):
            check_against_training(load_cronos_config(path), prov, load_cronos_config, False)
        prov = check_against_training(load_cronos_config(path), prov, load_cronos_config, True)
        self.assertTrue(prov["check"].startswith("MISMATCH"))

    def test_legacy_reduced_env_eval_config_is_rejected(self):
        from evaluation.provenance import resolve_config_path, check_against_training
        legacy = ROOT / "configs/eval/two_group_2x2.yaml"
        path, prov = resolve_config_path(str(legacy), str(self.ckpt), self.root)
        with self.assertRaisesRegex(ValueError, "num_envs"):
            check_against_training(load_cronos_config(path), prov, load_cronos_config, False)

    def test_eval_block_is_not_part_of_the_comparison(self):
        from evaluation.provenance import resolve_config_path, check_against_training
        other = self.root / "configs" / "other.yaml"
        other.write_text(self.train_cfg.read_text().replace("pose_sets: 1", "pose_sets: 2"))
        path, prov = resolve_config_path(str(other), str(self.ckpt), self.root)
        check_against_training(load_cronos_config(path), prov, load_cronos_config, False)
        self.assertTrue(prov["check"].startswith("match"))

    def test_no_config_anywhere(self):
        from evaluation.provenance import resolve_config_path
        with self.assertRaisesRegex(ValueError, "--config-path"):
            resolve_config_path("", str(self.root / "nowhere"), self.root)


class TestPolicyArgs(unittest.TestCase):
    """evaluation.provenance.resolve_policy_args: which VLA settings eval uses."""

    def args(self):
        # eval_only.EvalArgs defaults
        return SimpleNamespace(policy="openvla", vla_path="openvla/openvla-7b", vla_unnorm_key="bridge_orig",
                               vla_temperature_eval=0.6, vla_lora_rank=32)

    def ckpt(self, d, run_config):
        c = Path(d) / "glob" / "episode_0008"
        c.mkdir(parents=True)
        if run_config is not None:
            import yaml
            (c / "run_config.yaml").write_text(yaml.safe_dump(run_config))
        return str(c)

    SPATIAL_RC = dict(policy="spatialvla", vla_path="IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge",
                      vla_unnorm_key="bridge_orig/1.0.0", vla_temperature_eval=0.0, vla_lora_rank=32)

    def test_spatialvla_checkpoint_needs_no_flags(self):
        from evaluation.provenance import resolve_policy_args
        with tempfile.TemporaryDirectory() as d:
            a = self.args()
            rec = resolve_policy_args(a, [], self.ckpt(d, self.SPATIAL_RC))
            self.assertEqual((a.policy, a.vla_unnorm_key, a.vla_temperature_eval),
                             ("spatialvla", "bridge_orig/1.0.0", 0.0))
            self.assertEqual(set(rec["sources"].values()), {"checkpoint run_config"})
            self.assertEqual(rec["warnings"], [])

    def test_cli_wins_and_is_flagged(self):
        from evaluation.provenance import resolve_policy_args
        with tempfile.TemporaryDirectory() as d:
            a = self.args()
            a.vla_temperature_eval = 0.6
            rec = resolve_policy_args(a, ["--vla-temperature-eval", "0.6"], self.ckpt(d, self.SPATIAL_RC))
            self.assertEqual((a.policy, a.vla_temperature_eval), ("spatialvla", 0.6))
            self.assertEqual(rec["sources"]["vla_temperature_eval"], "cli")
            self.assertTrue(any("differs from training" in w for w in rec["warnings"]))

    def test_other_policy_ignores_checkpoint_settings(self):
        from evaluation.provenance import resolve_policy_args
        with tempfile.TemporaryDirectory() as d:
            a = self.args()
            a.policy = "spatialvla"
            rc = dict(policy="openvla", vla_path="openvla/openvla-7b", vla_unnorm_key="bridge_orig",
                      vla_temperature_eval=0.6)
            rec = resolve_policy_args(a, ["--policy", "spatialvla"], self.ckpt(d, rc))
            self.assertEqual((a.vla_unnorm_key, a.vla_temperature_eval), ("bridge_orig/1.0.0", 0.0))
            self.assertTrue(any("trained with policy='openvla'" in w for w in rec["warnings"]))

    def test_old_run_config_without_policy_is_openvla(self):
        from evaluation.provenance import resolve_policy_args
        with tempfile.TemporaryDirectory() as d:
            a = self.args()
            resolve_policy_args(a, [], self.ckpt(d, dict(vla_path="/my/openvla", vla_temperature_eval=1.0)))
            self.assertEqual((a.policy, a.vla_path, a.vla_temperature_eval), ("openvla", "/my/openvla", 1.0))

    def test_yaml_then_defaults_without_run_config(self):
        from evaluation.provenance import resolve_policy_args
        with tempfile.TemporaryDirectory() as d:
            a = self.args()
            cfg = load_cronos_config(ROOT / "configs/spatialvla_2x2_train.yaml")
            rec = resolve_policy_args(a, [], self.ckpt(d, None), cfg)
            self.assertEqual((a.policy, a.vla_unnorm_key, a.vla_temperature_eval),
                             ("spatialvla", "bridge_orig/1.0.0", 0.0))
            self.assertEqual(rec["sources"]["policy"], "config yaml")
            self.assertEqual(rec["sources"]["vla_temperature_eval"], "default")
            self.assertTrue(any("no run_config" in w for w in rec["warnings"]))
            b = self.args()
            resolve_policy_args(b, [], "")
            self.assertEqual((b.policy, b.vla_unnorm_key, b.vla_temperature_eval), ("openvla", "bridge_orig", 0.6))

    def test_unknown_policy(self):
        from evaluation.provenance import resolve_policy_args
        a = self.args()
        a.policy = "rt2"
        with self.assertRaisesRegex(ValueError, "unknown policy"):
            resolve_policy_args(a, ["--policy", "rt2"], "")


class FakeT:
    """Minimal tensor stand-in with .detach().cpu().numpy()."""
    def __init__(self, a): self.a = a
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


def old_inline_pose_writer(csv_path, poses, instr, num_envs, episode, segment_id, total_steps, phase):
    """Verbatim copy of the pre-refactor main.py::_record_segment_pose body."""
    write_hdr = not csv_path.exists()
    ep_1, seg_1 = episode, segment_id + 1

    def _np(pair):
        p, q = pair
        return p.detach().cpu().numpy(), q.detach().cpu().numpy()

    rows = []
    for kind, entries, name_lists in (
        ("obj", poses["obj"], poses["obj_names"]),
        ("recep", poses["recep"], poses["recep_names"]),
    ):
        for slot, (pair, names) in enumerate(zip(entries, name_lists)):
            p, q = _np(pair)
            rows.append((kind, slot, names, p, q))
    g_p, g_q = _np(poses["gripper"])
    rows.append(("gripper", 0, [""] * num_envs, g_p, g_q))

    with open(csv_path, "a") as f:
        if write_hdr:
            f.write("episode,segment,phase,total_steps,env,actor_kind,slot,"
                    "model_name,task,px,py,pz,qw,qx,qy,qz\n")
        for i in range(num_envs):
            task_str = str(instr[i]).replace('"', '""') if i < len(instr) else ""
            for kind, slot, names, p, q in rows:
                model = str(names[i]).replace('"', '""') if i < len(names) else ""
                f.write(f"{ep_1},{seg_1},{phase},{total_steps},{i},{kind},{slot},"
                        f"\"{model}\",\"{task_str}\","
                        f"{p[i,0]:.6f},{p[i,1]:.6f},{p[i,2]:.6f},"
                        f"{q[i,0]:.6f},{q[i,1]:.6f},{q[i,2]:.6f},{q[i,3]:.6f}\n")


class TestSegmentPoseWriter(unittest.TestCase):
    def poses(self, b=6):
        r = np.random.default_rng(1)
        pq = lambda: (FakeT(r.normal(size=(b, 3))), FakeT(r.normal(size=(b, 4))))
        nan_slot = (FakeT(np.full((b, 3), np.nan)), FakeT(np.full((b, 4), np.nan)))
        return {"obj": [pq(), nan_slot], "recep": [pq(), pq()], "gripper": pq(),
                "obj_names": [[f"o{i}" for i in range(b)], [""] * b],
                "recep_names": [['pl"ate'] * b, ["cloth"] * b]}

    def test_training_output_byte_identical(self):
        poses, instr = self.poses(), [f'put a, "b" on c{i}' for i in range(6)]
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "old.csv", Path(d) / "new.csv"
            for seg, phase in [(0, "start"), (0, "end"), (1, "start")]:
                old_inline_pose_writer(a, poses, instr, 6, 3, seg, 1234, phase)
                SegmentPoseWriter(b).write(poses, instr, 6, episode=3, segment=seg + 1, phase=phase,
                                           total_steps=1234)
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_eval_extra_columns(self):
        poses, instr = self.poses(), ["t"] * 6
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "eval.csv"
            w = SegmentPoseWriter(p, ("eval_kind", "group"))
            w.write(poses, instr, 6, episode=1, segment=2, phase="end", total_steps=0,
                    extra_values=[["in_domain"] * 6, [f"g{i}" for i in range(6)]])
            rows = list(csv.DictReader(io.StringIO(p.read_text())))
            self.assertEqual(len(rows), 6 * 5)
            self.assertEqual(rows[-1]["group"], "g5")
            self.assertEqual(rows[0]["px"] != "", True)
            self.assertEqual([r for r in rows if r["model_name"] == "" and r["actor_kind"] == "obj"][0]["px"], "nan")


if __name__ == "__main__":
    unittest.main()
