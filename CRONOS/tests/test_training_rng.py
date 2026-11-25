"""Training RNG split: scene draws vs task schedule (CPU only).

Run: python -m pytest tests/test_training_rng.py -q
"""
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import rng_streams as S                      # noqa: E402
from envs.scheduler import GroupState, TaskScheduler    # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

POOL = ["put a on x", "put a on y", "put b on x", "put b on y"]


def make_scheduler(mode):
    gs = GroupState.from_sequence("g", POOL, eval_tasks=POOL, num_envs=8)
    return TaskScheduler([gs], mode=mode, num_envs=8, fan_out=False)


def draw(sched, n):
    out = []
    for _ in range(n):
        objs, receps = sched.get_next_tasks()
        out.append((objs[0], receps[0]))
        sched.update_index()
    return out


class TestSceneStreams(unittest.TestCase):
    def test_depends_only_on_key(self):
        a = S.scene_ids(0, 16, "episode", episode=3)
        random.random()
        if np is not None:
            np.random.rand(100)
        self.assertEqual(a, S.scene_ids(0, 16, "episode", episode=3))

    def test_prefix_stable(self):
        self.assertEqual(S.scene_ids(0, 8, "respawn", episode=1, segment=2),
                         S.scene_ids(0, 16, "respawn", episode=1, segment=2)[:8])

    def test_keys_separate(self):
        base = S.scene_ids(0, 8, "episode", episode=1)
        self.assertNotEqual(base, S.scene_ids(0, 8, "episode", episode=2))
        self.assertNotEqual(base, S.scene_ids(1, 8, "episode", episode=1))
        self.assertNotEqual(base, S.scene_ids(0, 8, "respawn", episode=1))
        self.assertNotEqual(S.scene_ids(0, 8, "respawn", episode=1, segment=1),
                            S.scene_ids(0, 8, "respawn", episode=1, segment=2))

    def test_ids_fit_int64_and_cover_table(self):
        ids = S.scene_ids(0, 4096, "episode", episode=0)
        self.assertTrue(all(0 <= i < 2 ** 62 for i in ids))
        ltt = 432                     # 2x2 preset: xyz configs; any table size works
        self.assertEqual(len({i % ltt for i in ids}), ltt)

    def test_task_stream_differs_from_scene(self):
        self.assertNotEqual(S.task_rng(0).random(), S.stream(S.scene_key(0, "episode")).random())


class TestSchedulerRng(unittest.TestCase):
    def test_global_random_does_not_move_tasks(self):
        for mode in ("pure_random", "sequence_random"):
            a = make_scheduler(mode); a.set_rng(S.task_rng(0))
            b = make_scheduler(mode); b.set_rng(S.task_rng(0))
            ra = []
            for _ in range(20):
                random.random()           # a scene / other consumer on the global stream
                ra += draw(a, 1)
            self.assertEqual(ra, draw(b, 20), mode)

    def test_task_draws_are_random_and_seeded(self):
        for mode in ("pure_random", "sequence_random"):
            a = make_scheduler(mode); a.set_rng(S.task_rng(0))
            b = make_scheduler(mode); b.set_rng(S.task_rng(1))
            da, db = draw(a, 40), draw(b, 40)
            self.assertGreater(len(set(da)), 1, mode)
            self.assertNotEqual(da, db, mode)

    def test_state_roundtrip_resumes_exactly(self):
        for mode in ("pure_random", "sequence_random"):
            full = make_scheduler(mode); full.set_rng(S.task_rng(7))
            expect = draw(full, 30)
            first = make_scheduler(mode); first.set_rng(S.task_rng(7))
            head = draw(first, 13)
            state = json.loads(json.dumps(first.get_state()))
            resumed = make_scheduler(mode); resumed.set_rng(S.task_rng(7))
            resumed.load_state(state)
            self.assertEqual(head + draw(resumed, 17), expect, mode)

    def test_legacy_default_uses_global_random(self):
        a = make_scheduler("pure_random")
        self.assertIs(a.rng, random)
        self.assertIsNone(a.get_state()["rng_state"])
        random.seed(3); x = draw(a, 10)
        b = make_scheduler("pure_random")
        random.seed(3); self.assertEqual(x, draw(b, 10))


class TestOomReport(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_is_oom(self):
        import torch
        from oom_report import is_oom
        self.assertTrue(is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))
        self.assertFalse(is_oom(RuntimeError("shape mismatch")))
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None:
            self.assertTrue(is_oom(oom_cls("x")))

    def test_report_written_without_gpu(self):
        import tempfile
        from oom_report import write_oom_report
        with tempfile.TemporaryDirectory() as d:
            path = write_oom_report(RuntimeError("CUDA out of memory"), Path(d),
                                    {"phase": "rollout", "episode": 3, "segment": 2})
            text = Path(path).read_text()
            self.assertIn("phase=rollout", text)
            self.assertIn("episode=3", text)


if __name__ == "__main__":
    unittest.main()
