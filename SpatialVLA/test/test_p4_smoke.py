"""P-4 gate driver — smoke + observability + on-minibatch G-gate run.

End-to-end mini exercise of the production stack without ManiSkill:

  1. Build the real `SpatialVLAPolicy` (LoRA r32 active, value head init-zero,
     two AdamW). PEFT init-B=0 → forward unchanged from raw at init.
  2. Synthetic mini-rollout (8 real env steps; synthetic obs but real
     `policy.get_action` / `buffer.insert` calls), capturing per-step
     `(action_ids, lp_rollout, value_rollout)` for `gate_buffer_integrity`.
  3. `gate_buffer_integrity` (NF-3 + NF-9) — buffer round-trip byte-equal
     before any G2/G3 runs, so a failure localizes to the buffer plumbing
     rather than the model.
  4. `gate_geometry`, `gate_G2`, `gate_G2d_no_truncation`,
     `gate_G2d_warper_absent`, `gate_G3_anchor`, `gate_G3` on the same θ.
     Plan §P-4 calls for "epoch-0 first minibatch via production
     `policy.evaluate_actions`" — the gates here invoke `evaluate_actions`
     directly with the rollout obs, which is bit-identical to what
     `train_epoch`'s first minibatch would see (BUFFER stores the same
     `(obs, action_ids)`, FEED_FORWARD_GENERATOR yields them).
  5. One real `CronosPPO.train_epoch(buffer)` over the tiny buffer:
     - completes without NaN
     - returns a `list` of per-minibatch dicts (NF-10)
     - each dict carries the additive P-4 keys (ratio mean/median/max,
       approx_kl, clip_fraction, value_explained_variance,
       grad_norm_vla/grad_norm_vh, g2_logp_gap)
     - first minibatch's g2_logp_gap is within bf16 noise of 0 (NF-3)
  6. `aggregate_train_results` summarizes the minibatch list as the
     `wandb.log` site will (main.py:1028 NF-10 fix).

Skipped at P-4 (exercised at P-5/P-6): the rollout per-task observability
keys (`rollout/<task>/*`) come from a REAL env step, which the gate cannot
synthesize. The PLUMBING is verified by source assertions.

Usage (from `Autonomous_RL/SpatialVLA/`):
    conda activate spatialvla_cronos
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:../SimplerEnv:../CRONOS \\
        python -m test.test_p4_smoke \\
        --model-path IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge
"""
import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass
class _SmokeArgs:
    """Minimal namespace satisfying SpatialVLAPolicy + CronosReplayBuffer + CronosPPO."""
    # SpatialVLAPolicy fields
    vla_path: str = ""
    vla_load_path: str = ""
    vla_unnorm_key: str = "bridge_orig/1.0.0"
    seed: int = 0
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.0
    vla_lora_rank: int = 32
    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    # CronosReplayBuffer fields
    segment_len: int = 8
    num_envs: int = 1
    episode_len: int = 8
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95
    buffer_minibatch: int = 4
    # CronosPPO fields
    alg_entropy_coef: float = 0.0
    alg_gradient_accum: int = 1
    vla_grad_norm: float = 10.0


def _make_obs(device, image_path: Path, instruction: str, num_envs: int):
    """Build a CRONOS-format obs dict (uint8 [B, H, W, 3] on device + str list).

    Image shape matches CronosWrapper's output (480x640 uint8), which is the
    shape `CronosReplayBuffer` allocates for its obs slot. SpatialVLA's
    processor handles the resize-to-224 internally, so we don't need to
    pre-resize here. Bytes are deterministic-seeded (the gates only care
    that two runs of `get_action` on the SAME obs agree, which holds
    regardless of pixel content)."""
    pil = Image.open(image_path).convert("RGB").resize((640, 480))
    arr = np.asarray(pil, dtype=np.uint8)
    images = np.stack([arr for _ in range(num_envs)], axis=0)
    image_t = torch.from_numpy(images).to(device=device)
    return {"image": image_t, "task_description": [instruction] * num_envs}


def _next_obs_synthetic(device, num_envs: int, H=224, W=224):
    """A fresh random uint8 image so the buffer's `obs[step+1]` slot has plausible bytes.

    The buffer's image slot is `(ep_len+1, max_envs, *obs_dim)`; obs_dim is
    `(480, 640, 3)` by default. Match that shape so `buffer.insert` doesn't
    silently truncate. Image content is irrelevant — the smoke test only
    exercises the round-trip of `actions / action_log_probs / value_preds`.
    """
    return torch.randint(0, 256, (num_envs, 480, 640, 3), dtype=torch.uint8, device=device)


def _run_gate(name, fn):
    try:
        out = fn()
        msg = f"  PASS  {name}"
        if isinstance(out, float):
            msg += f"  · returned {out:.3e}"
        print(msg)
        return True
    except Exception as e:
        print(f"  FAIL  {name}  · {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser("P-4 PPO smoke gate")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", default="test/example.png")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Make adapter + CRONOS importable.
    here = Path(__file__).resolve().parent.parent          # Autonomous_RL/SpatialVLA
    for p in (str(here.parent / "SimplerEnv"),
              str(here.parent / "CRONOS"),
              str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from simpler_env.policies.spatialvla.spatialvla_train import SpatialVLAPolicy
    from training.buffer import CronosReplayBuffer
    from training.ppo import CronosPPO, aggregate_train_results
    from test.gates_ppo import (
        gate_geometry, gate_G2, gate_G2d_no_truncation, gate_G2d_warper_absent,
        gate_G3, gate_G3_anchor, gate_buffer_integrity,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    smoke_args = _SmokeArgs(
        vla_path=args.model_path,
        seed=args.seed,
        num_envs=args.num_envs,
        # Buffer geometry: ep_len = num_envs = num_envs (one segment per env).
        segment_len=8,
        episode_len=8,
        buffer_minibatch=4,
    )

    print(f"[load] SpatialVLAPolicy from {args.model_path}")
    policy = SpatialVLAPolicy(smoke_args, device_id=0)
    device = policy.tpdv["device"]

    # NF-2: buffer width comes off the policy. `act_dim=3` for SpatialVLA.
    buffer = CronosReplayBuffer(smoke_args, act_dim=policy.act_token_len)
    assert buffer.actions.shape[-1] == 3, "P-4: SpatialVLA buffer width should be 3 tokens (NF-2)"

    # Build a synthetic obs (image + instruction) and run a real mini-rollout.
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = here / image_path
    instruction = "pick the cup"
    obs = _make_obs(device, image_path, instruction, smoke_args.num_envs)

    policy.prep_rollout()
    buffer.warmup(obs["image"], obs["task_description"])
    capture = []                                              # NF-3/NF-9 round-trip evidence
    for step in range(smoke_args.segment_len):
        with torch.no_grad():
            val, action_ids, lp = policy.get_action(obs, deterministic=False)
        capture.append({
            "action_ids": action_ids.clone(),
            "lp":         lp.clone(),
            "value":      val.clone(),
        })
        # Synthetic env step — reward/mask kept simple so GAE math is well-defined.
        # The buffer pairs `obs[step]` with `action[step]`; we MUST advance
        # `obs` for the next iteration so each (obs, action) pair in the
        # buffer corresponds to a genuine `get_action(obs)` invocation. Without
        # this advance, the buffer's obs[t] for t≥1 (a random synthetic image)
        # would be paired with action[t] computed on obs[0] (the real image),
        # and `train_epoch`'s re-evaluation would produce a logπ for the wrong
        # context → `g2_logp_gap[0]` jumps to O(10+) and surfaces as a false NF-3.
        next_obs_img = _next_obs_synthetic(device, smoke_args.num_envs)
        reward = torch.zeros((smoke_args.num_envs, 1), device=device)
        truncated = torch.zeros((smoke_args.num_envs, 1), device=device)
        buffer.insert(next_obs_img, action_ids, lp, val, reward, 1.0 - truncated)
        obs = {"image": next_obs_img, "task_description": obs["task_description"]}

    # Close the segment with a final next_value computed on the now-current obs.
    with torch.no_grad():
        next_val, _, _ = policy.get_action(obs, deterministic=False)
    buffer.end_segment(next_val)

    results = []

    # NF-3 + NF-9: buffer round-trip BEFORE G2/G3, so failures localize.
    results.append(_run_gate("gate_buffer_integrity (NF-3+NF-9)",
                             lambda: gate_buffer_integrity(buffer, capture)))

    # NF-7 geometry holds through the production adapter.
    results.append(_run_gate("gate_geometry (NF-7)",            lambda: gate_geometry(policy)))

    # G2 family on the same obs the rollout used (bit-identical to what
    # train_epoch's first minibatch would see; only the random minibatch
    # permutation re-orders environments within the rollout, not their values).
    results.append(_run_gate("gate_G2 (G2a/G2b/G2c/m3/greedy)", lambda: gate_G2(policy, obs)))
    results.append(_run_gate("gate_G2d_no_truncation",          lambda: gate_G2d_no_truncation(policy, obs)))
    results.append(_run_gate("gate_G2d_warper_absent (NF-1)",   lambda: gate_G2d_warper_absent(policy, obs)))

    # G3 family. G3a (ground-truth anchor) then G3 (rollout↔evaluate value
    # consistency at the prefix-LM boundary).
    results.append(_run_gate("gate_G3_anchor G3a (NF-8)",       lambda: gate_G3_anchor(policy, obs)))
    results.append(_run_gate("gate_G3 (G3 + G3b)",              lambda: gate_G3(policy, obs)))

    # PPO smoke: one train_epoch over the just-populated buffer.
    def _smoke_train_epoch():
        buffer.compute_gae()
        ppo = CronosPPO(smoke_args, policy)
        policy.prep_training()
        train_results = ppo.train_epoch(buffer)
        policy.prep_rollout()

        assert isinstance(train_results, list) and train_results, \
            f"train_epoch must return a non-empty list (NF-10); got {type(train_results).__name__}"
        # NF-10: each minibatch dict carries the additive P-4 keys.
        required_keys = {
            "policy_loss", "value_loss", "entropy", "total_loss",
            "ratio_mean", "ratio_median", "ratio_max",
            "approx_kl", "clip_fraction", "value_explained_variance",
            "grad_norm_vla", "grad_norm_vh", "g2_logp_gap",
        }
        missing = required_keys - set(train_results[0].keys())
        assert not missing, f"P-4 NF-10: first minibatch missing keys {sorted(missing)}"
        # No NaN in any aggregate / per-minibatch scalar.
        for r in train_results:
            for k, v in r.items():
                if v is not None:
                    assert not (isinstance(v, float) and (v != v)), f"NaN in train_results[{k}]"
        # g2_logp_gap on the FIRST minibatch must be within bf16 noise (NF-3 /
        # m1: rollout θ is unchanged before the first PPO step).
        g2_gap = train_results[0]["g2_logp_gap"]
        assert g2_gap < 1.0, \
            f"NF-3 FAIL: g2_logp_gap on first minibatch = {g2_gap:.3e} — rollout θ should still match"
        # Aggregator: same scalar at the wandb.log site.
        agg = aggregate_train_results(train_results)
        assert "g2_logp_gap" in agg and abs(agg["g2_logp_gap"] - g2_gap) < 1e-9, \
            f"aggregator must read g2_logp_gap from train_results[0] (got {agg.get('g2_logp_gap')!r} vs {g2_gap!r})"

    results.append(_run_gate("smoke: train_epoch + NF-10 additive keys", _smoke_train_epoch))

    # main.py source-level assertion for the rollout per-task (m2) plumbing
    # — the run_rollout capture point is BEFORE the scheduler advance.
    def _main_source_observability():
        main_src = Path(__file__).resolve().parent.parent.parent / "CRONOS" / "main.py"
        s = main_src.read_text()
        assert "self._rollout_per_task" in s, "main.py: rollout per-task accumulator missing"
        assert "aggregate_train_results" in s, "main.py: wandb.log site must use aggregator (NF-10)"
        # Capture point must come BEFORE scheduler.update_index() in source order.
        cap_pos = s.find("self._rollout_per_task.setdefault")
        sched_pos = s.find("self.scheduler.update_index()")
        assert 0 < cap_pos < sched_pos, \
            "main.py: per-task capture must precede scheduler.update_index() (m2 — else curves mislabel)"
        # 1028-site uses aggregator instead of train_results[-1].
        assert "**train_results[-1]" not in s, "main.py: wandb.log still references train_results[-1] (NF-10)"

    results.append(_run_gate("main.py observability source (m2 + NF-10)", _main_source_observability))

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n[summary] {n_pass}/{n_total} gates passed")

    buffer.cleanup()
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
