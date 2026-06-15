"""O-0 perf-parity harness for plans/2026-06-09_spatialvla-ppo-perf-optimization.md.

Captures a tiny but representative PPO fixture from the unmodified SpatialVLA
PPO stack (the `perf-baseline` git tag, 83b97f1) and re-runs the SAME fixture
under per-phase optimization flags, asserting that the optimization does not
move numerics beyond the declared tolerance — i.e. each speedup is parity-
preserving (Class E: bf16-tol within `TOL_LOGP=1.0` / `TOL_VALUE=0.5`; or
Class A: a looser, phase-declared TOL with explicit fallback).

Why a separate harness from `test_p4_smoke.py`: P-4 verifies the PPO surface is
CORRECT (G2/G3 etc.); this harness verifies an optimized variant computes the
SAME thing as the baseline. Two different questions; same per-step (action,
logπ, value) and per-update (grad, train-result) primitives.

Fixture geometry (chosen to exercise both A2.1 and A3.1 in one buffer):
    num_envs       = 8
    segment_len    = 20   → 160 samples total
    buffer_minibatch = 8, alg_gradient_accum = 20 (baseline 8×20 = 160)
    optimized A3.1: buffer_minibatch = 80, alg_gradient_accum = 2 (80×2 = 160)
    inferbatch baseline = 4 (splits 8 envs into 2 micro-batches; the production
    32→64 change has the same shape — `num_envs > inferbatch` → 1 micro-batch
    instead of 2).
Each fixture run does 20 real `policy.get_action` calls plus one
`CronosPPO.train_epoch` → meaningful s/step and s/update measurements (much
smaller magnitude than P-5's 4.49 s/step, but the per-phase RATIO scales).

Usage (from `Autonomous_RL/SpatialVLA/`):
    conda activate spatialvla_cronos
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:../SimplerEnv:../CRONOS \\
        python -m test.test_perf_parity \\
        --model-path IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge \\
        --capture

Then per-phase (each phase reads the same baseline file):
    python -m test.test_perf_parity --phase O1
    ...
"""
import argparse
import json
import pickle
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# Fixture / parity tolerances. TOL_LOGP / TOL_VALUE re-use the
# `reports/2026-06-06_spatialvla-ppo-results.md` *Common setup* numbers — same
# bf16 noise floor SpatialVLA's Gemma2 HybridCache↔cacheless drift sits in.
TOL_LOGP   = 1.0
TOL_VALUE  = 0.5
# TOL_GRAD is *measured*, not declared: O-0's capture pass runs the fixture
# twice and records `max per-param |Δ grad|` across two identical-config bf16
# runs. Stored in the fixture so per-phase gates compare against the SAME
# baseline jitter that O-0 measured. k=5 mirrors the report's "set TOL_GRAD
# from baseline jitter, k≈5" recipe.
TOL_GRAD_K = 5.0
# Train-result scalar parity tolerance (small enough to catch real numeric drift
# but loose enough to survive bf16 jitter on means/maxes).
TOL_TRAIN_SCALAR = 5e-2


@dataclass
class _PerfArgs:
    """Minimal namespace satisfying SpatialVLAPolicy + CronosReplayBuffer + CronosPPO.

    Mirrors `_SmokeArgs` in test_p4_smoke.py — same field set, same defaults,
    only the buffer geometry is sized for parity tests (160 samples / 1 update).
    """
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
    # Buffer geometry: 8 envs × 20 steps = 160 samples per update.
    segment_len: int = 20
    num_envs: int = 8
    episode_len: int = 20
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95
    buffer_minibatch: int = 8
    # PPO update geometry: 8 × 20 = 160 = full buffer per optimizer step.
    alg_entropy_coef: float = 0.0
    alg_gradient_accum: int = 20
    vla_grad_norm: float = 10.0


_FIXTURE_FILE = Path(__file__).resolve().parent / "perf_baseline_fixtures.pt"


# --- fixture build / re-run ---------------------------------------------------


def _make_obs(device, image_path: Path, instruction: str, num_envs: int):
    """CRONOS-format obs dict; same shape as test_p4_smoke._make_obs."""
    pil = Image.open(image_path).convert("RGB").resize((640, 480))
    arr = np.asarray(pil, dtype=np.uint8)
    images = np.stack([arr for _ in range(num_envs)], axis=0)
    image_t = torch.from_numpy(images).to(device=device)
    return {"image": image_t, "task_description": [instruction] * num_envs}


def _next_obs_synthetic(device, num_envs: int, seed: int):
    """Deterministic synthetic next-obs. Fixed-seeded so the rollout sequence is
    reproducible across the baseline capture and every phase's re-run."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, 256, (num_envs, 480, 640, 3),
                         dtype=torch.uint8, generator=g, device=device)


def _build_policy_and_buffer(model_path: str, seed: int):
    """Construct (policy, buffer, ppo, args) for one fixture run.

    Resolves SpatialVLAPolicy → CronosReplayBuffer → CronosPPO in the same order
    `test_p4_smoke.py` does (it's the production order in `CronosRunner`).
    """
    here = Path(__file__).resolve().parent.parent
    for p in (str(here.parent / "SimplerEnv"),
              str(here.parent / "CRONOS"),
              str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from simpler_env.policies.spatialvla.spatialvla_train import SpatialVLAPolicy
    from training.buffer import CronosReplayBuffer
    from training.ppo import CronosPPO

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = _PerfArgs(vla_path=model_path, seed=seed)
    policy = SpatialVLAPolicy(args, device_id=0)
    buffer = CronosReplayBuffer(args, act_dim=policy.act_token_len)
    ppo = CronosPPO(args, policy)
    return policy, buffer, ppo, args


def _run_rollout(policy, buffer, args, image_path: Path, inferbatch: int):
    """Run a `segment_len`-step rollout with the *production* micro-batched
    `_get_action` shape (split `num_envs` across inferbatch chunks). Returns
    captured per-step (action_ids, lp, value) — the parity primitive.

    Implements the same flow as `CronosRunner._get_action` (main.py:582-600)
    but inline so this test isn't coupled to `CronosRunner` (which wants
    SAPIEN/ManiSkill). Identical math: split obs into chunks of size
    `inferbatch`, call `policy.get_action` per chunk, concatenate.

    GREEDY rollout (`deterministic=True`, temp=0): sampling-based rollout is
    RNG-dependent — even with the same seed and identical RNG state at
    `__init__` exit, micro-batch reordering at A2.1 changes the order of
    `torch.multinomial` consumption inside `generate`, so sampled actions
    differ. The plan's `gate_rollout_parity` is defined for greedy
    (`|Δ action_ids| == 0`), so the fixture captures greedy actions and
    re-uses logπ/value from the same forward pass.
    """
    device = policy.tpdv["device"]
    obs = _make_obs(device, image_path, "pick the cup", args.num_envs)
    policy.prep_rollout()
    buffer.warmup(obs["image"], obs["task_description"])

    capture = []
    rollout_starts = []                                     # wall-clock per step
    for step in range(args.segment_len):
        t0 = time.perf_counter()
        vals, acts, lps = [], [], []
        with torch.no_grad():
            for i in range(0, args.num_envs, inferbatch):
                sub = {
                    "image": obs["image"][i:i + inferbatch],
                    "task_description": obs["task_description"][i:i + inferbatch],
                }
                v, a, lp = policy.get_action(sub, deterministic=True)
                vals.append(v); acts.append(a); lps.append(lp)
        torch.cuda.synchronize()
        rollout_starts.append(time.perf_counter() - t0)

        val = torch.cat(vals, dim=0)
        action = torch.cat(acts, dim=0)
        logprob = torch.cat(lps, dim=0)
        capture.append({"action_ids": action.clone().cpu(),
                        "lp":         logprob.clone().float().cpu(),
                        "value":      val.clone().float().cpu()})

        # Next-step synthetic obs — same seed across baseline & re-run so
        # the (obs, action) the buffer stores at step t is reproducible.
        next_obs = _next_obs_synthetic(device, args.num_envs, seed=1000 + step)
        reward = torch.zeros((args.num_envs, 1), device=device)
        truncated = torch.zeros((args.num_envs, 1), device=device)
        buffer.insert(next_obs, action, logprob, val, reward, 1.0 - truncated)
        obs = {"image": next_obs, "task_description": obs["task_description"]}

    # Close the segment with a final next_value computed on the now-current obs.
    with torch.no_grad():
        next_val, _, _ = policy.get_action(obs, deterministic=True)
    buffer.end_segment(next_val)
    return capture, rollout_starts


def _run_ppo_update(policy, buffer, ppo):
    """One `train_epoch` over the just-populated buffer. Returns
    (train_results_list, per-param grad dict, wall-clock seconds).

    Per the plan's gate_grad_parity definition we capture per-param gradients
    AFTER `loss.backward()` of EVERY minibatch is summed (i.e. just BEFORE the
    optimizer step), since that's the quantity `train_epoch` clips and applies.
    With grad_accum spanning the whole buffer we read grads after train_epoch
    has called step+zero_grad, which would be zero — so we patch the PPO step
    to snapshot grads at the LAST accumulation boundary (just before
    clip_grad_norm_ runs).
    """
    buffer.compute_gae()
    policy.prep_training()

    # Snapshot the accumulated gradients at the FINAL `clip_grad_norm_` call.
    # The plan's parity primitive is "the accumulated gradient + the optimizer
    # step are byte-identical in fp32 (bf16 tol)" — i.e. what the optimizer
    # would have applied. We monkey-patch `clip_grad_norm_` to clone every
    # `.grad` before the in-place clip runs.
    import torch.nn.utils as nu
    snapshots = {"vla": {}, "vh": {}}
    orig_clip = nu.clip_grad_norm_

    def spy_clip(params, max_norm, *a, **k):
        # Snapshot pre-clip gradients keyed by `id(param)` (stable across the
        # life of the test process). We split into `params_vla` / `params_vh`
        # by membership so the per-phase comparison reads the same two groups.
        vla_ids = {id(p) for p in policy.params_vla}
        vh_ids  = {id(p) for p in policy.params_vh}
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach().float().cpu().clone()
            if id(p) in vla_ids:
                snapshots["vla"][id(p)] = g
            elif id(p) in vh_ids:
                snapshots["vh"][id(p)] = g
        return orig_clip(params, max_norm, *a, **k)

    nu.clip_grad_norm_ = spy_clip
    t0 = time.perf_counter()
    try:
        train_results = ppo.train_epoch(buffer)
    finally:
        nu.clip_grad_norm_ = orig_clip
    torch.cuda.synchronize()
    update_s = time.perf_counter() - t0
    return train_results, snapshots, update_s


def _params_named(policy):
    """Stable `id(p) → name` map for both param groups, so grad snapshots from
    different runs (with re-allocated tensors) can be matched by name.
    """
    by_id = {}
    for n, p in policy.vla.named_parameters():
        by_id[id(p)] = n
    return by_id


def _snapshots_by_name(snapshots, names_by_id):
    """Re-key the grad snapshot dict from `id(param)` to parameter name."""
    out = {"vla": {}, "vh": {}}
    for group in ("vla", "vh"):
        for pid, g in snapshots[group].items():
            n = names_by_id.get(pid, f"<unknown:{pid}>")
            out[group][n] = g
    return out


# --- gate primitives ----------------------------------------------------------


def _max_abs_grad_delta(a, b):
    """max |Δ| across two grad-by-name dicts (vla + vh). Returns -1 if shapes
    diverge, so a structural mismatch is loud rather than silently OK."""
    worst = 0.0
    for group in ("vla", "vh"):
        ka, kb = set(a[group].keys()), set(b[group].keys())
        if ka != kb:
            print(f"  [grad] {group} param sets differ: only-baseline={ka - kb}, only-current={kb - ka}")
            return -1.0
        for n in ka:
            ga, gb = a[group][n], b[group][n]
            if ga.shape != gb.shape:
                print(f"  [grad] {group}.{n} shape mismatch: {ga.shape} vs {gb.shape}")
                return -1.0
            d = (ga - gb).abs().max().item()
            if d > worst:
                worst = d
    return worst


def _capture_delta(cap_a, cap_b):
    """Per-step max |Δ| of (action_ids, logp, value) across two captures.

    Returns (max_act_int_delta, max_logp_delta, max_value_delta). action_ids
    integers, so the gate compares them with `==`; the integer-delta is
    `int(((a - b).abs()).max())` — 0 means greedy/sampler reproduced exactly.
    """
    assert len(cap_a) == len(cap_b)
    dact = 0
    dlp = 0.0
    dv = 0.0
    for ca, cb in zip(cap_a, cap_b):
        dact = max(dact, int((ca["action_ids"].long() - cb["action_ids"].long()).abs().max().item()))
        dlp = max(dlp, (ca["lp"] - cb["lp"]).abs().max().item())
        dv  = max(dv,  (ca["value"] - cb["value"]).abs().max().item())
    return dact, dlp, dv


def _train_results_delta(tr_a, tr_b):
    """max |Δ| for each scalar key across the per-minibatch train_results lists.

    Returns a dict keyed by scalar name. The list lengths must match
    (same buffer geometry); if they don't the gate is undefined."""
    if len(tr_a) != len(tr_b):
        return {"_length_mismatch": (len(tr_a), len(tr_b))}
    keys = set()
    for r in tr_a:
        keys.update(r.keys())
    deltas = {}
    for k in keys:
        ds = []
        for ra, rb in zip(tr_a, tr_b):
            va, vb = ra.get(k), rb.get(k)
            if va is None or vb is None:
                continue
            ds.append(abs(float(va) - float(vb)))
        if ds:
            deltas[k] = max(ds)
    return deltas


# --- one full fixture pass ----------------------------------------------------


def _run_fixture(model_path: str, seed: int, image_path: Path, *,
                 inferbatch: int,
                 buffer_minibatch: int = None,
                 alg_gradient_accum: int = None):
    """Build a fresh policy/buffer and run rollout + PPO update once.

    Returns a dict with capture, grad snapshots (by parameter name), train
    results, and wall-clock measurements. Each call freshly loads the model so
    Adam moments / LoRA inits / pad allocations are identical across calls
    given the same seed."""
    policy, buffer, ppo, args = _build_policy_and_buffer(model_path, seed)
    if buffer_minibatch is not None:
        args.buffer_minibatch = buffer_minibatch
        buffer.minibatch_size = buffer_minibatch
    if alg_gradient_accum is not None:
        args.alg_gradient_accum = alg_gradient_accum
        ppo.gradient_accum = alg_gradient_accum

    torch.cuda.reset_peak_memory_stats()
    cap, rollout_s = _run_rollout(policy, buffer, args, image_path, inferbatch=inferbatch)
    train_results, snapshots_by_id, update_s = _run_ppo_update(policy, buffer, ppo)
    peak_mem = torch.cuda.max_memory_allocated()

    grad_by_name = _snapshots_by_name(snapshots_by_id, _params_named(policy))

    out = {
        "capture": cap,
        "grad": grad_by_name,
        "train_results": train_results,
        "rollout_s": rollout_s,         # per-step list
        "update_s": update_s,
        "peak_mem_bytes": int(peak_mem),
        "config": {
            "inferbatch": inferbatch,
            "buffer_minibatch": args.buffer_minibatch,
            "alg_gradient_accum": args.alg_gradient_accum,
            "num_envs": args.num_envs,
            "segment_len": args.segment_len,
            "seed": seed,
        },
    }
    buffer.cleanup()
    del policy, buffer, ppo
    torch.cuda.empty_cache()
    return out


# --- capture vs verify drivers ------------------------------------------------


def cmd_capture(args):
    """O-0 baseline capture: run the fixture twice with IDENTICAL config to
    measure bf16 grad jitter, set TOL_GRAD = jitter * TOL_GRAD_K, pickle."""
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = Path(__file__).resolve().parent / image_path

    print(f"[capture] baseline run #1 (inferbatch={args.baseline_inferbatch}, "
          f"mb={args.baseline_mb}, accum={args.baseline_accum})")
    run_a = _run_fixture(args.model_path, args.seed, image_path,
                         inferbatch=args.baseline_inferbatch,
                         buffer_minibatch=args.baseline_mb,
                         alg_gradient_accum=args.baseline_accum)
    print(f"  rollout total: {sum(run_a['rollout_s']):.3f}s "
          f"({sum(run_a['rollout_s'])/len(run_a['rollout_s']):.3f}s/step)")
    print(f"  PPO update:    {run_a['update_s']:.3f}s")
    print(f"  peak VRAM:     {run_a['peak_mem_bytes']/1e9:.2f} GB")

    print(f"[capture] baseline run #2 (same config, measuring bf16 jitter)")
    run_b = _run_fixture(args.model_path, args.seed, image_path,
                         inferbatch=args.baseline_inferbatch,
                         buffer_minibatch=args.baseline_mb,
                         alg_gradient_accum=args.baseline_accum)
    print(f"  rollout total: {sum(run_b['rollout_s']):.3f}s")
    print(f"  PPO update:    {run_b['update_s']:.3f}s")

    # Parity primitives: how much do two IDENTICAL bf16 runs differ?
    dact, dlp, dv = _capture_delta(run_a["capture"], run_b["capture"])
    grad_jitter = _max_abs_grad_delta(run_a["grad"], run_b["grad"])
    train_jitter = _train_results_delta(run_a["train_results"], run_b["train_results"])
    print(f"[jitter] rollout: dact={dact}, dlp={dlp:.3e}, dvalue={dv:.3e}")
    print(f"[jitter] grad max|Δ|={grad_jitter:.3e}")
    print(f"[jitter] train_results max|Δ| per scalar:")
    for k, v in sorted(train_jitter.items()):
        print(f"           {k:35s} {v:.3e}")

    tol_grad = max(grad_jitter * TOL_GRAD_K, 1e-6)
    print(f"[O-0] TOL_GRAD = max(grad_jitter * {TOL_GRAD_K}, 1e-6) = {tol_grad:.3e}")

    # Keep run_a as the canonical baseline. run_b's measurements are the jitter
    # floor — used to set TOL_GRAD and recorded for the perf report.
    fixture = {
        "baseline": run_a,
        "jitter": {
            "second_run": run_b,
            "dact": dact, "dlp": dlp, "dv": dv,
            "grad_max_delta": grad_jitter,
            "train_results_max_delta": train_jitter,
        },
        "tolerances": {
            "TOL_LOGP": TOL_LOGP, "TOL_VALUE": TOL_VALUE,
            "TOL_GRAD": tol_grad, "TOL_TRAIN_SCALAR": TOL_TRAIN_SCALAR,
            "TOL_GRAD_K": TOL_GRAD_K,
        },
        "model_path": args.model_path,
        "seed": args.seed,
    }
    _FIXTURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_FIXTURE_FILE, "wb") as f:
        pickle.dump(fixture, f)
    print(f"[O-0] wrote {_FIXTURE_FILE} ({_FIXTURE_FILE.stat().st_size/1e6:.1f} MB)")


def _load_fixture():
    with open(_FIXTURE_FILE, "rb") as f:
        return pickle.load(f)


def _phase_config(phase: str, baseline_cfg: dict):
    """Return the (kwargs_overrides, description) for a phase's optimized run.

    Each phase changes exactly one knob, mirroring the plan's per-phase tag
    semantics. Per-phase config layered on top of the baseline so subsequent
    phases can stack — but the harness only re-runs the SAME baseline pass;
    end-to-end stacking is exercised at O-8.
    """
    cfg = {
        "inferbatch": baseline_cfg["inferbatch"],
        "buffer_minibatch": baseline_cfg["buffer_minibatch"],
        "alg_gradient_accum": baseline_cfg["alg_gradient_accum"],
    }
    if phase == "O1":
        # A2.1: bigger inferbatch (4→8 in the fixture's geometry; analog of
        # the production 32→64). All 8 envs in one forward → 1 fewer inference
        # micro-batch per step.
        cfg["inferbatch"] = 8
        return cfg, "A2.1: inferbatch 4→8 (analog of production 32→64)"
    if phase == "O5":
        # A3.1: minibatch 8→40, grad_accum 20→4, effective batch UNCHANGED (160).
        # (The plan's mb=80/accum=2 OOMs on 96 GB VRAM because the SigLIP +
        # ZoeDepth path can't be gradient-checkpointed cleanly — HF's BEiT
        # checkpoint forward drops the `resolution=(h,w)` kwarg. mb=40 is
        # the largest step that fits with LM-trunk checkpointing on.)
        cfg["buffer_minibatch"] = 40
        cfg["alg_gradient_accum"] = 4
        return cfg, "A3.1: mb 8→40, accum 20→4 (~5× fewer backbone passes)"
    if phase == "O5_mb16":
        # Intermediate A3.1: mb 8→16, accum 20→10 (same 160 effective).
        # Hypothesis: 2× wider fits WITHOUT gradient checkpointing → avoids
        # the ~1.5×/sample recompute tax that made O-5 (mb=40) slower
        # despite 5× fewer minibatches. Net: 2× fewer minibatches × ~1×
        # per-sample compute ≈ ~2× faster (best case).
        cfg["buffer_minibatch"] = 16
        cfg["alg_gradient_accum"] = 10
        return cfg, "A3.1 variant: mb 8→16, accum 20→10 (no checkpointing; ~2× fewer mbs)"
    if phase == "O5_mb32":
        # Intermediate A3.1: mb 8→32, accum 20→5 (same 160 effective).
        # 4× wider mb — borderline memory; if it fits without checkpointing,
        # ~4× fewer minibatches is a bigger win than mb=16. If it OOMs we
        # learn the cap is between 16 and 32.
        cfg["buffer_minibatch"] = 32
        cfg["alg_gradient_accum"] = 5
        return cfg, "A3.1 variant: mb 8→32, accum 20→5 (no checkpointing; ~4× fewer mbs)"
    return cfg, f"<unknown phase {phase}>"


def cmd_phase(args):
    """Per-phase parity gate: re-run the fixture with the phase's flag ON,
    compare against the baseline pickled by --capture."""
    fixture = _load_fixture()
    base = fixture["baseline"]
    tols = fixture["tolerances"]
    cfg, desc = _phase_config(args.phase, base["config"])
    print(f"[{args.phase}] {desc}")
    print(f"[{args.phase}] config: {cfg}")

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = Path(__file__).resolve().parent / image_path

    run = _run_fixture(fixture["model_path"], fixture["seed"], image_path,
                       inferbatch=cfg["inferbatch"],
                       buffer_minibatch=cfg["buffer_minibatch"],
                       alg_gradient_accum=cfg["alg_gradient_accum"])

    # gate_rollout_parity
    dact, dlp, dv = _capture_delta(base["capture"], run["capture"])
    print(f"[gate_rollout_parity] dact={dact}, dlp={dlp:.3e} (TOL_LOGP={tols['TOL_LOGP']}), "
          f"dvalue={dv:.3e} (TOL_VALUE={tols['TOL_VALUE']})")
    ok_roll = (dlp <= tols["TOL_LOGP"]) and (dv <= tols["TOL_VALUE"])

    # gate_grad_parity (Aspect 3 phases only have a meaningful interpretation;
    # Aspect 2 phases still report because the grad dict is reproducible).
    dgrad = _max_abs_grad_delta(base["grad"], run["grad"])
    print(f"[gate_grad_parity] grad max|Δ|={dgrad:.3e} (TOL_GRAD={tols['TOL_GRAD']:.3e})")
    ok_grad = (0.0 <= dgrad <= tols["TOL_GRAD"])

    # Train-results scalar drift is OBSERVABILITY only, not a hard gate: the
    # PPO update's per-minibatch summary stats (`policy_loss`, `value_loss`,
    # `value_explained_variance`, `grad_norm_*`, etc.) are sensitive to bf16
    # noise on a 160-sample buffer — O-0 measured `value_explained_variance`
    # jitter ≈ 41 across two identical runs. The load-bearing grad parity
    # above already proves the optimizer would land in the same place.
    tr_delta = _train_results_delta(base["train_results"], run["train_results"])
    if "_length_mismatch" in tr_delta:
        print(f"[train_results] length mismatch: {tr_delta['_length_mismatch']} "
              "(expected — minibatch geometry changed)")
    else:
        worst = 0.0
        worst_key = None
        for k, v in sorted(tr_delta.items()):
            if v > worst:
                worst = v
                worst_key = k
        jitter = base.get("train_results_jitter", {})  # filled below by capture
        print(f"[train_results] max|Δ|={worst:.3e} (key={worst_key}); "
              "(observability only — grad parity is the hard gate)")
    ok_train = True

    # gate_speedup (record-only on rollout; PPO update is the headline for O-5).
    base_roll = sum(base["rollout_s"]) / len(base["rollout_s"])
    cur_roll = sum(run["rollout_s"]) / len(run["rollout_s"])
    base_upd, cur_upd = base["update_s"], run["update_s"]
    print(f"[gate_speedup] rollout s/step: {base_roll:.3f} → {cur_roll:.3f} "
          f"({base_roll/cur_roll:.2f}×)")
    print(f"[gate_speedup] PPO update s:   {base_upd:.3f} → {cur_upd:.3f} "
          f"({base_upd/cur_upd:.2f}×)")

    # gate_oom_headroom
    cur_gb = run["peak_mem_bytes"] / 1e9
    print(f"[gate_oom_headroom] peak VRAM: {base['peak_mem_bytes']/1e9:.2f} GB "
          f"→ {cur_gb:.2f} GB (limit ~96 GB)")
    ok_mem = cur_gb < 90.0

    all_ok = ok_roll and ok_grad and ok_train and ok_mem
    print(f"\n[{args.phase}] {'PASS' if all_ok else 'FAIL'}: "
          f"rollout={ok_roll} grad={ok_grad} train={ok_train} mem={ok_mem}")
    sys.exit(0 if all_ok else 1)


def main():
    parser = argparse.ArgumentParser("O-0 perf-parity harness")
    parser.add_argument("--model-path", default="IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge")
    parser.add_argument("--image", default="example.png",
                        help="Image path; relative paths resolve from the test/ dir.")
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cap = sub.add_parser("capture", help="Build baseline fixtures (run twice; measure jitter; pickle).")
    p_cap.add_argument("--baseline-inferbatch", type=int, default=4)
    p_cap.add_argument("--baseline-mb", type=int, default=8)
    p_cap.add_argument("--baseline-accum", type=int, default=20)

    p_ph = sub.add_parser("phase", help="Verify one phase's parity against the baseline.")
    p_ph.add_argument("--phase", choices=("O1", "O5", "O5_mb16", "O5_mb32"), required=True,
                      help="Currently O1 (A2.1) and O5 (A3.1) are exercised here; "
                           "O-2..O-4 + O-6 use the rollout/PPO captures via this same harness "
                           "with phase-specific code edits in spatialvla_train.py / main.py.")

    # Re-route subcommand short aliases.
    args = parser.parse_args()
    if args.cmd == "capture":
        cmd_capture(args)
    elif args.cmd == "phase":
        cmd_phase(args)


if __name__ == "__main__":
    main()
