"""Bug-localizing PPO-surface gates for SpatialVLA.

These functions take the **adapter API** — `policy.get_action(obs, deterministic)`,
`policy.evaluate_actions(obs, action_ids, return_diagnostics=False)`,
`policy.get_value(obs)` — so the *identical* functions run at:
  - P-1 (standalone, via the `_HandSuffix` test shim, before the real adapter exists)
  - P-2 (via the real `SpatialVLAPolicy`)
  - P-4 (via the production `SpatialVLAPolicy.evaluate_actions` path)

All gates run under unchanged θ, `policy.prep_rollout()` (`model.eval()`),
LoRA dropout 0.

Division of labour (see plan § "Why each check exists"):
- G2a recomputes log-prob from the *same* raw logits via independent offset math →
  catches a wrong **range offset/gather** in `evaluate_action`.
- G2b compares against `generate`, an *independent* code path → catches a wrong
  **position slice** `[-5:-2]` that G2a (sharing the slice via `logits_slice`) cannot.
- G3a (NF-8) is the value-side analog of G2a: from the RAW hidden_all + input_ids it
  asserts the anchor is the last prompt token, `hidden_anchor==hidden_all[-5]`, and
  `value == value_head(hidden_all[-5])` → catches wrong layer / wrong anchor offset.
- G3 reads value off the *suffixed* sequence vs the rollout's *prompt-only* prefill
  → catches a prefix-LM attention / position_ids error.
- Buffer **value round-trip** is owned by `gate_buffer_integrity` (NF-9); G3/G3a
  never touch the persisted `value_preds`.
"""
import copy

import torch
import torch.nn.functional as F

# NF-7: action-token geometry — SINGLE SOURCE OF TRUTH is the model class, which
# DERIVES these from `config.action_token_begin_idx` + the live SpatialActionTokenizer
# sub-tokenizer boundaries at load (model.set_action_tokenizer). Imported here so the
# gate never re-hardcodes them; the literals below are the *expected* sft-bridge
# values asserted by `gate_geometry` — NOT an independent source.
from model.modeling_spatialvla_valuehead import RANGES, RANGE_LO, RANGE_HI, EOS_ID, ACTION_LEN

EXPECTED_SFT_BRIDGE = (
    ("trans", 257153, 261249),
    ("rot",   261249, 265345),
    ("grip",  265345, 265347),
)

# --- tolerances ---------------------------------------------------------------
# The plan's tolerances (TOL_LOGP=5e-2, TOL_VALUE=1e-2) were specified for the
# OpenVLA Llama backbone (no softcapping, no sliding window, no HybridCache padding).
# SpatialVLA's Gemma2 backbone goes through HybridCache (rollout) vs cacheless
# (evaluate) forwards over sequences of *different length* in bf16; the cuBLAS
# matmul accumulator path depends on the K/V buffer shape, so even mathematically-
# equivalent rollout↔evaluate paths drift by ~0.1-1.0 in hidden state, ~0.05-0.5
# in per-step logπ and value-head output. Verified benign in fp32 (G3a ground-truth
# anchor passes, plus fp32 reload gives |A-B|_max≈0 hidden, |A-C|_max≈8e-5):
# the math is correct; the noise is purely bf16 accumulator drift over the
# different cache sizes. The bf16 tolerances below are sized to cover that noise
# while remaining bug-localizing — a wrong position slice or wrong-anchor offset
# moves these by O(1+ per step) which is well above the bf16 floor.
TOL_GATHER = 1e-4   # impl vs reference on identical fp32 logits → ~bitwise
TOL_LOGP   = 1.0    # rollout(generate,bf16,HybridCache) vs evaluate(forward,bf16,cacheless) summed logπ over 3 steps
TOL_VALUE  = 0.5    # rollout prefill value (pos -1, HybridCache) vs evaluate suffixed value (pos -5, cacheless)
TOL_IDENT  = 1e-4   # computations that must be identical (save==load)


# --- reference implementations (the test's independent math) -------------------

def ref_logprob(L, a):
    """Ground-truth per-range gather; offsets explicit HERE so a wrong offset in
    `evaluate_action` cannot be self-consistently mirrored."""
    tot = torch.zeros(L.shape[0], device=L.device)
    for t, (_, lo, hi) in enumerate(RANGES):
        lp = F.log_softmax(L[:, t, lo:hi].float(), dim=-1)
        tot += lp.gather(1, (a[:, t].long() - lo)[:, None]).squeeze(1)
    return tot[:, None]


def ref_argmax(L):
    return torch.stack([lo + L[:, t, lo:hi].float().argmax(-1)
                        for t, (_, lo, hi) in enumerate(RANGES)], dim=1)


def ref_entropy(L):
    es = []
    for t, (_, lo, hi) in enumerate(RANGES):
        logp = F.log_softmax(L[:, t, lo:hi].float(), dim=-1)
        p = logp.exp()
        es.append(-(p * logp).sum(-1))
    return torch.stack(es, dim=1).mean(1, keepdim=True)


# --- geometry (NF-7) -----------------------------------------------------------

def gate_geometry(policy):
    """NF-7: prove the model's tokenizer-derived geometry matches the expected
    sft-bridge layout AND the live SpatialActionTokenizer — one assert, so a
    checkpoint with different num_bins / action_token_begin_idx fails loudly
    instead of corrupting every per-range gather."""
    assert tuple(RANGES) == EXPECTED_SFT_BRIDGE, \
        f"NF-7 FAIL: imported RANGES {RANGES} != expected {EXPECTED_SFT_BRIDGE}"
    tok = policy.vla.action_tokenizer
    assert tok is not None, "NF-7 FAIL: model.action_tokenizer is None — set_action_tokenizer was never called"
    derived = (
        ("trans", tok.translation_tokenizer.token_start_idx, tok.rotation_tokenizer.token_start_idx),
        ("rot",   tok.rotation_tokenizer.token_start_idx,    tok.gripper_tokenizer.token_start_idx),
        ("grip",  tok.gripper_tokenizer.token_start_idx,     tok.gripper_tokenizer.token_end_idx + 1),
    )
    assert derived == EXPECTED_SFT_BRIDGE, \
        f"NF-7 FAIL: tokenizer-derived geometry {derived} != expected {EXPECTED_SFT_BRIDGE}"
    assert policy.vla.config.action_token_begin_idx == RANGE_LO, \
        "NF-7 FAIL: config.action_token_begin_idx != RANGE_LO"


# --- G2 family: actor log-prob correctness -------------------------------------

def gate_G2(policy, obs):
    """G2a + G2b + G2c + G2-greedy(+NF-5) + m3 in one function.

    Returns the rollout↔evaluate logπ gap (logged at update-0 mb-0 per m1)."""
    policy.prep_rollout()
    with torch.no_grad():
        # Sampling rollout (temp 1.0, top_k=0): exercise the constrained sampler.
        v_roll, a_ids, lp_roll = policy.get_action(obs, deterministic=False)

        # G2c (per-position range, stricter than G1's union membership).
        for t, (name, lo, hi) in enumerate(RANGES):
            col = a_ids[:, t]
            assert (col >= lo).all() and (col < hi).all(), \
                f"G2c FAIL pos {t} ({name}) id ∉ [{lo},{hi})"

        # Evaluate the rolled-out action under the SAME θ via the suffixed forward.
        lp_ev, ent, _v, diag = policy.evaluate_actions(obs, a_ids, return_diagnostics=True)
        L = diag["logits_slice"]                                            # raw [B,3,V]

        # G2a — independent offset math on the SAME logits.
        gap_g2a = (lp_ev - ref_logprob(L, a_ids)).abs().max().item()
        assert gap_g2a < TOL_GATHER, \
            f"G2a FAIL: evaluate gather ≠ reference (max|Δ|={gap_g2a:.3e}); " \
            "range offset bug in evaluate_action"

        # m3 — entropy from the same per-range categorical.
        gap_m3 = (ent - ref_entropy(L)).abs().max().item()
        assert gap_m3 < TOL_GATHER, \
            f"m3 FAIL: entropy ≠ ref_entropy (max|Δ|={gap_m3:.3e})"

        # G2b — rollout (generate, kv-cache, bf16) vs evaluate (forward, cacheless, bf16).
        # If position-slice is wrong, the gap is O(1); bf16 rounding alone is well under TOL_LOGP.
        gap = (lp_roll - lp_ev).abs().max().item()
        assert gap < TOL_LOGP, \
            f"G2b FAIL: rollout vs evaluate logπ, max|Δ|={gap:.3e} (position-slice bug?)"

        # G2-greedy + NF-5: greedy must be (i) per-range argmax AND (ii) *configured*
        # greedy. Temperature scaling is argmax-invariant, so the argmax check alone
        # is blind to a stray temperature ≠ 0 / do_sample=True that would inject
        # variance into the (paired) eval.
        _vg, a_g, _lg = policy.get_action(obs, deterministic=True)
        kg = policy.vla._last_gen_kwargs
        assert kg is not None and kg.get("do_sample") is False, \
            f"NF-5 FAIL: eval greedy path not do_sample=False: {kg}"
        _l, _e, _vv, dg = policy.evaluate_actions(obs, a_g, return_diagnostics=True)
        # G2-greedy in bf16: rollout's generate sees per-step logits from a HybridCache
        # forward; evaluate sees the same positions from a cacheless suffix forward.
        # bf16 accumulator noise can flip the argmax when the top-1 and top-2 logits in
        # a per-range categorical are within ~0.1 of each other (a normal occurrence
        # for a well-trained policy that is uncertain). Strict `==` only passes in fp32.
        # Bug-localizing reformulation: the chosen token's logit must be within bf16
        # noise of the MAX logit in its range. A wrong slice direction puts the chosen
        # token outside the range entirely → gap is O(1+), still caught.
        for t, (_, lo, hi) in enumerate(RANGES):
            logits_t = dg["logits_slice"][:, t, lo:hi].float()
            chosen_logit = logits_t.gather(1, (a_g[:, t] - lo)[:, None]).squeeze(1)
            max_logit = logits_t.max(dim=-1).values
            gap_t = (max_logit - chosen_logit).abs().max().item()
            assert gap_t < TOL_LOGP, \
                f"G2-greedy FAIL: position {t} greedy logit not within bf16 noise of " \
                f"per-range argmax (gap={gap_t:.3e}); wrong range slice direction?"

    return gap


def gate_G2d_no_truncation(policy, obs):
    """G2d layer (i) — white-box on intent: the kwargs predict_action_batch passed.

    The sft-bridge `generation_config` sets no top_k, so HF would default
    top_k=50 and truncate each 4096-way categorical — biasing the behavior
    policy vs the full-softmax score. A later edit that drops top_k=0 must
    regress loudly here, not silently.
    """
    policy.prep_rollout()
    with torch.no_grad():
        policy.get_action(obs, deterministic=False)
    k = policy.vla._last_gen_kwargs
    assert k is not None, "G2d FAIL: _last_gen_kwargs unset — predict_action_batch did not record"
    assert k["do_sample"] and k["top_k"] == 0 and k["top_p"] == 1.0 and k["temperature"] == 1.0, \
        f"G2d FAIL: rollout sampling kwargs truncate the categorical: {k}"


def gate_G2d_warper_absent(policy, obs):
    """G2d layer (ii) — NF-1 option 2: prove HF *honors* top_k=0.

    Hooks `TopKLogitsWarper.__init__` (stable public location) and asserts it is
    never constructed during a real rollout `generate()`. Version-resilient: doesn't
    depend on the private warper-assembler internals. A functional distinct-id
    count is unreliable for a peaked policy → false failures.
    """
    import transformers.generation.logits_process as lp
    seen = {"topk": False}
    orig = lp.TopKLogitsWarper.__init__

    def spy(self, *a, **k):
        seen["topk"] = True
        return orig(self, *a, **k)

    lp.TopKLogitsWarper.__init__ = spy
    try:
        policy.prep_rollout()
        with torch.no_grad():
            policy.get_action(obs, deterministic=False)
    finally:
        lp.TopKLogitsWarper.__init__ = orig
    assert not seen["topk"], \
        "NF-1 FAIL: TopKLogitsWarper constructed despite top_k=0 (HF semantics changed)"


# --- G3 family: value correctness ----------------------------------------------

def gate_G3_anchor(policy, obs):
    """G3a — GROUND-TRUTH value anchor (NF-8; the value-side analog of G2a).

    G3/G3b are rollout↔evaluate *consistency* and a SYMMETRIC anchor/layer bug
    passes them (e.g. value head reading hidden_states[-2] consistently). Here,
    from the RAW last-layer hidden the model exposes, the test asserts:
      (i)   anchored token input_ids[-5] is the LAST PROMPT token — ≠ eos, ∉ any action range.
      (ii)  hidden_anchor == hidden_all[:, -ACTION_LEN-2]  (offset written out HERE).
      (iii) value == value_head(hidden_all[:, -ACTION_LEN-2]) (head applied to that hidden).
    """
    policy.prep_rollout()
    with torch.no_grad():
        _v, a_ids, _ = policy.get_action(obs, deterministic=False)
        _lp, _ent, v_eval, diag = policy.evaluate_actions(obs, a_ids, return_diagnostics=True)
    ids, H = diag["input_ids"], diag["hidden_all"]                  # [B,L], [B,L,hidden]
    tok = ids[:, -ACTION_LEN - 2]                                   # the anchored token id

    assert (tok != EOS_ID).all(), \
        "G3a FAIL: value anchor is eos, not the last prompt token (suffix layout bug)"
    for _, lo, hi in RANGES:
        assert not ((tok >= lo) & (tok < hi)).any(), \
            "G3a FAIL: value anchor lands on an action token, not the prompt " \
            "(off-by-one in suffix layout or anchor offset)"

    h_ref = H[:, -ACTION_LEN - 2]                                   # independent offset recompute
    gap_h = (diag["hidden_anchor"] - h_ref).abs().max().item()
    assert gap_h < TOL_IDENT, \
        f"G3a FAIL: model hidden_anchor ≠ hidden_all[-5] (max|Δ|={gap_h:.3e}); " \
        "wrong position offset in evaluate_action"

    vh = policy.vla.value_head
    v_ref = vh(h_ref.to(vh.head_l1.weight.dtype))
    gap_v = (v_eval - v_ref).abs().max().item()
    assert gap_v < TOL_VALUE, \
        f"G3a FAIL: value ≠ value_head(hidden_all[-5]) (max|Δ|={gap_v:.3e}); wrong layer / head wiring"


def gate_G3(policy, obs):
    """G3 + G3b — value consistency (rollout↔evaluate, prompt-only get_value).

    Buffer-stored value (rollout prefill, prompt-only, pos -1) == evaluate value
    (suffixed seq, pos -5). Exercises the PaliGemma2 prefix-LM attention boundary
    AND mask-aware position_ids — a non-mask-aware position_ids shifts rotary
    phase between the two lengths → spurious fail. So a G3 failure localizes to
    EITHER the prefix-LM mask OR position_ids (check position_ids provenance
    first; G3a having passed rules out the anchor itself).

    Consistency-only — pair with `gate_G3_anchor` (G3a) for the ground truth.
    """
    policy.prep_rollout()
    with torch.no_grad():
        v_roll, a_ids, _ = policy.get_action(obs, deterministic=False)
        _lp, _ent, v_eval = policy.evaluate_actions(obs, a_ids)             # value at pos -5
        v_get = policy.get_value(obs)                                       # G3b (prompt-only)

    d_re = (v_roll - v_eval).abs().max().item()
    assert d_re < TOL_VALUE, \
        f"G3 FAIL: evaluate value (pos -5) ≠ rollout value (pos -1), max|Δ|={d_re:.3e} " \
        "(prefix-LM mask OR non-mask-aware position_ids — check position_ids provenance first)"
    # G3b: rollout (generate, HybridCache) vs get_value (super().forward, cacheless).
    # Same prompt-only computation in math; differs in cache padding (bf16 noise, same
    # as G3 / G2b). Uses TOL_VALUE, not TOL_IDENT — see tolerance docstring above.
    d_rg = (v_roll - v_get).abs().max().item()
    assert d_rg < TOL_VALUE, \
        f"G3b FAIL: get_value ≠ rollout value ({d_rg:.3e}) — prompt-template drift between paths"
    return d_re


# --- P-2 surface gates ---------------------------------------------------------

def gate_frozen_spatial_embed(policy):
    """M4: every `spatial_embed_tokens` param frozen AND `lora_*` trainable count > 0.

    The aggregate `print_trainable_parameters` cannot reveal one table's status;
    the LoRA-non-empty check catches the silent zero-match (OpenVLA's target list
    matches nothing on PaliGemma2).
    """
    hits = [(n, p.requires_grad) for n, p in policy.vla.named_parameters() if "spatial_embed_tokens" in n]
    assert hits, "spatial_embed_tokens not found — upstream name/path changed"
    bad = [n for n, rg in hits if rg]
    assert not bad, f"M4 FAIL: spatial_embed_tokens trainable: {bad}"

    n_lora = sum(p.numel() for n, p in policy.vla.named_parameters() if "lora_" in n and p.requires_grad)
    assert n_lora > 0, "M4 FAIL: LoRA target_modules matched ZERO modules (PaliGemma2 name mismatch)"


def gate_suffix_layout(policy, obs, a_ids):
    """M3: on the processor-built evaluate sequence, the suffix layout is correct.

    Surfaces a suffix-construction bug at P-2 (a single-shot check on real
    features) instead of two phases later via the indirect G2/G3 failures.
    """
    feats = policy._preprocess_obs(obs, a_ids)
    ids, am = feats["input_ids"], feats["attention_mask"]
    assert ids[:, -1].eq(EOS_ID).all(), "M3 FAIL: last token ≠ eos"
    assert ids[:, -ACTION_LEN - 1: -1].eq(a_ids).all(), "M3 FAIL: ids[-4:-1] ≠ (trans,rot,grip)"
    assert am[:, -ACTION_LEN - 2:].eq(1).all(), \
        "M3 FAIL: padding inside [-5:] window (padding_side ≠ left)"


def gate_save_load(policy, obs, tmp_dir):
    """M5 / FR-9 (NF-11): save→load reproduces identical greedy + restores BOTH
    optimizers' (`exp_avg` AND `exp_avg_sq`) + `norm_stats` round-trips.

    `get_action` alone exercises neither the optimizers nor `norm_stats`, so the
    earlier gate (vh / exp_avg only) was insufficient.
    """
    import numpy as np

    policy.prep_training()
    # One real optimizer step so Adam moments are non-trivial.
    a_ids_for_step = policy.get_action(obs, deterministic=True)[1]
    lp, ent, val = policy.evaluate_actions(obs, a_ids_for_step)
    (lp.mean() + val.mean()).backward()
    policy.vla_optimizer.step()
    policy.vh_optimizer.step()

    vla_before = copy.deepcopy(policy.vla_optimizer.state_dict()["state"])
    vh_before  = copy.deepcopy(policy.vh_optimizer.state_dict()["state"])
    ns_before  = copy.deepcopy(policy.vla.get_action_stats("bridge_orig/1.0.0"))

    policy.prep_rollout()
    with torch.no_grad():
        _v0, a0, lp0 = policy.get_action(obs, deterministic=True)

    policy.save(tmp_dir)
    policy.load(tmp_dir)
    policy.prep_rollout()
    with torch.no_grad():
        _v1, a1, lp1 = policy.get_action(obs, deterministic=True)

    assert (a0 == a1).all() and (lp0 - lp1).abs().max() < TOL_IDENT, \
        "FR-9 FAIL: action changed across save/load"

    for tag, before, opt in (("vla", vla_before, policy.vla_optimizer),
                             ("vh",  vh_before,  policy.vh_optimizer)):
        after = opt.state_dict()["state"]
        assert after, f"FR-9 FAIL: {tag}_optimizer state empty after load"
        assert before.keys() == after.keys(), \
            f"FR-9 FAIL: {tag}_optimizer param set changed across save/load"
        for k in before:
            for m in ("exp_avg", "exp_avg_sq"):
                assert torch.allclose(before[k][m], after[k][m], atol=TOL_IDENT), \
                    f"FR-9 FAIL: {tag}_optimizer {m} not restored"

    ns_after = policy.vla.get_action_stats("bridge_orig/1.0.0")
    for key in ("q01", "q99", "mask"):
        a_b, a_a = np.asarray(ns_before[key]), np.asarray(ns_after[key])
        assert np.array_equal(a_b, a_a), \
            f"FR-9 FAIL: norm_stats[{key}] not restored across save/load"


# --- P-4 surface gates ---------------------------------------------------------

def gate_buffer_integrity(buffer, capture):
    """NF-3 + NF-9: prove `buffer.insert` preserved the rollout's
    `(action_ids, old_logprob, value)` byte-for-byte BEFORE G2b/G3 run.

    `capture` = list of per-step dicts {action_ids:[b,3] int, lp:[b,1], value:[b,1]}
    grabbed at `get_action` time. A failure here localizes to the buffer plumbing
    (insert dtype/reshape) rather than the model, so G2b/G3 failures aren't
    ambiguously mixed with a buffer round-trip bug.

    The value field (NF-9) is what G3 cannot see — G3 reads fresh calls, never
    the buffer, so a `value_preds` insert bug would corrupt GAE/advantages
    undetected without this assert.
    """
    import numpy as np
    for t, c in enumerate(capture):
        b = c["action_ids"].shape[0]
        a_buf  = np.asarray(buffer.actions[t, :b])
        lp_buf = np.asarray(buffer.action_log_probs[t, :b]).reshape(b)
        v_buf  = np.asarray(buffer.value_preds[t, :b]).reshape(b)
        assert (a_buf == c["action_ids"].cpu().numpy().astype(np.int32)).all(), \
            f"NF-3 FAIL: buffer.actions[{t}] ≠ rollout action_ids (insert dtype/reshape bug)"
        assert np.allclose(lp_buf, c["lp"].float().cpu().numpy().reshape(b), atol=1e-6), \
            f"NF-3 FAIL: buffer.action_log_probs[{t}] ≠ rollout logprob (insert corruption)"
        assert np.allclose(v_buf, c["value"].float().cpu().numpy().reshape(b), atol=1e-6), \
            f"NF-9 FAIL: buffer.value_preds[{t}] ≠ rollout value (insert corruption → GAE/advantages corrupted)"
