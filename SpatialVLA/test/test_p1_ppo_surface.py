"""P-1 standalone gate driver — model class PPO surface, no adapter yet.

Runs the P-1 surface gates (gate_geometry, gate_G2 family, gate_G3_anchor, gate_G3)
against the freshly-loaded `SpatialVLAForActionPredictionWithValueHead` via a
minimal `_HandSuffix` shim that mimics the not-yet-written adapter API
(`get_action`, `evaluate_actions`, `get_value`, `_preprocess_obs`, `prep_rollout`).

NF-13 caveat: the shim must build the prompt+suffix byte-identically to the
real processor path, else a mirrored wrong assumption yields a false P-1 pass.
P-2 re-runs these gates through the real adapter for the authoritative check.

Usage (from `Autonomous_RL/SpatialVLA/`):
    conda activate spatialvla_cronos
    CUDA_VISIBLE_DEVICES=0 python -m test.test_p1_ppo_surface \\
        --model-path IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor


def build_obs(processor, device, image_path: Path, instructions):
    """Build the CRONOS `obs` dict that the shim's `_preprocess_obs` consumes.

    `image` shape mirrors CRONOS's contract (uint8 [B,H,W,3] on device); the
    shim reshapes/permutes to processor expectations.
    """
    pil = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.asarray(pil, dtype=np.uint8)
    images = np.stack([arr for _ in instructions], axis=0)                   # [B,H,W,3]
    image_t = torch.from_numpy(images).to(device=device)
    return {"image": image_t, "task_description": list(instructions)}


class HandSuffixShim:
    """Minimal SpatialVLAPolicy stand-in for P-1 gates (NF-13).

    Mirrors the eval-side `SpatialVLAPolicy` paths needed by the gates:
      - get_action(obs, deterministic)               → predict_action_batch
      - get_value(obs)                               → model.get_value(prompt-only)
      - evaluate_actions(obs, a_ids, return_diag)    → preprocess(prompt + suffix) → model.evaluate_action
      - _preprocess_obs(obs, action_ids=None)        → exposed for gate_suffix_layout (P-2 reuse)

    The suffix path goes through the REAL processor (same code path the P-2
    adapter will use), so this shim is byte-identical to the processor path
    — only the surrounding policy plumbing is "hand-built" here.
    """
    _PROMPT_TEMPLATE = "What action should the robot take to {instruction}?"

    def __init__(self, model, processor, device, unnorm_key="bridge_orig/1.0.0"):
        self.vla = model
        self.processor = processor
        self.device = device
        self.unnorm_key = unnorm_key
        self.tpdv = dict(device=device, dtype=torch.bfloat16)

    def prep_rollout(self):
        self.vla.eval()

    def _preprocess_obs(self, x, action_ids=None):
        images = x["image"]
        task_description = x["task_description"]
        images_np = [images[i].cpu().numpy() for i in range(images.shape[0])]
        prompts = [self._PROMPT_TEMPLATE.format(instruction=t.lower()) for t in task_description]

        kwargs = dict(
            images=images_np,
            text=prompts,
            unnorm_key=self.unnorm_key,
            return_tensors="pt",
            padding=True,
        )
        if action_ids is not None:
            # Build the suffix from the EXACT sampled token ids (not from continuous
            # actions, which would re-discretize). The processor concatenates the
            # 3 action-token strings + eos and exposes token_type_ids+labels — the
            # same path the P-2 adapter will use.
            tok = self.processor.tokenizer
            suffix_strs = []
            for row in action_ids:
                pieces = tok.convert_ids_to_tokens(row.tolist())
                # convert_ids_to_tokens may return strings with the sentencepiece
                # underscore prefix; the action tokens are `<ACTION%05d>` literals
                # with no leading underscore, so concatenation is exact.
                suffix_strs.append("".join(pieces))
            kwargs["suffix"] = suffix_strs

        inputs = self.processor(**kwargs)
        inputs = inputs.to(**self.tpdv)
        return inputs

    @torch.no_grad()
    def get_action(self, obs, deterministic):
        # Mirror SpatialVLAPolicy.get_action — temp 1.0/top_k=0 for sampling;
        # do_sample=False for greedy (matches upstream predict_action).
        features = self._preprocess_obs(obs)
        values, action_ids, logprobs = self.vla.predict_action_batch(
            **features,
            unnorm_key=self.unnorm_key,
            do_sample=(not deterministic),
            temperature=1.0,
            top_k=0,
            top_p=1.0,
        )
        return values, action_ids, logprobs

    @torch.no_grad()
    def get_value(self, obs):
        features = self._preprocess_obs(obs)
        return self.vla.get_value(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            pixel_values=features["pixel_values"],
            intrinsic=features["intrinsic"],
        )

    def evaluate_actions(self, obs, action_ids, return_diagnostics=False):
        features = self._preprocess_obs(obs, action_ids=action_ids)
        out = self.vla.evaluate_action(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            pixel_values=features["pixel_values"],
            intrinsic=features["intrinsic"],
            labels=features["labels"],
            token_type_ids=features.get("token_type_ids"),
            unnorm_key=self.unnorm_key,
            return_diagnostics=return_diagnostics,
        )
        return out


def _run_gate(name, fn):
    try:
        out = fn()
        print(f"  PASS  {name}" + (f"  · returned {out:.3e}" if isinstance(out, float) else ""))
        return True, out
    except Exception as e:
        print(f"  FAIL  {name}  · {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser("P-1 PPO-surface gates")
    parser.add_argument("--model-path", required=True, help="HF model id or local path")
    parser.add_argument("--image", default="test/example.png", help="image to feed the gates")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    from model.modeling_spatialvla_valuehead import SpatialVLAForActionPredictionWithValueHead

    print(f"[load] processor + model from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    model = SpatialVLAForActionPredictionWithValueHead.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cuda:0",
        vh_mode="a0",
    )
    model.value_head._init_weights()
    # E-2 wiring (Option A' + NF-7): bridge processor stats + derive geometry.
    model.set_action_stats(processor.statistics)
    model.set_action_tokenizer(processor.action_tokenizer)

    shim = HandSuffixShim(model, processor, device=device, unnorm_key="bridge_orig/1.0.0")

    # Build a small obs batch from the example image.
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = Path(__file__).resolve().parent.parent / image_path
    instructions = ["pick the cup", "stack the blocks"][: args.batch_size]
    if len(instructions) < args.batch_size:
        instructions += [instructions[-1]] * (args.batch_size - len(instructions))
    obs = build_obs(processor, device, image_path, instructions)

    # Import gates after model is loaded (so the module-level RANGES reflect
    # the derived geometry, not the file-level defaults).
    from test.gates_ppo import (
        gate_geometry, gate_G2, gate_G2d_no_truncation, gate_G2d_warper_absent,
        gate_G3, gate_G3_anchor,
    )

    print("\n[P-1 gates]")
    results = []
    results.append(_run_gate("gate_geometry (NF-7)",         lambda: gate_geometry(shim)))
    results.append(_run_gate("gate_G2 (G2a/G2b/G2c/m3/greedy+NF-5)", lambda: gate_G2(shim, obs)))
    results.append(_run_gate("gate_G2d_no_truncation",       lambda: gate_G2d_no_truncation(shim, obs)))
    results.append(_run_gate("gate_G2d_warper_absent (NF-1)", lambda: gate_G2d_warper_absent(shim, obs)))
    results.append(_run_gate("gate_G3_anchor G3a (NF-8)",    lambda: gate_G3_anchor(shim, obs)))
    results.append(_run_gate("gate_G3 (G3+G3b)",             lambda: gate_G3(shim, obs)))

    n_pass = sum(1 for ok, _ in results if ok)
    n_total = len(results)
    print(f"\n[summary] {n_pass}/{n_total} gates passed")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
