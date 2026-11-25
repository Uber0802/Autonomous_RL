import torch
import torch.nn as nn


def _grad_norm(params, norm_type=2.0):
    """Compute the joint Lp norm of a parameter list's gradients (NO clipping).

    Mirrors `torch.nn.utils.clip_grad_norm_`'s norm computation; logging-only.
    Returns 0.0 if no parameter has a grad (e.g. after `zero_grad`).
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type) for g in grads]),
        norm_type,
    )
    return float(total.item())


class CronosPPO:
    """Manages PPO training for the OpenVLA / SpatialVLA agents.

    This adds **additive, non-breaking** richer per-minibatch keys (ratio
    mean/median/max, `approx_kl`, `clip_fraction`, value-head
    explained-variance, per-group grad-norm logging, and the `g2_logp_gap`
    sanity scalar). Note: the return type stays a `list` of per-minibatch
    dicts — `train_results.extend(...)` at `main.py:774` is untouched. The
    aggregator at the wandb.log site (`main.py:1028`) summarizes across
    minibatches.

    The PPO MATH/algorithm (clipped surrogate, Huber value loss, joint
    `clip_grad_norm_`, two-optimizer step) is unchanged — the per-group
    grad-norm is logging-only.
    """

    def __init__(self, args, policy):
        self.args = args
        self.policy = policy
        self.clip_param = 0.2
        self.entropy_coef = args.alg_entropy_coef
        self.max_grad_norm = getattr(args, "vla_grad_norm", 10.0)
        self.gradient_accum = getattr(args, "alg_gradient_accum", 1)

    def train_epoch(self, buffer, log_path=None):
        """Runs one Epoch of PPO training on the provided buffer with gradient accumulation."""
        train_stats = []
        total_batches = (buffer.ep_len * buffer.num_env) // buffer.minibatch_size
        generator = buffer.feed_forward_generator()
        device = self.policy.tpdv["device"]

        self.policy.vla_optimizer.zero_grad()
        self.policy.vh_optimizer.zero_grad()

        for idx, batch in enumerate(generator):
            # 1. Prepare Batch (Move to device once)
            obs_dict = {
                "image": torch.tensor(batch["obs"]).to(device),
                "task_description": batch["instruct"]
            }
            actions = torch.tensor(batch["actions"]).to(device)
            returns = torch.tensor(batch["returns"]).to(device)
            logprobs_old = torch.tensor(batch["logprobs"]).to(device)
            advantages = torch.tensor(batch["advantages"]).to(device)
            values_old = torch.tensor(batch["values"]).to(device)

            # 2. Forward Pass & Evaluate
            logprobs, entropy, values = self.policy.evaluate_actions(obs_dict, actions)

            # 3. Policy Loss (Clipped Surrogate)
            ratio = torch.exp(logprobs - logprobs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 4. Value Loss (Clipped for Stability mapped to AutoRL Unscaled Huber)
            v_clipped = values_old + (values - values_old).clamp(-self.clip_param, self.clip_param)

            # AutoRL custom huber definition: a = abs(e)<=d; b = abs(e)>d -> a*e^2/2 + b*d*(abs(e)-d/2)
            delta = 10.0
            def huber_loss(err, d):
                a = (torch.abs(err) <= d).to(torch.float32)
                b = (torch.abs(err) > d).to(torch.float32)
                return a * err**2 / 2.0 + b * d * (torch.abs(err) - d / 2.0)

            error_clipped = returns - v_clipped
            error_unclipped = returns - values

            v_loss_clipped = huber_loss(error_clipped, delta)
            v_loss_unclipped = huber_loss(error_unclipped, delta)

            value_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

            # 5. Total Loss & Backward
            entropy_loss = entropy.mean()
            loss = (policy_loss + value_loss - self.entropy_coef * entropy_loss) / self.gradient_accum

            loss.backward()

            # Additive logging — compute BEFORE the clip+step+zero_grad block
            # below so per-group grad-norms read the gradients actually about
            # to be (jointly) clipped + applied. `approx_kl` follows the
            # Schulman convention (mean(old - new)); `clip_fraction` is the
            # fraction of minibatch ratios outside the clip band; value
            # `explained_variance` is the standard 1 - Var(ret - val) / Var(ret)
            #  — sampling-noise-robust on small minibatches.
            logp_diff = (logprobs - logprobs_old).detach()
            ratio_det = ratio.detach()
            approx_kl_val = float((-logp_diff).mean().item())
            clip_fraction_val = float(
                ((ratio_det - 1.0).abs() > self.clip_param).to(torch.float32).mean().item()
            )
            returns_f = returns.detach().to(torch.float32)
            values_f = values.detach().to(torch.float32)
            ret_var = float(returns_f.var(unbiased=False).item())
            value_explained_variance = (
                float(1.0 - ((returns_f - values_f).var(unbiased=False) / ret_var).item())
                if ret_var > 0 else 0.0
            )
            # G2 / m1: on the FIRST minibatch of an update, θ still equals the
            # rollout θ, so `(new - old) logprob` should be ≈ 0 (bf16 noise
            # only). Persistent non-zero g2_logp_gap[0] of either training run
            # is a real wiring bug.
            g2_logp_gap = float(logp_diff.abs().max().item())

            # 6. Optimization Step (Gradient Accumulation). Per-group
            # grad-norms are LOGGING-ONLY — read off `params_vla` and
            # `params_vh` separately just before the joint clip; the joint
            # `clip_grad_norm_` below still clips them together (PPO math
            # unchanged, plan §Scope discipline).
            grad_norm_val = None
            grad_norm_vla = _grad_norm(self.policy.params_vla)
            grad_norm_vh = _grad_norm(self.policy.params_vh)
            if (idx + 1) % self.gradient_accum == 0 or (idx + 1) == total_batches:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.params_vla + self.policy.params_vh, self.max_grad_norm
                )
                grad_norm_val = float(grad_norm.item())
                self.policy.vla_optimizer.step()
                self.policy.vh_optimizer.step()
                self.policy.vla_optimizer.zero_grad()
                self.policy.vh_optimizer.zero_grad()

            if log_path:
                log_str = (
                    f"[PPO Step {idx}/{total_batches}] "
                    f"Returns: {returns.mean().item():.4f} | "
                    f"Advantages: {batch['advantages'].mean():.4f} | "
                    f"Policy Loss: {policy_loss.item():.4f} | "
                    f"Value Loss: {value_loss.item():.4f} | "
                    f"Entropy Loss: {entropy_loss.item():.4f}\n"
                )
                with open(log_path, "a") as f:
                    f.write(log_str)

            stats = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_loss.item(),
                "total_loss": loss.item() * self.gradient_accum,
                # Additive keys.
                "ratio_mean": float(ratio_det.mean().item()),
                "ratio_median": float(ratio_det.median().item()),
                "ratio_max": float(ratio_det.max().item()),
                "approx_kl": approx_kl_val,
                "clip_fraction": clip_fraction_val,
                "value_explained_variance": value_explained_variance,
                "grad_norm_vla": grad_norm_vla,   # per-group, logging only
                "grad_norm_vh": grad_norm_vh,
                "g2_logp_gap": g2_logp_gap,
            }
            if grad_norm_val is not None:
                stats["grad_norm"] = grad_norm_val
            train_stats.append(stats)

        return train_stats


def aggregate_train_results(train_results):
    """Summarize a list of per-minibatch dicts into one dict for `wandb.log`.

    Used at `main.py:1028` (the `wandb.log` site) which previously logged
    `train_results[-1]` (the LAST minibatch only — dominated by post-step
    drift). Aggregate over the minibatch list rebuilt per update
    in `_run_ppo_update`.

    Aggregation choice (one per key — kept Python-side, no numpy required):
      - means         : losses, ratios, KL-style, clip fraction, expl-var,
                        per-group grad-norms.
      - medians       : `ratio_median` (already a median, just take median of medians).
      - maxes         : `ratio_max`, `grad_norm` (the joint clip norm).
      - first-mb only : `g2_logp_gap` (per plan: read from `train_results[0]`).
    """
    if not train_results:
        return {}

    keys = set()
    for r in train_results:
        keys.update(r.keys())

    def _mean(key):
        vals = [r[key] for r in train_results if key in r and r[key] is not None]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _max(key):
        vals = [r[key] for r in train_results if key in r and r[key] is not None]
        return float(max(vals)) if vals else 0.0

    def _median(key):
        vals = sorted(r[key] for r in train_results if key in r and r[key] is not None)
        if not vals:
            return 0.0
        n = len(vals)
        return float((vals[n // 2 - 1] + vals[n // 2]) / 2 if n % 2 == 0 else vals[n // 2])

    agg = {}
    # Means
    # `grpo_adv_zero_frac` is emitted only by CronosGRPO; the loop is
    # key-driven, so it is simply absent from PPO payloads.
    for k in ("policy_loss", "value_loss", "entropy", "total_loss",
              "ratio_mean", "approx_kl", "clip_fraction",
              "value_explained_variance", "grad_norm_vla", "grad_norm_vh",
              "grpo_adv_zero_frac"):
        if k in keys:
            agg[k] = _mean(k)
    # Median of medians (representative central ratio)
    if "ratio_median" in keys:
        agg["ratio_median"] = _median("ratio_median")
    # Maxes — the joint grad-norm is from the clip step (only the
    # accumulation-boundary minibatches have it); ratio_max is per-minibatch.
    if "ratio_max" in keys:
        agg["ratio_max"] = _max("ratio_max")
    if "grad_norm" in keys:
        agg["grad_norm"] = _max("grad_norm")
    # First-minibatch only: g2_logp_gap is the sanity scalar — must
    # come from `train_results[0]` (rollout θ unchanged on the very first
    # minibatch); reading the mean would smear it across post-step drift.
    if "g2_logp_gap" in train_results[0]:
        agg["g2_logp_gap"] = float(train_results[0]["g2_logp_gap"])
    return agg
