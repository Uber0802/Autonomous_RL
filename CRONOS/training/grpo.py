"""CRONOS — GRPO trainer.

Port of AutoRL's `OpenVLAPPO.train_grpo_step` / `train_grpo`
(`AutoRL/SimplerEnv/simpler_env/policies/openvla/openvla_train.py:402-449, 530-556`)
onto CRONOS's buffer and policy interface. Deliberately mirrors
`training/ppo.py::CronosPPO` so the two can be swapped at one call site:
same constructor signature, same `train_epoch(buffer, log_path)` returning a
list of per-minibatch dicts, so `aggregate_train_results` is shared unchanged.

Differences from `CronosPPO`, all of them consequences of GRPO having no critic:

- No value loss. `evaluate_actions` still returns `values` (the head is baked
  into the model class) but the loss never references them, so no gradient
  reaches `params_vh`.
- Only `params_vla` is clipped, only `vla_optimizer` steps. `--vla-vhlr` is
  therefore inert under GRPO; the value head is frozen in practice even though
  it is still nominally trainable.
- The advantage comes from `buffer.compute_grpo_returns()` rather than
  `compute_gae()`, so `--buffer-gamma` / `--buffer-lambda` are also inert.

**Numerical note vs AutoRL.** AutoRL's `train_grpo_step` reduces the surrogate
with `.sum(dim=-1, keepdim=True).mean()` while its PPO step uses `.mean()`. In
AutoRL that is not cosmetic: its buffer allocates `action_log_probs` with width
`act_dim` (`replay_buffer.py:20`) but stores a `[B, 1]` joint log-prob into it,
which numpy broadcasts into `act_dim` identical columns — so the `sum` inflates
the GRPO policy loss by exactly `act_dim` (7 for OpenVLA, 3 for SpatialVLA)
relative to PPO. CRONOS's buffer stores `action_log_probs` at width 1
(`training/buffer.py`), so that broadcast cannot happen and `sum(dim=-1)` would
be a no-op. This implementation uses `.mean()`, which is the intended form.
Consequence: CRONOS GRPO losses are `act_dim` times *smaller* than AutoRL's for
the same rollout. Compare gradient norms, not raw loss values, across the two.
"""

import torch
import torch.nn as nn

from .ppo import _grad_norm


class CronosGRPO:
    """Critic-free GRPO with PPO's clipped surrogate.

    Constructor and `train_epoch` signature match `CronosPPO` exactly so
    `main.py` can pick between them with a single branch.
    """

    def __init__(self, args, policy):
        self.args = args
        self.policy = policy
        # Same constants as CronosPPO — AutoRL's GRPO path reuses the PPO
        # clip/grad-norm values verbatim (`openvla_train.py:286-287`).
        self.clip_param = 0.2
        self.entropy_coef = args.alg_entropy_coef
        self.max_grad_norm = getattr(args, "vla_grad_norm", 10.0)
        self.gradient_accum = getattr(args, "alg_gradient_accum", 1)

    def train_epoch(self, buffer, log_path=None):
        """One GRPO epoch over the buffer. Advantages must already be computed
        by `buffer.compute_grpo_returns(...)` — `main._run_ppo_update` does that,
        mirroring how the PPO path calls `compute_gae()` first."""
        train_stats = []
        total_batches = (buffer.ep_len * buffer.num_env) // buffer.minibatch_size
        generator = buffer.feed_forward_generator()
        device = self.policy.tpdv["device"]

        self.policy.vla_optimizer.zero_grad()

        for idx, batch in enumerate(generator):
            obs_dict = {
                "image": torch.tensor(batch["obs"]).to(device),
                "task_description": batch["instruct"],
            }
            actions = torch.tensor(batch["actions"]).to(device)
            logprobs_old = torch.tensor(batch["logprobs"]).to(device)
            advantages = torch.tensor(batch["advantages"]).to(device)
            # batch["values"] / batch["returns"] are deliberately unused.

            logprobs, entropy, _values = self.policy.evaluate_actions(obs_dict, actions)

            # Clipped surrogate — identical to the PPO path.
            ratio = torch.exp(logprobs - logprobs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy_loss = entropy.mean()
            loss = (policy_loss - self.entropy_coef * entropy_loss) / self.gradient_accum
            loss.backward()

            # Diagnostics, computed before the clip+step so the grad norms
            # describe the gradients actually about to be applied. Same key
            # names as CronosPPO where the quantity is the same, so wandb
            # panels line up across the two algorithms.
            logp_diff = (logprobs - logprobs_old).detach()
            ratio_det = ratio.detach()
            approx_kl_val = float((-logp_diff).mean().item())
            clip_fraction_val = float(
                ((ratio_det - 1.0).abs() > self.clip_param).to(torch.float32).mean().item()
            )
            adv_det = advantages.detach().to(torch.float32)
            # Fraction of the minibatch whose advantage is exactly zero. Under
            # per-group normalization this is the "dead group" rate — a group
            # whose rewards were all identical contributes no gradient at all.
            # Worth watching: it rises as the group size falls.
            adv_zero_frac = float((adv_det == 0).to(torch.float32).mean().item())
            g2_logp_gap = float(logp_diff.abs().max().item())

            grad_norm_val = None
            grad_norm_vla = _grad_norm(self.policy.params_vla)
            if (idx + 1) % self.gradient_accum == 0 or (idx + 1) == total_batches:
                # params_vla only — no critic to co-clip (AutoRL:
                # `openvla_train.py:394`).
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.params_vla, self.max_grad_norm
                )
                grad_norm_val = float(grad_norm.item())
                self.policy.vla_optimizer.step()
                self.policy.vla_optimizer.zero_grad()

            if log_path:
                log_str = (
                    f"[GRPO Step {idx}/{total_batches}] "
                    f"Advantages: {batch['advantages'].mean():.4f} | "
                    f"Policy Loss: {policy_loss.item():.4f} | "
                    f"Entropy Loss: {entropy_loss.item():.4f} | "
                    f"Zero-adv frac: {adv_zero_frac:.4f}\n"
                )
                with open(log_path, "a") as f:
                    f.write(log_str)

            stats = {
                "policy_loss": policy_loss.item(),
                "entropy": entropy_loss.item(),
                "total_loss": loss.item() * self.gradient_accum,
                "ratio_mean": float(ratio_det.mean().item()),
                "ratio_median": float(ratio_det.median().item()),
                "ratio_max": float(ratio_det.max().item()),
                "approx_kl": approx_kl_val,
                "clip_fraction": clip_fraction_val,
                "grad_norm_vla": grad_norm_vla,
                "g2_logp_gap": g2_logp_gap,
                "grpo_adv_zero_frac": adv_zero_frac,
            }
            if grad_norm_val is not None:
                stats["grad_norm"] = grad_norm_val
            train_stats.append(stats)

        return train_stats
