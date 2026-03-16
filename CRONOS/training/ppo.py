import torch
import torch.nn as nn
from torch.optim import AdamW

class CronosPPO:
    """Manages PPO training for the OpenVLA agent."""
    
    def __init__(self, args, policy):
        self.args = args
        self.policy = policy
        self.clip_param = 0.2
        self.entropy_coef = args.alg_entropy_coef
        self.max_grad_norm = getattr(args, "vla_grad_norm", 10.0)
        self.gradient_accum = getattr(args, "alg_gradient_accum", 1)
        
    def train_epoch(self, buffer):
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
            # if idx == 0:
            #     print(f"[DEBUG MINIBATCH] returns: {returns.mean().item():.6f}/{returns.std().item():.6f}, values_old: {values_old.mean().item():.6f}/{values_old.std().item():.6f}, advantages: {advantages.mean().item():.6f}/{advantages.std().item():.6f}, logprobs_old: {logprobs_old.mean().item():.6f}/{logprobs_old.std().item():.6f}", flush=True)

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
            
            # 6. Optimization Step (Gradient Accumulation)
            if (idx + 1) % self.gradient_accum == 0 or (idx + 1) == total_batches:
                nn.utils.clip_grad_norm_(self.policy.params_vla + self.policy.params_vh, self.max_grad_norm)
                self.policy.vla_optimizer.step()
                self.policy.vh_optimizer.step()
                self.policy.vla_optimizer.zero_grad()
                self.policy.vh_optimizer.zero_grad()
            
            train_stats.append({
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_loss.item(),
                "total_loss": loss.item() * self.gradient_accum
            })
            
        return train_stats
