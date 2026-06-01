import torch

from .unsuitable import UnsuitableDetector, get_detector


VALID_RESET_SCOPES = ("per_actor", "per_env", "all")


class ResetStrategy:
    """Manages environment reset logic (robot, objects, unsuitable states)."""

    def __init__(self, env, num_envs=64, device="cuda", detector="low_z",
                 reset_scope: str = "per_env"):
        self.env = env
        self.num_envs = num_envs
        self.device = device
        self.reset_unsuitable_count = 0
        # V0.2 M2 Phase A+B: pluggable detector. Default `low_z` preserves
        # V0.1 behavior. Phase B closes the loop: `reset_unsuitable_envs`
        # now consults the detector via `per_actor_class()` and passes the
        # resulting masks to the env's respawn path. The detector is the
        # single source of truth for which envs/actors are unsuitable.
        self.detector: UnsuitableDetector = get_detector(detector)
        # V0.4 M3: HSR respawn scope. Three granularities:
        #   "per_actor" — respawn only the specific actor (obj OR recep) that
        #                 tripped the detector. Other actors of the same env
        #                 stay where they were. V0.3.1 behavior.
        #   "per_env"   — respawn BOTH actors (obj + recep) of any env where
        #                 either is flagged. Other envs untouched. DEFAULT —
        #                 matches the natural "reset this env's task" semantic.
        #   "all"       — respawn every actor of every env whenever any env
        #                 trips. Visually uniform across the batch.
        if reset_scope not in VALID_RESET_SCOPES:
            raise ValueError(
                f"reset_scope must be one of {VALID_RESET_SCOPES}, got {reset_scope!r}"
            )
        self.reset_scope = reset_scope

    def get_unsuitable_envs(self):
        """Identifies environments with objects in unsuitable positions (e.g., fallen).

        Delegates to the registered detector. Returns a list of env indices
        for backward compatibility with the V0.1 API; convert to a bool mask
        upstream if you need it in tensor form.
        """
        mask = self.detector(self.env.unwrapped)
        env_indices = torch.nonzero(mask, as_tuple=False).squeeze()
        if env_indices.ndim == 0:
            return [env_indices.item()] if env_indices.numel() > 0 else []
        return env_indices.tolist()

    def reset_robot(self):
        """Resets the robot to its initial pose without resetting the entire scene."""
        # Logic from simpler_wrapper.py
        env_idx = torch.arange(0, self.num_envs, device=self.device)
        self.env.unwrapped._elapsed_steps[env_idx] = 0
        self.env.unwrapped._clear_sim_state()
        self.env.unwrapped.agent.reset()
        self.env.unwrapped.agent.robot.set_pose(self.env.unwrapped.initial_robot_pos)
        self.env.unwrapped._settle(0.5)
        self.env.unwrapped.agent.reset(init_qpos=self.env.unwrapped.initial_qpos)
        
        self.env.unwrapped.scene._gpu_apply_all()
        self.env.unwrapped.scene.px.gpu_update_articulation_kinematics()
        self.env.unwrapped.scene._gpu_fetch_all()
        
        if isinstance(self.env.unwrapped.agent.controller, dict):
            for controller in self.env.unwrapped.agent.controller.values():
                controller.reset()
        else:
            self.env.unwrapped.agent.controller.reset()
        
        self.env.unwrapped.reset_grasp_stats()

    def reset_unsuitable_envs(self):
        """Respawns objects/receptacles in envs where the detector flagged
        them as unsuitable (fallen, ejected from workspace, etc.).

        V0.4 M3 (HSR Bug 1): the previous V0.3.1 implementation called
        ``self.reset_robot()`` unconditionally before the respawn, which made
        ``--reset-unsuitable --no-reset-robot`` silently behave like LSR+HSR.
        Now this method does ONLY the object-side respawn; the train loop
        decides independently whether to also reset the robot (matching
        AutoRL's two-independent-if pattern in train_ms3_ppo.py:637-643).
        """
        unwrapped = self.env.unwrapped
        reasons = None
        if hasattr(self.detector, "per_actor_class"):
            masks = self.detector.per_actor_class(unwrapped)
            obj_mask = masks["obj"]
            recep_mask = masks["recep"]
            reasons = masks.get("reasons")  # V0.4 M3c: optional per-env diagnostics
        else:
            any_mask = self.detector(unwrapped)
            obj_mask = any_mask
            recep_mask = any_mask

        # V0.4 M3 (full-env fix): derive a `fully_reset_envs` mask from the
        # scope. bridge_multi.reset_unsuitable_envs uses this to respawn
        # EVERY actor (all N carrots + all M plates) of each flagged env —
        # not just the task-relevant pair, which leaves non-task-relevant
        # actors lingering wherever they fell. per_actor scope keeps the
        # V0.3.1 single-actor path (obj_mask / recep_mask passed through).
        # Reasons are preserved unchanged so the log line still explains *why*
        # the trigger fired, regardless of how widely the reset is applied.
        fully_reset_envs = None
        if self.reset_scope == "per_env":
            env_bad = obj_mask | recep_mask
            fully_reset_envs = env_bad
            # The full path covers every actor of these envs; null out the
            # per-actor masks so bridge_multi doesn't double-set_pose the
            # task-relevant carrot/plate.
            obj_mask = obj_mask & ~env_bad
            recep_mask = recep_mask & ~env_bad
        elif self.reset_scope == "all" and bool((obj_mask | recep_mask).any().item()):
            fully_reset_envs = torch.ones_like(obj_mask, dtype=torch.bool)
            obj_mask = torch.zeros_like(obj_mask)
            recep_mask = torch.zeros_like(recep_mask)
        # else scope=="per_actor": pass masks through as the detector emitted.

        count = unwrapped.reset_unsuitable_envs(
            obj_mask=obj_mask, recep_mask=recep_mask,
            reasons=reasons, fully_reset_envs=fully_reset_envs,
        )
        self.reset_unsuitable_count += count
        return count
