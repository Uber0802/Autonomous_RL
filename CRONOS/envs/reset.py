import torch

from .unsuitable import UnsuitableDetector, get_detector


class ResetStrategy:
    """Manages environment reset logic (robot, objects, unsuitable states)."""

    def __init__(self, env, num_envs=64, device="cuda", detector="low_z"):
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
        """Resets specific environments where objects have fallen or are unreachable.

        V0.2 M2 Phase B: ask the registered detector for per-channel masks
        (``obj_mask``, ``recep_mask``) and forward them to the env-side
        respawn path. Detectors that don't expose ``per_actor_class`` fall
        back to deriving both masks from the single-mask ``__call__``, which
        respawns both carrot and plate together for any unsuitable env.
        """
        self.reset_robot()
        unwrapped = self.env.unwrapped
        if hasattr(self.detector, "per_actor_class"):
            masks = self.detector.per_actor_class(unwrapped)
            obj_mask = masks["obj"]
            recep_mask = masks["recep"]
        else:
            any_mask = self.detector(unwrapped)
            obj_mask = any_mask
            recep_mask = any_mask
        count = unwrapped.reset_unsuitable_envs(obj_mask=obj_mask, recep_mask=recep_mask)
        self.reset_unsuitable_count += count
        return count
