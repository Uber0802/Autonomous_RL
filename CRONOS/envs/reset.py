import torch
import numpy as np

class ResetStrategy:
    """Manages environment reset logic (robot, objects, unsuitable states)."""
    
    def __init__(self, env, num_envs=64, device="cuda"):
        self.env = env
        self.num_envs = num_envs
        self.device = device
        self.reset_unsuitable_count = 0

    def get_unsuitable_envs(self):
        """Identifies environments with objects in unsuitable positions (e.g., fallen)."""
        # Logic from simpler_wrapper.py
        obj_pos = self.env.unwrapped.get_obj_pos()
        recep_pos = self.env.unwrapped.get_recep_pos()
        obj_z = obj_pos[:, 2]
        recep_z = recep_pos[:, 2]

        low_z_mask = (obj_z < 0.7) | (recep_z < 0.7)
        env_indices = torch.nonzero(low_z_mask, as_tuple=False).squeeze()

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
        """Resets specific environments where objects have fallen or are unreachable."""
        self.reset_robot()
        count = self.env.unwrapped.reset_unsuitable_envs()
        self.reset_unsuitable_count += count
        return count
