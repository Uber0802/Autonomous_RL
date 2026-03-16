import torch

class RewardShaper:
    """Computes rewards and extends environment information based on tasks."""
    
    def __init__(self, num_envs=64, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        self.reward_old = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.backward = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Stateful info tracking
        self.consecutive_grasp = torch.zeros((num_envs,), dtype=torch.int32, device=self.device)
        self.is_src_obj_grasped_ever = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self.consecutive_grasp_ever = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)

    def set_backward_mask(self, mask):
        """Sets which environments are in 'backward' mode."""
        self.backward = mask

    def compute_reward(self, info):
        """Computes current reward difference matching AutoRL exactly."""
        backward_mask = self.backward.reshape(-1, 1)

        reward = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        
        reward += info["is_src_obj_grasped"].reshape(-1, 1).float() * 0.1
        reward += info["consecutive_grasp"].reshape(-1, 1).float() * 0.1
        
        success_grasped = (info["success"].reshape(-1, 1) & info["is_src_obj_grasped"].reshape(-1, 1)).float()
        backward_success_grasped = (info["src_on_table"].reshape(-1, 1) & info["is_src_obj_grasped"].reshape(-1, 1)).float()
        
        reward += torch.where(
            backward_mask,
            backward_success_grasped * 1.0,  # if backward, only src_on_table counts
            success_grasped * 1.0            # else, success & grasped
        )

        reward_diff = reward - self.reward_old
        self.reward_old = reward.clone()
        
        return reward_diff

    def compute_info(self, env):
        """Computes all performance metrics from raw environment state using vectorized operations."""
        unwrapped = env.unwrapped
        b = self.num_envs
        
        # 1. Fetch raw states (Standardized vectorized retrieval)
        carrot_p = unwrapped.get_carrot_pose()
        plate_p = unwrapped.get_plate_pose()
        plate2_p = unwrapped.get_extra_plate_pose()
        
        # 2. Compute Grasping Stats
        # is_src_obj_grasped needs to be checked per environment
        is_src_obj_grasped = torch.zeros((b,), dtype=torch.bool, device=self.device)
        carrot_actors = unwrapped.get_carrot_actors()
        for i, actor in enumerate(carrot_actors):
            is_src_obj_grasped[i] = unwrapped.agent.is_grasping(actor)[i]
            
        self.consecutive_grasp += is_src_obj_grasped.int()
        self.consecutive_grasp[is_src_obj_grasped == 0] = 0
        consecutive_grasp = self.consecutive_grasp >= 5
        
        self.is_src_obj_grasped_ever |= is_src_obj_grasped
        self.consecutive_grasp_ever |= consecutive_grasp
        
        # 3. Compute Success (On Target)
        xy_flag_req = 0.015 # Slightly relaxed for stability
        z_flag_req = 0.05
        
        # Use world_bbox if available, otherwise default
        tgt_half = unwrapped.plate_bbox_world / 2 if hasattr(unwrapped, "plate_bbox_world") else torch.tensor([0.05, 0.05, 0.01], device=self.device)
        src_half = unwrapped.carrot_bbox_world / 2 if hasattr(unwrapped, "carrot_bbox_world") else torch.tensor([0.02, 0.02, 0.04], device=self.device)
        
        offset = carrot_p - plate_p
        xy_flag = torch.linalg.norm(offset[:, :2], dim=1) <= tgt_half.max(dim=1).values + xy_flag_req
        z_flag = (offset[:, 2] > 0) & (offset[:, 2] - tgt_half[:, 2] - src_half[:, 2] <= z_flag_req)
        src_on_target = xy_flag & z_flag
        
        # 4. Compute Table State (Target 2)
        offset2 = carrot_p - plate2_p
        xy_flag2 = torch.linalg.norm(offset2[:, :2], dim=1) <= tgt_half.max(dim=1).values + xy_flag_req
        z_flag2 = (offset2[:, 2] > 0) & (offset2[:, 2] - tgt_half[:, 2] - src_half[:, 2] <= z_flag_req)
        src_on_target2 = xy_flag2 & z_flag2
        
        src_not_on_any = (~src_on_target) & (~src_on_target2)
        src_on_table = src_not_on_any & (~is_src_obj_grasped) & (carrot_p[:, 2] >= 0.7) # Table height is ~0.87
        
        # 5. Assemble Info
        tcp_p = unwrapped.agent.robot.get_links()[-1].pose.p
        dist_tcp_to_obj = torch.linalg.norm(tcp_p - carrot_p, dim=1)
        dist_obj_to_recep = torch.linalg.norm(carrot_p - plate_p, dim=1)
        
        info = {
            "success": src_on_target,
            "is_src_obj_grasped": is_src_obj_grasped,
            "consecutive_grasp": consecutive_grasp,
            "src_on_target": src_on_target,
            "src_on_target2": src_on_target2,
            "src_on_table": src_on_table,
            "gripper_carrot_dist": dist_tcp_to_obj,
            "dist_obj_to_recep": dist_obj_to_recep,
            "target_object": unwrapped.source_obj_name,
            "target_receptacle": unwrapped.target_obj_name,
            "target_object_pose": carrot_p,
            "target_receptacle_pose": plate_p
        }
        
        return info

    def reset_reward_stats(self):
        """Resets reward history and performance tracking for a new sequence."""
        self.reward_old.zero_()
        self.consecutive_grasp.zero_()
        self.is_src_obj_grasped_ever.zero_()
        self.consecutive_grasp_ever.zero_()
