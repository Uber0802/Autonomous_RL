import copy
import random
import numpy as np
import os
import json
import torch

from transforms3d.euler import euler2axangle
from .action_utils import get_pose_base, get_pose_world
from .create_costmaps_gemini import CostMapCreator, CostMapHandler
# from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from .planner import plan_task
from ..prismatic.vla.action_tokenizer import ActionTokenizer

def initialize_cost_map(env, env_reset_options, save_dir, obs_camera_name, num_instruction, num_pos):
    os.makedirs(save_dir, exist_ok=True)
    costmap_root = os.path.join(save_dir, "costmaps")
    os.makedirs(costmap_root, exist_ok=True)
    all_costmap_handler = []
    for j in range(num_pos):
        env.rand_episode_id = env.save_episode_id + j * 10
        obs, instruction, _, _ = env.reset(**env_reset_options)     
        group_size = len(obs) // num_instruction 

        for i in range(num_instruction):
            print(f"[INFO] Generating costmap {j * num_pos + i}")

            image = obs[i * group_size].cpu().numpy()
            task_description = instruction[i * group_size]

            sub_dir = os.path.join(costmap_root, str(j * num_pos + i))
            os.makedirs(sub_dir, exist_ok=True)

            action_info_path = os.path.join(sub_dir, "action_info.json")
            if not os.path.exists(action_info_path):
                plan_traj_data = plan_task(task_description, image, idx=j * num_pos + i)
                with open(action_info_path, "w") as f:
                    json.dump(plan_traj_data, f, indent=4)
            else:
                with open(action_info_path, "r") as f:
                    plan_traj_data = json.load(f)

            # logger.debug("[DEBUG] Cost maps created using CostMapCreator.")
            costmap_creator = CostMapCreator(sub_dir, obs_camera_name)
            costmaps = costmap_creator.create_costmaps(
                env, env_reset_options, plan_traj_data, i * group_size)
            costmap_handler = CostMapHandler(costmaps)
            all_costmap_handler.append(costmap_handler)

    return all_costmap_handler

def reconstruct_action_from_norm_spatialvla(
    norm_vec: np.ndarray,
    action_tokenizer,
    model,
    unnorm_min: np.ndarray,
    unnorm_max: np.ndarray
) -> np.ndarray:
    """
    Tokenize a normalized action vector and then decode it back to the original scale.

    Args:
        norm_vec (np.ndarray): Action vector normalized to [-1, 1].
        action_tokenizer: Tokenizer for discretizing the action.
        model: Model containing the tokenizer for converting tokens to ids.
        unnorm_min (np.ndarray): Minimum values used for unnormalization.
        unnorm_max (np.ndarray): Maximum values used for unnormalization.

    Returns:
        decoded_unnorm (np.ndarray): Action vector in original scale.
    """
    # Discretize (tokenize)
    token_ids = action_tokenizer(norm_vec)
    token_ids = np.array([
        model.processor.tokenizer.convert_tokens_to_ids(t)
        for t in token_ids[0]
    ], dtype=np.int64)
    # Decode token ids back to normalized vector
    decoded_norm = action_tokenizer.decode_token_ids_to_actions(token_ids)[0]
    # Unnormalize to original scale
    decoded_unnorm = 0.5 * (decoded_norm + 1.0) * (unnorm_max - unnorm_min) + unnorm_min
    return decoded_unnorm


def reconstruct_action_from_norm_openvla(
    norm_vec: np.ndarray,
    action_tokenizer,
    model,
    unnorm_min: np.ndarray,
    unnorm_max: np.ndarray
) -> np.ndarray:
    """
    Tokenize a normalized action vector according to OpenVLA discretization
    and then decode it back to the original scale.

    Args:
        norm_vec (np.ndarray): Action vector normalized to [-1, 1].
        action_tokenizer: Tokenizer providing `bins` and `decode_token_ids_to_actions`.
        model: Unused in OpenVLA discretization.
        unnorm_min (np.ndarray): Minimum values used for unnormalization.
        unnorm_max (np.ndarray): Maximum values used for unnormalization.

    Returns:
        decoded_unnorm (np.ndarray): Action vector in original scale.
    """
    # 1) Discretize normalized vector into bins [1..n_bins]
    discretized = np.digitize(norm_vec, action_tokenizer.bins)
    # 2) Convert to token IDs by inverting the bin index
    token_ids = action_tokenizer.tokenizer.vocab_size - discretized
    token_ids = token_ids.astype(np.int64)

    # 3) Decode token IDs back to normalized action
    decoded_norm = action_tokenizer.decode_token_ids_to_actions(token_ids)

    # 4) Unnormalize to original scale
    decoded_unnorm = 0.5 * (decoded_norm + 1.0) * (unnorm_max - unnorm_min) + unnorm_min

    return decoded_unnorm

def encode_actions_from_norm_openvla(norm_vec: np.ndarray, action_tokenizer) -> np.ndarray:
    if isinstance(norm_vec, torch.Tensor):
        norm_vec = norm_vec.detach().cpu().numpy()
    discretized = np.digitize(norm_vec, action_tokenizer.bins)
    token_ids = action_tokenizer.tokenizer.vocab_size - discretized
    return token_ids.astype(np.int64)



def costmap_guided_sampling(
    model,
    env,
    color_img,
    task_description,
    costmap_handler,
    subtask_id,
    obs_camera_name,
    unnorm_min: np.ndarray,
    unnorm_max: np.ndarray,
    mode: str = None, 
    num_candidates: int = 600,
    noise_std: float = 0.04,
    top_k: int = 10,
    tolerance: float = 0.1,
) -> tuple:
    """
    Perform cost-map-guided sampling to select the best action.

    Returns:
        best_raw_action (dict): the raw action dict chosen
        best_action (dict): the formatted action dict chosen
    """
    # pick the correct reconstruct fn
    if mode.lower().startswith("spatialvla"):
        reconstruct_fn = reconstruct_action_from_norm_spatialvla
        action_tokenizer = model.processor.action_tokenizer
    elif mode.lower().startswith("openvla"):
        reconstruct_fn = reconstruct_action_from_norm_openvla
        action_tokenizer = ActionTokenizer(model.processor.tokenizer)
    else:
        reconstruct_fn = None
        action_tokenizer = None


    # 1) Get a base action from the model
    base_raw_action, base_action = model.step(color_img, task_description, do_sample=True)

    candidates = []
    for _ in range(num_candidates):
        # 2) Copy base action for candidate perturbation
        cand_raw = copy.deepcopy(base_raw_action)
        cand_act = copy.deepcopy(base_action)

        # 3) Add noise to translation and clip to valid range
        cand_act["world_vector"][:3] += np.random.normal(0, noise_std, size=3)
        cand_act["world_vector"][:3] = np.clip(
            cand_act["world_vector"][:3],
            unnorm_min[:3],
            unnorm_max[:3],
        )
        # Skip if out of action space bounds
        # if not (
        #     np.all(cand_act["world_vector"][:3] >= env.action_space.low[:3])
        #     and np.all(cand_act["world_vector"][:3] <= env.action_space.high[:3])
        # ):
        #     continue

        # 4) Normalize
        raw_vec = np.concatenate([
            cand_act["world_vector"],
            cand_raw["rotation_delta"],
            cand_raw["open_gripper"].ravel()
        ]).astype(np.float32)
        norm_vec = 2 * (raw_vec - unnorm_min) / (unnorm_max - unnorm_min) - 1
        norm_vec = np.clip(norm_vec, -1.0, 1.0)

        # 5) Reconstruct action using helper
        if reconstruct_fn is not None:
            decoded_unnorm = reconstruct_fn(
                norm_vec,
                action_tokenizer,
                model,
                unnorm_min,
                unnorm_max
            )
        else:
            decoded_unnorm = raw_vec

        # 6) Write back to cand_raw
        cand_raw["world_vector"]   = decoded_unnorm[:3]
        cand_raw["rotation_delta"] = decoded_unnorm[3:6]
        cand_raw["open_gripper"]   = decoded_unnorm[6:].reshape(1)

        # 7) Construct cand_act for cost evaluation
        cand_act["world_vector"] = cand_raw["world_vector"]
        roll, pitch, yaw = cand_raw["rotation_delta"]
        axis, angle = euler2axangle(roll, pitch, yaw)
        cand_act["rot_axangle"] = axis * angle
        cand_act["gripper"] = 2.0 * (cand_raw["open_gripper"] > 0.5) - 1.0

        # 8) IK check and forward kinematics error tolerance
        ik = env.unwrapped.agent.controller.controllers['arm'].compute_ik(
            get_pose_base(env, cand_act)
        )
        if ik is None:
            continue
        fk = env.unwrapped.agent.controller.controllers['arm'].compute_fk(ik)
        if np.linalg.norm(np.array(fk.p) - np.array(get_pose_base(env, cand_act).p)) > tolerance:
            continue

        # 9) Evaluate cost using costmap
        world_pos = get_pose_world(env, cand_act).p
        cost, grid_idx = (
            costmap_handler.eval_cost(subtask_id, world_pos, env, obs_camera_name)
            if costmap_handler else (0, None)
        )

        candidates.append((cost, grid_idx, cand_act, cand_raw))

    # 10) If no valid candidates, fallback to model.sample()
    if not candidates:
        print("[WARNING] No cand_action can solve IK")
        best_raw_action, best_action = model.step(color_img, task_description, do_sample=True)
        return best_raw_action, best_action, 10

    # 11) Sort by cost, pick one from top_k at random
    candidates.sort(key=lambda x: x[0])
    best_cost, _, best_action, best_raw_action = random.choice(candidates[:top_k])
    return best_raw_action, best_action, best_cost
