import os
import re
import pprint
import random
import gc
import signal
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated
import torch
import numpy as np
import tyro
import wandb
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video

from simpler_env.env.simpler_wrapper import SimlerWrapper, SDSimlerWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer

from split_decisions.utils.sampling_utils import costmap_guided_sampling, initialize_cost_map, encode_actions_from_norm_openvla, reconstruct_action_from_norm_openvla
from split_decisions.utils.action_utils import get_pose_base, get_pose_world
from split_decisions.prismatic.vla.action_tokenizer import ActionTokenizer
from split_decisions.utils.observation_utils import world_to_screen, world_to_screen_idx
import imageio
import cv2

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1"""

    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    name: str = "PPO-test"

    # env
    num_envs: int = 64
    episode_len: int = 80 # 80
    training_len: int = 80
    use_same_init: bool = True

    steps_max: int = 2000000
    steps_vh: int = 0  # episodes
    interval_eval: int = 2
    interval_save: int = 40

    # buffer
    buffer_inferbatch: int = 32
    buffer_minibatch: int = 8
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95

    # vla
    vla_path: str = "openvla/openvla-7b"
    vla_unnorm_key: str = "bridge_orig"
    vla_load_path: str = ""
    vla_lora_rank: int = 32

    vla_lr: float = 3e-5
    vla_vhlr: float = 1e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6

    # ppo & grpo
    alg_name: str = "ppo"  # ppo, grpo
    alg_grpo_fix: bool = True
    alg_gradient_accum: int = 640
    alg_ppo_epoch: int = 1
    alg_entropy_coef: float = 0.05

    # other
    wandb: bool = True
    only_render: bool = False
    render_info: bool = False



class Runner:
    def __init__(self, all_args: Args):
        self.args = all_args

        # alg_name
        assert self.args.alg_name in ["ppo", "grpo"]

        # set seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # set wandb
        wandb.init(
            config=all_args.__dict__,
            project="RLVLA",
            name=self.args.name,
            mode="online" if self.args.wandb else "offline",
        )
        self.save_dir = Path(wandb.run.dir)
        self.glob_dir = Path(wandb.run.dir) / ".." / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)

        yaml.dump(all_args.__dict__, open(self.glob_dir / "config.yaml", "w"))

        # policy
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device("cuda:" + str(device_id))
        self.policy = OpenVLAPolicy(all_args, device_id_other)

        self.alg = OpenVLAPPO(all_args, self.policy)

        # env
        unnorm_state = self.policy.vla.get_action_stats(self.args.vla_unnorm_key)
        self.env = SDSimlerWrapper(self.args, unnorm_state)

        # buffer
        self.buffer = SeparatedReplayBuffer(
            all_args,
            obs_dim=(480, 640, 3),
            act_dim=7,
        )
        minibatch_count = self.buffer.get_minibatch_count()
        print(f"Buffer minibatch count: {minibatch_count}")

        # Task Switch
        self.task_id = 0
        self.task_list = self.env.get_task_pool()[0]

    def extract_obj_recep(self, text_string):
        pattern = r"put (.*?) on (.*)"
        match = re.search(pattern, text_string)

        if match:
            obj = match.group(1)
            recep = match.group(2)
            return obj, recep
        else:
            return None, None

    @torch.no_grad()
    def _get_action(self, obs, deterministic=False):
        total_batch = obs["image"].shape[0]

        values = []
        actions = []
        logprobs = []

        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = {k: v[i:i + self.args.buffer_inferbatch] for k, v in obs.items()}
            value, action, logprob = self.policy.get_action(obs_batch, deterministic)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)

        values = torch.cat(values, dim=0).to(device=self.device)
        actions = torch.cat(actions, dim=0).to(device=self.device)
        logprobs = torch.cat(logprobs, dim=0).to(device=self.device)

        return values, actions, logprobs

    @torch.no_grad()
    def _get_action_costmap(self, envs, obs, costmap_handler, subtask_id, viz_writers, deterministic=False):
        total_batch = obs["image"].shape[0]

        final_values = []
        final_actions = []
        final_logprobs = []
        final_candidates = []
        final_costs = []

        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = {k: v[i:i + self.args.buffer_inferbatch] for k, v in obs.items()}

            top_k = 10
            action_low = torch.tensor(envs.unnorm_state["q01"], device=self.device).unsqueeze(0) 
            action_high = torch.tensor(envs.unnorm_state["q99"], device=self.device).unsqueeze(0)
            B = 50
            sigma = 0.2 * (action_low - action_high)
            action_tokenizer = ActionTokenizer(self.policy.processor.tokenizer)
            
            batch_values, batch_actions, batch_logprobs = self.policy.get_action(obs_batch, deterministic)
            for j in range(self.args.buffer_inferbatch):
                env_idx = i + j
                vis_img = obs_batch['image'][j]
                final_candidates = []

                actions = batch_actions[j]
                # processed_actions = envs._process_action(actions)
                decoded_norm = action_tokenizer.decode_token_ids_to_actions(actions.detach().cpu().numpy())
                actions = 0.5 * (decoded_norm + 1.0) * (action_high.cpu().numpy() - action_low.cpu().numpy()) + action_low.cpu().numpy()
                processed_actions = torch.from_numpy(actions).to(dtype=torch.float64, device="cuda:0")
                
                action_dim = actions.shape[-1]
                mask = torch.zeros(1, action_dim, device=self.device)
                mask[:, :3] = 1
                noise = torch.randn(B, action_dim, device=self.device) * sigma * mask
                processed_noised_actions = processed_actions.repeat(B, 1) + noise
                processed_noised_actions = torch.cat([processed_noised_actions, processed_actions], dim=0)
                processed_noised_actions = torch.clamp(processed_noised_actions, min=action_low, max=action_high)
                # print("0 :", processed_noised_actions)
                # processed_noised_actions = reconstruct_action_from_norm_openvla(processed_noised_actions.cpu().numpy(), action_tokenizer, envs, action_low.cpu().numpy(), action_high.cpu().numpy())
                # processed_noised_actions = torch.from_numpy(processed_noised_actions).to(dtype=torch.float64, device="cuda:0")
                # print("1 :", processed_noised_actions)
                normalized_actions = 2 * (processed_noised_actions - action_low) / (action_high - action_low) - 1.0
                token_ids_np = encode_actions_from_norm_openvla(normalized_actions, action_tokenizer) 
                token_ids = torch.tensor(token_ids_np, device=self.device)
                

                for k in range(len(processed_noised_actions)):
                    # check cand
                    cand = processed_noised_actions[k].unsqueeze(0)
                    # check get_pose_world points
                    points = get_pose_world(envs.env, cand, env_idx).p.detach().cpu().numpy()


                    cost, grid_idx, affordance_points = costmap_handler[env_idx].eval_cost(
                        subtask_id=subtask_id[env_idx],
                        point=points[env_idx],
                        env=envs.env,
                        obs_camera_name="3rd_view_camera",
                        idx=env_idx,
                    )

                    # visualize
                    # cand_2d = world_to_screen_idx(envs.env, "3rd_view_camera", points[env_idx], env_idx)
                    # cand_2d = cand_2d.flatten()
                    

                    # if isinstance(vis_img, torch.Tensor):
                    #     vis_img = vis_img.detach().cpu().numpy()

                    # h, w = vis_img.shape[:2]
                    # scale_x = w / 640.0
                    # scale_y = h / 480.0

                    # u, v = cand_2d
                    # x = int(u * scale_x)
                    # y = int(v * scale_y)

                    # if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
                    #     cv2.circle(vis_img, (int(x), int(y)), 1, (255, 0, 0), -1)

                    # u, v = affordance_points.flatten()
                    # x = int(u * scale_x)
                    # y = int(v * scale_y)

                    # if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
                    #     cv2.circle(vis_img, (int(x), int(y)), 3, (0, 255, 0), -1)

                    
                    final_candidates.append((cost, grid_idx, token_ids[k]))

                

                



                topk_candidates = sorted(final_candidates, key=lambda x: x[0])[:top_k]
                
                # if isinstance(vis_img, torch.Tensor):
                #     vis_img = vis_img.detach().cpu().numpy()
                # h, w = vis_img.shape[:2]
                # scale_x = w / 640.0
                # scale_y = h / 480.0

                # for _, _, _, c2d in topk_candidates:
                #     u, v = c2d.flatten()
                #     x = int(u * scale_x); y = int(v * scale_y)
                #     if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
                #         cv2.circle(vis_img, (int(x), int(y)), 2, (0, 0, 255), -1)
                # viz_writers[env_idx].append_data(vis_img)

                selected = random.choice(topk_candidates)
                selected_cost, _, selected_action = selected

                

                final_actions.append(selected_action.unsqueeze(0))
                final_costs.append(torch.tensor([selected_cost], device=self.device, dtype=torch.float32))  # [1]


            N = self.args.buffer_inferbatch
            tok_selected = torch.cat(final_actions[-N:], dim=0).to(self.device)        
            tok_selected = tok_selected.to(dtype=torch.long)                          

            mini = 8
            for s in range(0, N, mini):
                e = min(s + mini, N)
                obs_chunk = {
                    "image": obs_batch["image"][s:e],                                  
                    "task_description": [obs_batch["task_description"][k] for k in range(s, e)], 
                }
                tok_chunk = tok_selected[s:e]                                           
                logp, ent, val = self.policy.evaluate_actions(obs_chunk, tok_chunk)
                final_values.append(val)                                                
                final_logprobs.append(logp)     


        

        final_values = torch.cat(final_values, dim=0).to(device=self.device)
        final_actions = torch.cat(final_actions, dim=0).to(device=self.device)
        final_logprobs = torch.cat(final_logprobs, dim=0).to(device=self.device)
        final_costs = torch.cat(final_costs, dim=0).to(device=self.device)


        return final_values, final_actions, final_logprobs, final_costs

    def collect(self):
        self.policy.prep_rollout()

        obs_image = self.buffer.obs[self.buffer.step]
        obs_image = torch.tensor(obs_image).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        value, action, logprob = self._get_action(obs)

        return value, action, logprob

    def collect_costmap(self, envs, costmap_handler, subtask_id, viz_writers):
        self.policy.prep_rollout()

        obs_image = self.buffer.obs[self.buffer.step]
        obs_image = torch.tensor(obs_image).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        value, action, logprob, cost = self._get_action_costmap(envs=envs, obs=obs, costmap_handler=costmap_handler, subtask_id=subtask_id, viz_writers=viz_writers)
        # with open("/workspace/AutoRL_SD/log.txt", "a") as f:
        #     f.write(f"logprob : {logprob}\n")

        return value, action, logprob, cost

    def insert(self, data):
        obs_img, actions, logprob, value_preds, rewards, done = data
        masks = 1.0 - done.to(torch.float32)

        obs_img = obs_img.cpu().numpy()
        actions = actions.to(torch.int32).cpu().numpy()
        logprob = logprob.to(torch.float32).cpu().numpy()
        value_preds = value_preds.to(torch.float32).cpu().numpy()
        rewards = rewards.cpu().numpy()
        masks = masks.cpu().numpy()

        self.buffer.insert(obs_img, actions, logprob, value_preds, rewards, masks)

    def compute_endup(self):
        self.policy.prep_rollout()

        obs_image = torch.tensor(self.buffer.obs[-1]).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        with torch.no_grad():
            next_value, _, _ = self._get_action(obs)
        next_value = next_value.to(torch.float32).cpu().numpy()

        self.buffer.endup(next_value)

    def train(self, episode):
        self.policy.prep_training()

        if self.args.alg_name == "ppo":
            train_info = self.alg.train_ppo(self.buffer, episode)
        elif self.args.alg_name == "grpo":
            train_info = self.alg.train_grpo(self.buffer)
        else:
            raise ValueError(f"Unknown alg_name: {self.args.alg_name}")

        info = {f"train/{k}": v for k, v in train_info.items()}
        info["buffer/reward_mean"] = np.mean(self.buffer.rewards)
        info["buffer/mask_mean"] = np.mean(1.0 - self.buffer.masks)

        return info

    @torch.no_grad()
    def eval(self, obj_set: str, object: list[str], receptacle: list[str]) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])

        obs_img, instruction, info, _ = self.env.reset(obj_set=obj_set, same_init=self.args.use_same_init, object=object, receptacle=receptacle)
        print("Evaluating:", instruction[0])

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img, reward, done, env_info = self.env.step(action)

            # info
            # print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats = env_stats.copy()

        # print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        # print(f"")

        return env_stats

    @torch.no_grad()
    def eval_video(self, epoch: int, obj_set: str, object: list[str], receptacle: list[str]) -> dict:
        self.policy.prep_rollout()

        env_infos = defaultdict(lambda: [])
        obs_img, instruction, info, _ = self.env.reset(obj_set=obj_set, same_init=self.args.use_same_init, object=object,receptacle=receptacle)

        data = {
            "image": [],
            "instruction": "",
            "action": [],
            "info": [],
        }

        print("Evaluating:", instruction[0])
        data["instruction"] = instruction[0]

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img_new, reward, done, env_info = self.env.step(action)

            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    value = v[0]
                    if isinstance(value, bool):
                        env_infos[k].append(value)
                    else:
                        env_infos[k].append(value.item())


            post_action = self.env._process_action(action)
            log_image = obs_img[0].cpu().numpy()
            log_action = post_action[0].cpu().numpy().tolist()
            log_info = {k: v[0].tolist() for k, v in env_info.items() if k != "episode"}

            data["image"].append(log_image)
            data["action"].append(log_action)
            data["info"].append(log_info)

            obs_img = obs_img_new

        data["image"].append(obs_img[0].cpu().numpy())

        exp_dir = Path(self.glob_dir) / f"eval_{epoch}_{obj_set}_{instruction[0]}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        if self.args.render_info:
            for j in range(len(data["info"])):
                data["image"][j + 1] = visualization.put_info_on_image(
                    data["image"][j + 1],
                    data["info"][j],
                    extras=[f"Ins: {data['instruction']}"]
                )

        success = int(data["info"][-1]["success"])
        images_to_video(
            data["image"],
            str(exp_dir),
            f"video_{object[0]}_{receptacle[0]}-s_{success}",
            fps=10,
            verbose=False
        )

        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats_ret = env_stats.copy()


        save_stats = {
            "env_name": self.args.env_id,
            "ep_len": self.args.episode_len,
            "epoch": epoch,
            "stats": {k: float(v) for k, v in env_stats.items()},
            "instruction": data["instruction"],
            "last_info": data["info"][-1],
        }
        yaml.dump(save_stats, open(exp_dir / "stats.yaml", "w"))

        return env_stats_ret

    @torch.no_grad()
    def render(self, epoch: int, obj_set: str, object: list[str], receptacle: list[str]) -> dict:
        self.policy.prep_rollout()

        # init logger
        env_infos = defaultdict(lambda: [])
        datas = [{
            "image": [],  # obs_t: [0, T-1]
            "instruction": "",
            "action": [],  # a_t: [0, T-1]
            "info": [],  # info after executing a_t: [1, T]
        } for idx in range(self.args.num_envs)]

        obs_img, instruction, info, _ = self.env.reset(obj_set=obj_set, same_init=self.args.use_same_init, object=object, receptacle=receptacle)
        print("Rendering:", instruction[0])

        # data dump: instruction
        for idx in range(self.args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img_new, reward, done, env_info = self.env.step(action)

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

            for i in range(self.args.num_envs):
                post_action = self.env._process_action(action)
                log_image = obs_img[i].cpu().numpy()
                log_action = post_action[i].cpu().numpy().tolist()
                log_info = {k: v[i].tolist() for k, v in env_info.items() if k != "episode"}
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)

            # update obs_img
            obs_img = obs_img_new

        # data dump: last image
        for i in range(self.args.num_envs):
            log_image = obs_img[i].cpu().numpy()
            datas[i]["image"].append(log_image)

        # save video
        exp_dir = Path(self.glob_dir) / f"vis_{epoch}_{obj_set}"

        print("exp_dir : ", exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.args.num_envs):
            images = datas[i]["image"]
            infos = datas[i]["info"]
            assert len(images) == len(infos) + 1

            if self.args.render_info:
                for j in range(len(infos)):
                    images[j + 1] = visualization.put_info_on_image(
                        images[j + 1], infos[j],
                        extras=[f"Ins: {instruction[i]}"]
                    )

            success = int(infos[-1]["success"])
            images_to_video(images, str(exp_dir), f"video_{i}-{object[0]}_{receptacle[0]}-s_{success}",
                            fps=10, verbose=False)

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats_ret = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        # save stats
        last_info = {
            idx: {k: env_infos[k][idx] for k in env_infos.keys()}
            for idx in range(self.args.num_envs)
        }

        save_stats = {}
        save_stats["env_name"] = self.args.env_id
        save_stats["ep_len"] = self.args.episode_len
        save_stats["epoch"] = epoch
        save_stats["stats"] = {k: v.item() for k, v in env_stats.items()}
        save_stats["instruction"] = {idx: ins for idx, ins in enumerate(instruction)}
        save_stats["last_info"] = last_info

        yaml.dump(save_stats, open(exp_dir / "stats.yaml", "w"))

        return env_stats_ret

    def run(self):
        max_episodes = self.args.steps_max // self.args.episode_len // self.args.num_envs
        max_episodes = 100
        instruction_switch_interval = 80
        steps = 0

        num_envs = self.args.num_envs 
        group_size = num_envs // 4    

        for episode in range(max_episodes):
            env_infos = defaultdict(lambda: [])
            ep_time = time.time()

            # ***** [ADD] COSTMAP-SWITCH init *****
            subtask_id           = [0] * 64          # 目前 costmap 的索引
            current_costmap_id   = [-1] * 64         # 用來偵測是否已切換
            grasp_success_count  = [0] * 64          # 若 costmap 指定 grasp-based switch
            best_cost            = [1e9] * 64        # 用來記錄該 step 的最佳 cost
            # =========================================

            objects, receptacles = [], []
            # test
            # for i in range(4):
            #     obj, recep = self.extract_obj_recep(self.task_list[(self.task_id + i) % len(self.task_list)])
            #     objects.extend([obj] * group_size)
            #     receptacles.extend([recep] * group_size)

            self.task_id = 2
            obj, recep = self.extract_obj_recep(self.task_list[self.task_id])
            objects.extend([obj] * 64)
            receptacles.extend([recep] * 64)
            ### 


            costmap_dir = f"costmap/rlvla/costmaps2"
            env_reset_options = {
                "obj_set": "train",
                "same_init": self.args.use_same_init,
                "object": objects,
                "receptacle": receptacles
            }

            costmap_handler_list = initialize_cost_map(
                self.env, env_reset_options, costmap_dir, "3rd_view_camera", 4
            )

            obs_img, instruction, info, _ = self.env.reset(
                obj_set="train",
                same_init=self.args.use_same_init,
                object=objects,
                receptacle=receptacles
            )

            task_id_map = []
            # test
            # for i in range(4):
            #     task_id_map.extend([(self.task_id + i) % len(self.task_list)] * group_size)
            task_id_map.extend([self.task_id] * 64)
            ### 

            self.buffer.warmup(obs_img.cpu().numpy(), instruction)

            
            # obj, recep = self.extract_obj_recep(self.task_list[self.task_id])
            # obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init, object=[obj]*self.args.num_envs, receptacle=[recep]*self.args.num_envs)
            # self.buffer.warmup(obs_img.cpu().numpy(), instruction)
            rollout_images = [[] for _ in range(self.args.num_envs)]

            print("instruction 0 : ", instruction[0])
            print("instruction 16 : ", instruction[16])
            print("instruction 32 : ", instruction[32])
            print("instruction 48 : ", instruction[48])

            # import ipdb; ipdb.set_trace()



            cost_map_info = []
            viz_writers = []
            for i in range(self.args.num_envs):
                viz_path = f"videos_with_costmap/{episode}/costmap_vis_{i:04d}.mp4"
                os.makedirs(os.path.dirname(viz_path), exist_ok=True)
                viz_writer = imageio.get_writer(viz_path, fps=10, codec="libx264")
                viz_writers.append(viz_writer)

            for step_idx in tqdm(range(self.args.training_len), desc="rollout"):
                costmap_handler = []
                # test
                for i in  range(self.args.num_envs):
                #     costmap_handler.append(costmap_handler_list[(self.task_id + (i // group_size)) % len(self.task_list)])
                    costmap_handler.append(costmap_handler_list[self.task_id])
                ###
                for i in range(self.args.num_envs):
                    if len(cost_map_info) != self.args.num_envs:
                        cost_map_info.append(costmap_handler[i].get_costmap_info(subtask_id[i]))
                        current_costmap_id[i] = subtask_id[i]
                    elif subtask_id[i] != current_costmap_id[i]:
                        cost_map_info[i] = costmap_handler[i].get_costmap_info(subtask_id[i])
                        current_costmap_id[i] = subtask_id[i]

                value, action, logprob, best_cost = self.collect_costmap(envs=self.env, costmap_handler=costmap_handler, subtask_id=subtask_id, viz_writers=viz_writers)
                # value, action, logprob = self.collect()
                obs_img, reward, done, env_info = self.env.step(action)
                for idx in range(self.args.num_envs):
                    if env_info["success"][idx]:
                        print(f"[{idx}] success!!!!!!")
                        # with open("/workspace/AutoRL_SD/log.txt", "a") as f:
                        #     f.write("[{idx}] success!\n")
                for env_i in range(self.args.num_envs):
                    rollout_images[env_i].append(obs_img[env_i].cpu().numpy())

                # =========================================
                # ***** [ADD] COSTMAP-SWITCH decision *****
                for idx in range(num_envs):
                    if cost_map_info[idx] is not None:
                        # grasp-based or cost-based
                        if cost_map_info[idx]["grasp_subtask_switch"]:
                            if self.env.env.unwrapped.agent.is_grasping(self.env.env.unwrapped.objs[idx][self.env.env.unwrapped.source_obj_name[idx]])[idx]:
                                grasp_success_count[idx] += 1
                            else:
                                grasp_success_count[idx]  = 0
                            if grasp_success_count[idx] >= 5 and subtask_id[idx] < len(costmap_handler[idx].costmaps) - 1:
                                # print(f"[DEBUG][{idx}] grasp success x10 → switch subtask")
                                subtask_id[idx]          += 1
                                grasp_success_count[idx]  = 0
                        else:
                            thresh = 10 if cost_map_info[idx]["affordance_types"] == "motion" else 0.03
                            # print("idx :", idx, "subtask :", subtask_id[idx], "best_cost :", best_cost[idx], "thresh :", thresh)
                            if best_cost[idx] <= thresh and subtask_id[idx] < len(costmap_handler[idx].costmaps) - 1:
                                # print(f"[DEBUG][{idx}] cost ({best_cost[idx]}) ≤ {thresh:.3f} → switch to subtask {subtask_id[idx] + 1}")
                                subtask_id[idx] += 1
                # =========================================


                data = (obs_img, action, logprob, value, reward, done)
                self.insert(data)

                

                # info
                if "episode" in env_info.keys():
                    for k, v in env_info["episode"].items():
                        env_infos[f"{k}"] += v

                if (step_idx+1) % instruction_switch_interval == 0 and step_idx > 0:
                    
                    self.compute_endup()

                    render_dir = Path(self.glob_dir) / f"rollout_ep{episode}_step{step_idx}"
                    render_dir.mkdir(parents=True, exist_ok=True)

                    for env_i in range(self.args.num_envs):
                        images = rollout_images[env_i]
                        images_to_video(images, str(render_dir), f"env{env_i}", fps=10, verbose=False)

                    rollout_images = [[] for _ in range(self.args.num_envs)]

                    del value, action, logprob, reward, done
                    gc.collect()
                    torch.cuda.empty_cache()

                    # train
                    infos = self.train(episode)
                    for k, v in env_infos.items():
                        infos[f"env/{k}"] = np.mean(v)
                    # wandb.log(infos, step=step_idx + episode * self.args.training_len)
                    self.buffer.warmup(obs_img.cpu().numpy(), instruction)

                    # Switch Instruction
                    # test 
                    # self.task_id = (self.task_id + 1) % len(self.task_list)
                    # objects, receptacles = [], []
                    # for i in range(4):
                    #     obj, recep = self.extract_obj_recep(self.task_list[(self.task_id + i) % len(self.task_list)])
                    #     objects.extend([obj] * group_size)
                    #     receptacles.extend([recep] * group_size)
                    # self.env.set_task(objects, receptacles)
                    objects, receptacles = [], []
                    for i in range(4):
                        obj, recep = self.extract_obj_recep(self.task_list[self.task_id])
                        objects.extend([obj] * 64)
                        receptacles.extend([recep] * 64)
                    self.env.set_task(objects, receptacles)
                    ###

                    # obj, recep = self.extract_obj_recep(self.task_list[self.task_id])
                    # self.env.set_task([obj]*self.args.num_envs, [recep]*self.args.num_envs)
                    instruction = self.env.get_language_instruction()
                    print(step_idx, "switch instruction to ", instruction[0], instruction[16], instruction[32], instruction[48])
                    self.buffer.update_instruction(instruction)

            # steps
            steps = (episode + 1) * self.args.training_len * self.args.num_envs
            # print(pprint.pformat({k: round(np.mean(v), 4) for k, v in env_infos.items()}))

            for i in range(self.args.num_envs):
                viz_writers[i].close()

            # eval
            if episode % self.args.interval_eval == self.args.interval_eval - 1 or episode == max_episodes - 1:
                print(f"Evaluating at {steps}")
                a = 0
                for object in self.env.get_object_names()[0]:
                    for receptacle in self.env.get_receptacle_names()[0]:
                        if a == 3:
                            sval_stats = self.eval("train", [object]*self.args.num_envs, [receptacle]*self.args.num_envs)
                            sval_stats = {f"eval＿put_{object}_in_{receptacle}/{k}": v for k, v in sval_stats.items()}
                            wandb.log(sval_stats, step=steps)
                        a += 1

            # save
            if episode % self.args.interval_save == self.args.interval_save - 1 or episode == max_episodes - 1:
                print(f"Saving model at {steps}")
                save_path = self.glob_dir / f"steps_{episode:0>4d}"
                self.policy.save(save_path)
                
                for object in self.env.get_object_names()[0]:
                    for receptacle in self.env.get_receptacle_names()[0]:
                        self.render(episode, "train", [object]*self.args.num_envs, [receptacle]*self.args.num_envs)



def main():
    args = tyro.cli(Args)
    runner = Runner(args)

    if args.only_render:
        ll = [
            "OneObjectTwoReceptacle-v1",
            "TwoObjectOneReceptacle-v1",
            "TwoObjectTwoReceptacle-v1"
        ]
        if args.env_id not in ll:
            runner.render(epoch=0, obj_set="train")
        for object in runner.env.get_object_names()[0]:
            for receptacle in runner.env.get_receptacle_names()[0]:
                runner.render(0, "train", [object]*runner.args.num_envs, [receptacle]*runner.args.num_envs)

    else:
        runner.run()


if __name__ == "__main__":
    main()
