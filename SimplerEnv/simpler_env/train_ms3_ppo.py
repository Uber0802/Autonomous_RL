import os
import re
import pprint
import random
import gc
import signal
import shutil 
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
import itertools
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video

from simpler_env.env.simpler_wrapper import SimlerWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer, PreallocReplayBuffer, FIFOReplayBuffer
import copy

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
    log: str = "/workspace/Autonomous_RL/two_buffer_160_seed2.txt"

    # Environment
    obj_set: str = "rand"   # "fixed" "rand" "rand_ood" "rand_8"
    obj1_index: int = 7     # ketchup bottle
    obj2_index: int = 2     # shovel
    plate1_index: int = 1   # plate
    plate2_index: int = 2   # cloth

    # training
    num_envs: int = 64
    episode_len: int = 80
    training_len: int = 320
    instruction_switch_interval: int = 80
    training_interval: int = 160
    max_episodes: int = 32
    max_reset: int = 8192

    # evaluaiton
    interval_eval: int = 4
    interval_save: int = 8
    eval_at_start: bool = False

    # forward backward
    enable_backward: bool = False
    backward_interval: int = 1

    # reset unsuitable
    reset_unsuitable: bool = False

    # reset robot
    reset_robot: bool = True

    # random task order
    random_task_order: bool = False

    few_position: bool = False

    # other env settings
    use_same_init: bool = True
    steps_max: int = 200000000
    steps_vh: int = 0  # episodes

    # buffer
    fifo_buffer: bool = False
    fifo_length: int = 1024
    buffer_inferbatch: int = 32
    buffer_minibatch: int = 8
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95

    # vla
    vla_path: str = "openvla/openvla-7b"
    vla_unnorm_key: str = "bridge_orig"
    vla_load_path: str = ""
    bc_init_path: str = ""  # Path to BC-pretrained Gaussian head (bc_init.pt)
    vla_lora_rank: int = 32

    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6

    # ppo & grpo
    alg_name: str = "ppo"  # ppo, grpo
    alg_grpo_fix: bool = True
    alg_gradient_accum: int = 20
    alg_ppo_epoch: int = 1
    alg_entropy_coef: float = 0.0
    lambda_bc: float = 0.1

    # policy type
    policy_type: str = "openvla"  # "openvla" or "cogact"

    # other
    wandb: bool = True
    wandb_dir: str = ""
    only_render: bool = False
    only_render_seq: bool = False
    render_info: bool = False

class Runner:
    def __init__(self, all_args: Args):
        self.args = all_args

        # alg_name
        assert self.args.alg_name in ["ppo", "grpo"]
        
        # training lengths
        assert self.args.training_len % self.args.episode_len == 0, "training_len must be divisible by episode_len"
        assert self.args.instruction_switch_interval % self.args.episode_len == 0, "instruction_switch_interval must be divisible by episode_len"
        assert self.args.training_interval % self.args.episode_len == 0, "training_interval must be divisible by episode_len"

        # set seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # set wandb
        wandb.init(
            config=all_args.__dict__,
            project="RLVLA",
            name=self.args.name,
            dir=self.args.wandb_dir,
            mode="online" if self.args.wandb else "offline",
        )
        self.save_dir = Path(wandb.run.dir)
        self.glob_dir = Path(wandb.run.dir) / ".." / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)

        yaml.dump(all_args.__dict__, open(self.glob_dir / "config.yaml", "w"))

        # policy
        device_id = 0
        device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device("cuda:" + str(device_id))
        self.use_cogact = (self.args.policy_type == "cogact")

        if self.use_cogact:
            from simpler_env.policies.cogact.cogact_train import CogACTPolicy, CogACTPPO
            self.policy = CogACTPolicy(all_args, device_id_other)
            self.alg = CogACTPPO(all_args, self.policy)
            unnorm_state = self.policy.cogact.get_action_stats(self.args.vla_unnorm_key)
        else:
            from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
            self.policy = OpenVLAPolicy(all_args, device_id_other)
            self.alg = OpenVLAPPO(all_args, self.policy)
            unnorm_state = self.policy.vla.get_action_stats(self.args.vla_unnorm_key)

        # env
        self.env = SimlerWrapper(self.args, unnorm_state, continuous_actions=self.use_cogact)

        # buffer: CogACT uses float32 actions + scalar logprobs; OpenVLA uses int32 + per-token logprobs
        action_dtype = np.float32 if self.use_cogact else np.int32
        logprob_dim = 1 if self.use_cogact else 7

        if self.args.fifo_buffer:
            assert self.args.fifo_length % self.args.num_envs == 0, "fifo_length must be divisible by num_envs"
            self.prealloc_buffer = FIFOReplayBuffer(
                all_args,
                obs_dim=(480, 640, 3),
                act_dim=7,
                max_num_envs=self.args.fifo_length,
                action_dtype=action_dtype,
                logprob_dim=logprob_dim,
            )
        else:
            self.prealloc_buffer = PreallocReplayBuffer(
                all_args,
                obs_dim=(480, 640, 3),
                act_dim=7,
                action_dtype=action_dtype,
                logprob_dim=logprob_dim,
            )
        self.buffer = SeparatedReplayBuffer(
            all_args,
            obs_dim=(480, 640, 3),
            act_dim=7,
            action_dtype=action_dtype,
            logprob_dim=logprob_dim,
        )
        minibatch_count = self.buffer.get_minibatch_count()
        print(f"Buffer minibatch count: {minibatch_count}")

        # Task Switch
        self.task_id = 0
        self.task_list = self.env.get_task_pool()[0]

        # Reset Counter
        self.exceed_reset_limit = False
        self.hard_reset_count = 0   # Fixed reset at the start of each episode
        self.soft_reset_count = 0   # Reset unsuitable envs

        if self.args.few_position:
            self.args.obj_set = "fixed"

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

    def collect(self):
        self.policy.prep_rollout()

        obs_image = self.buffer.obs[self.buffer.step]
        obs_image = torch.tensor(obs_image).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        value, action, logprob = self._get_action(obs)

        return value, action, logprob

    def insert(self, data):
        obs_img, actions, logprob, value_preds, rewards, done = data
        masks = 1.0 - done.to(torch.float32)

        obs_img = obs_img.cpu().numpy()
        # CogACT: actions are float32 continuous; OpenVLA: actions are int32 tokens
        if self.use_cogact:
            actions = actions.to(torch.float32).cpu().numpy()
        else:
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

    def train(self):
        self.policy.prep_training()
        info = None
        train_info = None

        if self.args.alg_name == "ppo":
            train_info = self.alg.train_ppo(self.prealloc_buffer)

        elif self.args.alg_name == "grpo":
            train_info = self.alg.train_grpo(self.buffer)
        else:
            raise ValueError(f"Unknown alg_name: {self.args.alg_name}")
        if train_info:
            info = {f"train/{k}": v for k, v in train_info.items()}
            info["buffer/reward_mean"] = np.mean(self.buffer.rewards)
            info["buffer/mask_mean"] = np.mean(1.0 - self.buffer.masks)

        return info

    @torch.no_grad()
    def eval(self, obj_set: str, object: list[str], receptacle: list[str]) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])

        obs_img, instruction, info = self.env.reset(obj_set=obj_set, same_init=self.args.use_same_init, object=object, receptacle=receptacle)
        print("Evaluating:", instruction[0])

        for _ in tqdm(range(self.args.episode_len), desc="eval"):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img, reward, done, env_info = self.env.step(action)

            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        return env_stats

    @torch.no_grad()
    def eval_video(self, epoch: int, obj_set: str, object: list[str], receptacle: list[str]) -> dict:
        self.policy.prep_rollout()

        env_infos = defaultdict(lambda: [])
        obs_img, instruction, info = self.env.reset(obj_set=obj_set, same_init=self.args.use_same_init, object=object,receptacle=receptacle)

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
    def render(self, epoch: int, obj_set: str, object: list[str], receptacle: list[str], reset=True, exp_dir=None) -> dict:
        if reset:
            self.policy.prep_rollout()

        # init logger
        env_infos = defaultdict(lambda: [])
        datas = [{
            "image": [],  # obs_t: [0, T-1]
            "instruction": "",
            "action": [],  # a_t: [0, T-1]
            "info": [],  # info after executing a_t: [1, T]
        } for idx in range(self.args.num_envs)]
        if reset:
            obs_img, instruction, info = self.env.reset(obj_set=obj_set, same_init=self.args.use_same_init, object=object, receptacle=receptacle)
        else:
            self.env.set_task(object, receptacle)
            obs_img = self.env.get_obs_image()
            instruction = self.env.get_language_instruction()
        print("Rendering:", instruction[0])

        # data dump: instruction
        for idx in range(self.args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        for idx in tqdm(range(self.args.episode_len), desc=f"Rendering {instruction[0]}"):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img_new, reward, done, env_info = self.env.step(action)

            # info
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
        if exp_dir:
            exp_dir = Path(self.glob_dir) / exp_dir
        else:
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

    @torch.no_grad()
    def render_seq(self, obj_set: str, n = 4, eval_training_seq=True):
        perms = list(itertools.permutations(self.task_list))
        training_seq = perms.pop(0)
        selected = random.sample(perms, n)
        if eval_training_seq:
            selected = [training_seq] + selected
        
        for i, task_lst in enumerate(tqdm(selected)):
            print(task_lst)           
            for j, task in enumerate(task_lst):
                object, receptacle = self.extract_obj_recep(task)
                if j == 0:
                    reset = True
                else:
                    reset = False
                stats = self.render(0, obj_set, [object]*self.args.num_envs, [receptacle]*self.args.num_envs, reset, f"{i}-{j}")

    def run(self):

        max_episodes = self.args.steps_max // self.args.episode_len // self.args.num_envs
        max_episodes = self.args.max_episodes
        instruction_switch_interval = self.args.instruction_switch_interval
        steps = 0
        if self.args.eval_at_start:
            print(f"Evaluating at {steps}")
            for task in self.task_list:
                object, receptacle = self.extract_obj_recep(task)
                sval_stats = self.eval(self.args.obj_set, [object]*self.args.num_envs, [receptacle]*self.args.num_envs)
                sval_stats = {f"eval_ood＿put_{object}_in_{receptacle}/{k}": v for k, v in sval_stats.items()}
                wandb.log(sval_stats, step=steps)

        num_envs = self.args.num_envs 
        group_size = num_envs // 4   

        for episode in range(max_episodes):
            self.env.set_forward()
            forward_count = 1
            env_infos = defaultdict(lambda: [])
            ep_time = time.time()
            self.prealloc_buffer.reset()

            objects, receptacles = [], []
            if not self.args.random_task_order:
                for i in range(4):
                    obj, recep = self.extract_obj_recep(self.task_list[(self.task_id + i) % len(self.task_list)])
                    objects.extend([obj] * group_size)
                    receptacles.extend([recep] * group_size)
            else:
                for i in range(self.args.num_envs):
                    rand = random.randint(0, 3)
                    obj, recep = self.extract_obj_recep(self.task_list[(self.task_id + rand) % len(self.task_list)])
                    objects.extend([obj])
                    receptacles.extend([recep])

            obs_img, instruction, info = self.env.reset(
                obj_set=self.args.obj_set,
                same_init=self.args.use_same_init,
                object=objects,
                receptacle=receptacles
            )
            self.hard_reset_count += self.args.num_envs
            print(f"Total reset: {self.hard_reset_count + self.soft_reset_count}")

            task_id_map = []
            for i in range(4):
                task_id_map.extend([(self.task_id + i) % len(self.task_list)] * group_size)

            self.buffer.warmup(obs_img.cpu().numpy(), instruction)
            rollout_images = [[] for _ in range(self.args.num_envs)]
            print("instruction : ", instruction[0], instruction[group_size], instruction[group_size*2], instruction[group_size*3])
            with open(self.args.log, "a") as f:
                f.write(f"step : {steps}\n")
            # Reset buffer for 0-79 step    
            for step_idx in tqdm(range(self.args.training_len), desc="rollout"):
                value, action, logprob = self.collect()
                obs_img, reward, done, env_info = self.env.step(action)
                for env_i in range(self.args.num_envs):
                    rollout_images[env_i].append(obs_img[env_i].cpu().numpy())
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
                    self.prealloc_buffer.cat_buffer(self.buffer)
                    if (step_idx+1) % self.args.training_interval == 0 and step_idx > 0:
                        infos = self.train()
                        self.prealloc_buffer.reset()

                    #Switch Instruction
                    if self.args.enable_backward and forward_count >= self.args.backward_interval:
                        self.env.set_backward()
                        forward_count = 0
                    else:
                        self.task_id = (self.task_id + 1) % len(self.task_list)
                        objects, receptacles = [], []
                        if not self.args.random_task_order:
                            for i in range(4):
                                obj, recep = self.extract_obj_recep(self.task_list[(self.task_id + i) % len(self.task_list)])
                                objects.extend([obj] * group_size)
                                receptacles.extend([recep] * group_size)
                        else:
                            for i in range(self.args.num_envs):
                                rand = random.randint(0, 3)
                                obj, recep = self.extract_obj_recep(self.task_list[(self.task_id + rand) % len(self.task_list)])
                                objects.extend([obj])
                                receptacles.extend([recep])
                        self.env.set_forward()
                        self.env.set_task(objects, receptacles)
                        forward_count += 1
                        
                    instruction = self.env.get_language_instruction()
                    print(step_idx, "switch instruction to ", instruction[0], instruction[group_size], instruction[group_size*2], instruction[group_size*3])
                    if self.args.reset_unsuitable:
                        obs_img = self.env.reset_unsuitable_envs()
                        self.soft_reset_count = self.env.reset_unsuitable_envs_count
                        print(f"Total unsuitable envs reset: {self.soft_reset_count}")
                        print(f"Total reset: {self.hard_reset_count + self.soft_reset_count}")
                    if self.args.reset_robot:
                        obs_img = self.env.reset_robot()
                    # Check if exceed total resets
                    if self.hard_reset_count + self.soft_reset_count > self.args.max_reset:
                        self.exceed_reset_limit = True
                        break
                    print("Rollouts in Unsuitable Envs: ", self.env.get_unsuitable_envs())
                    print("Total Rollouts in Unsuitable Envs: ", self.env.unsuitable_envs)
                    self.buffer.warmup(obs_img.cpu().numpy(), instruction)
                    self.buffer.update_instruction(instruction)

            # steps
            steps = (episode + 1) * self.args.training_len * self.args.num_envs
            # Check if exceed total resets
            if self.hard_reset_count + self.soft_reset_count > self.args.max_reset - self.args.num_envs:
                self.exceed_reset_limit = True
           
            # eval
            if episode % self.args.interval_eval == self.args.interval_eval - 1 or episode == max_episodes - 1 or self.exceed_reset_limit:
                print(f"Evaluating Random Scene at {steps}")
                for task in self.task_list:
                    object, receptacle = self.extract_obj_recep(task)
                    sval_stats = self.eval(self.args.obj_set, [object]*self.args.num_envs, [receptacle]*self.args.num_envs)
                    sval_stats = {f"eval_ood＿put_{object}_in_{receptacle}/{k}": v for k, v in sval_stats.items()}
                    wandb.log(sval_stats, step=steps)
                print(f"Evaluating Train Scene at {steps}")
                for task in self.task_list:
                    object, receptacle = self.extract_obj_recep(task)
                    sval_stats = self.eval("rand_ood", [object]*self.args.num_envs, [receptacle]*self.args.num_envs)
                    sval_stats = {f"eval_true_ood_put_{object}_in_{receptacle}/{k}": v for k, v in sval_stats.items()}
                    wandb.log(sval_stats, step=steps)

            # save
            if episode % self.args.interval_save == self.args.interval_save - 1 or episode == max_episodes - 1 or self.exceed_reset_limit:
                print(f"Saving model at {steps}")
                save_path = self.glob_dir / f"steps_{episode:0>4d}"
                self.policy.save(save_path)
                
                for task in self.task_list:
                    object, receptacle = self.extract_obj_recep(task)
                    self.render(episode, self.args.obj_set, [object]*self.args.num_envs, [receptacle]*self.args.num_envs)

            if self.exceed_reset_limit:
                print(f"Total reset: {self.hard_reset_count + self.soft_reset_count} exceed reset limit: {self.args.max_reset}")
                break
                
            print(f"Total reset: {self.hard_reset_count + self.soft_reset_count}")

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
            runner.render(epoch=0, obj_set=args.obj_set)
        for idx, task in enumerate(runner.task_list):
            object, receptacle = runner.extract_obj_recep(task)
            runner.render(0, args.obj_set, [object]*runner.args.num_envs, [receptacle]*runner.args.num_envs, exp_dir=f"0-{idx}")

    elif args.only_render_seq:
        runner.render_seq(args.obj_set, 4, True)
    
    else:
        runner.run()

    # Remove Directory for buffer memmap data
    pid = os.getpid()
    pid_dir = os.path.join(os.getcwd(), str(pid))
    if os.path.exists(pid_dir):
            shutil.rmtree(pid_dir)


if __name__ == "__main__":
    main()
