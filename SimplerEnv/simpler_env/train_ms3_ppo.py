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
from dataclasses import dataclass, replace
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
# from mani_skill.utils.visualization.misc import images_to_video

from simpler_env.env.simpler_wrapper import SimlerWrapper, SDSimlerWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer

from split_decisions.utils.sampling_utils import costmap_guided_sampling, initialize_cost_map, encode_actions_from_norm_openvla, reconstruct_action_from_norm_openvla
from split_decisions.utils.action_utils import get_pose_base, get_pose_world
from split_decisions.prismatic.vla.action_tokenizer import ActionTokenizer
from split_decisions.utils.observation_utils import world_to_screen, world_to_screen_idx
import imageio
import cv2


import os
import re
from pathlib import Path
import numpy as np
import cv2

def _pad_to_even(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return img

def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)

def _prep_frame(im: np.ndarray, size_hw):
    H, W = size_hw
    im = np.asarray(im)
    if im.ndim == 2:  # gray -> 3ch
        im = np.repeat(im[..., None], 3, axis=2)
    if im.shape[2] == 4:  # RGBA -> RGB
        im = im[..., :3]
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255).astype(np.uint8)
    im = _pad_to_even(im)
    if im.shape[0] != H or im.shape[1] != W:
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(im)

def _validate_video(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    ok, fr = cap.read()
    cap.release()
    return bool(ok) and (fr is not None)

def _try_opencv_mp4(frames, path: Path, fps: int, fourcc_code: str, size_hw):
    H, W = size_hw
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    if not writer.isOpened():
        return False
    try:
        # 第一張
        f0 = _prep_frame(frames[0], size_hw)
        writer.write(cv2.cvtColor(f0, cv2.COLOR_RGB2BGR))
        # 其餘
        for im in frames[1:]:
            im = _prep_frame(im, size_hw)
            writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return _validate_video(path)

def _try_imageio_mp4(frames, path: Path, fps: int, size_hw):
    try:
        import imageio
    except Exception:
        return False
    H, W = size_hw
    try:
        writer = imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    except Exception:
        return False

    try:
        for im in frames:
            im = _prep_frame(im, size_hw)
            # imageio 走的是 RGB，不用轉 BGR
            writer.append_data(im)
    finally:
        writer.close()
    return _validate_video(path)

def images_to_video(frames, out_dir, filename, fps=10, verbose=False):
    """
    可靠 MP4 寫入：
    1) OpenCV: mp4v -> avc1 -> H264 -> X264
    2) 失敗則用 imageio-ffmpeg(libx264, yuv420p)
    成功會回傳 mp4 檔；若都不可用才會拋錯。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        return

    filename = _sanitize_filename(filename)

    # 以第一幀決定輸出尺寸（處理 dtype/通道/偶數尺寸）
    f0 = np.asarray(frames[0])
    if f0.ndim == 2:
        f0 = np.repeat(f0[..., None], 3, axis=2)
    if f0.shape[2] == 4:
        f0 = f0[..., :3]
    if f0.dtype != np.uint8:
        f0 = np.clip(f0, 0, 255).astype(np.uint8)
    f0 = _pad_to_even(f0)
    H, W = f0.shape[:2]
    size_hw = (H, W)

    out_path = out_dir / f"{filename}.mp4"

    # 1) 先試 OpenCV 的 MP4 編碼
    fourcc_candidates = ["mp4v", "avc1", "H264", "X264"]
    for code in fourcc_candidates:
        if verbose:
            print(f"[images_to_video] Try OpenCV fourcc={code} → {out_path.name}")
        try:
            if _try_opencv_mp4(frames, out_path, int(fps), code, size_hw):
                if verbose:
                    print(f"[images_to_video] OpenCV fourcc={code} OK")
                return
        except Exception as e:
            if verbose:
                print(f"[images_to_video] OpenCV {code} failed: {e}")

    # 2) 再試 imageio-ffmpeg (libx264)
    if verbose:
        print(f"[images_to_video] Fallback to imageio-ffmpeg(libx264) → {out_path.name}")
    ok = False
    try:
        ok = _try_imageio_mp4(frames, out_path, int(fps), size_hw)
    except Exception as e:
        if verbose:
            print(f"[images_to_video] imageio-ffmpeg failed: {e}")

    if ok:
        if verbose:
            print("[images_to_video] imageio-ffmpeg OK")
        return

    # 如果你真的一定要 mp4，這裡選擇拋出具體錯誤（比產生壞檔好）
    raise RuntimeError(
        "Failed to write MP4 with both OpenCV and imageio-ffmpeg. "
        "Please ensure at least one of: (a) OpenCV built with MP4 encoders "
        "(mp4v/avc1/H264), or (b) `pip install imageio-ffmpeg` so ffmpeg is available."
    )


signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _augment_uint8_batch(imgs_np: np.ndarray, *,
                         prob: float, brightness: float, contrast: float,
                         noise_std: float, cutout_p: float, cutout_ratio: float) -> np.ndarray:
    """
    imgs_np: [B,H,W,C] uint8
    回傳同型別，僅做輕量標籤不變增強（不做水平翻轉/旋轉，以免需同步變換 action）。
    """
    if imgs_np.size == 0:
        return imgs_np
    t = torch.from_numpy(imgs_np.astype(np.float32))  # [0,255]
    B, H, W, C = t.shape
    sel = torch.rand(B) < prob
    if sel.any():
        x = t[sel]  # 選到的那些影像
        # 亮度 (乘法)
        if brightness > 0:
            bf = 1.0 + (torch.rand(x.shape[0], 1, 1, 1) * 2 - 1) * brightness
            x = x * bf
        # 對比 (乘法，繞 mean)
        if contrast > 0:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            cf = 1.0 + (torch.rand(x.shape[0], 1, 1, 1) * 2 - 1) * contrast
            x = (x - mean) * cf + mean
        # 高斯雜訊
        if noise_std > 0:
            std = noise_std * 255.0
            x = x + torch.randn_like(x) * std
        # Cutout（小遮擋）
        if cutout_p > 0 and cutout_ratio > 0:
            ch = max(1, int(H * cutout_ratio))
            cw = max(1, int(W * cutout_ratio))
            for i in range(x.shape[0]):
                if torch.rand(()) < cutout_p:
                    y0 = torch.randint(0, max(1, H - ch + 1), (1,)).item()
                    x0 = torch.randint(0, max(1, W - cw + 1), (1,)).item()
                    x[i, y0:y0+ch, x0:x0+cw, :] = 0
        t[sel] = x
    t = t.clamp(0, 255).to(torch.uint8)
    return t.numpy()



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
    num_envs: int = 32

    episode_len: int = 80 # 80
    training_len: int = 80
    use_same_init: bool = True

    steps_max: int = 99000000
    steps_vh: int = 0  # episodes
    interval_eval: int = 1
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
    vla_temperature_eval: float = 0

    # ppo & grpo
    alg_name: str = "ppo"  # ppo, grpo
    alg_grpo_fix: bool = True
    alg_gradient_accum: int = 80
    alg_ppo_epoch: int = 15
    alg_entropy_coef: float = 0

    # other
    wandb: bool = True
    only_render: bool = False
    render_info: bool = False


    # augmentation
    aug_enable_sft: bool = False
    aug_enable_ppo: bool = True
    aug_prob: float = 0.8           # 每張圖做增強的機率
    aug_brightness: float = 0    # 乘法亮度抖動幅度 (±15%)
    aug_contrast: float = 0      # 乘法對比抖動幅度 (±15%)
    aug_noise_std: float = 0.01     # 高斯雜訊強度，對應 0.01*255
    aug_cutout_p: float = 0      # 以此機率對圖片做一次 cutout
    aug_cutout_ratio: float = 0  # cutout 方塊邊長佔 H/W 的比例



class Runner:
    def __init__(self, all_args: Args):
        self.args = all_args
        self.args.seed = 2

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
        def pick_device(prefer: int = 0):
            if not torch.cuda.is_available():
                return torch.device("cpu")
            n = torch.cuda.device_count()
            if prefer < 0 or prefer >= n:
                print(f"[WARN] Requested GPU {prefer} is unavailable")
                prefer = 0
            torch.cuda.set_device(prefer)
            return torch.device(f"cuda:{prefer}")

        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
        requested = 2  # 想優先用第 3 張卡就寫 2；不在就自動回 0
        self.device = pick_device(requested)
        device_id_other = self.device.index if self.device.type == "cuda" else -1
        self.policy = OpenVLAPolicy(all_args, device_id_other)

        self.alg = OpenVLAPPO(all_args, self.policy)

        # env
        unnorm_state = self.policy.vla.get_action_stats(self.args.vla_unnorm_key)
        self.unnorm_state = unnorm_state
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

        self._traj_cache = [[] for _ in range(self.args.num_envs)]
        self._traj_has_success = [False] * self.args.num_envs
        self.success_buffer = []

        self.video_env_ids = self._compute_video_env_ids(k=32)  # 例如 64 -> [0, 16, 32, 48]
        self.eval_idx = 0

        self.did_sft = False          # 是否已經做過一次 SFT
        self.sft_min_trajs = 5        # 需要的成功軌跡數量門檻


        self.successed = [False] * self.args.num_envs 

    def _make_eval_env(self, num_eval_envs: int = 1):
        # 用目前 args 的其他設定，但把 num_envs 換成 1
        eval_args = replace(self.args, num_envs=num_eval_envs)
        # 用訓練時算好的 action 統計（同一份 unnorm_state）
        return SDSimlerWrapper(eval_args, self.unnorm_state)

    def _compute_video_env_ids(self, k: int = 4) -> list[int]:
        N = self.args.num_envs
        k = max(1, min(k, N))
        if k == 1:
            return [0]
        idxs = sorted({int(round(i)) for i in np.linspace(0, N - 1, k)})
        i = 0
        while len(idxs) < k and i < N:
            if i not in idxs:
                idxs.append(i)
            i += 1
        return sorted(idxs)

    def extract_obj_recep(self, text_string):
        pattern = r"put (.*?) on (.*)"
        match = re.search(pattern, text_string)

        if match:
            obj = match.group(1)
            recep = match.group(2)
            return obj, recep
        else:
            return None, None

    def _accumulate_for_success_only(
        self,
        obs_img, action, logprob, value, reward, done, env_info, instruction
    ):
        """
        把每個 env 的當步 transition 先塞到暫存。
        當某個 env 的 episode 結束（done=True）:
          - 若那條軌跡曾成功過（has_success=True），提交到 success_buffer
          - 否則丟棄。
        """
        num_envs = self.args.num_envs

        # 1) 逐 env 暫存當步 transition
        for i in range(num_envs):
            step_rec = dict(
                obs = obs_img[i].cpu().numpy(),                                # [H, W, C] uint8
                action = action[i].detach().to(torch.int32).cpu().numpy(),     # token ids / action 格式
                logprob = logprob[i].detach().to(torch.float32).cpu().numpy(), # (1,)
                value = value[i].detach().to(torch.float32).cpu().numpy(),     # (1,)
                reward = float(reward[i].item() if torch.is_tensor(reward[i]) else reward[i]),
                mask = float(1.0 - float(done[i].item() if torch.is_tensor(done[i]) else done[i])),
                instruction = instruction[i],
            )
            self._traj_cache[i].append(step_rec)

            # 若當步已經成功，做個標記（有些環境 success 會在 done 前就為 True）
            if isinstance(env_info, dict) and "success" in env_info:
                try:
                    if bool(env_info["success"][i]):
                        self._traj_has_success[i] = True
                except Exception:
                    pass  # 防禦性處理

        # 2) 若該 env 結束，決定提交或丟棄，並清空暫存
        for i in range(num_envs):
            is_done = bool(done[i].item() if torch.is_tensor(done[i]) else done[i])
            if is_done:
                if self._traj_has_success[i]:
                    # 提交：把整條軌跡收進成功資料池
                    self.success_buffer.append(self._traj_cache[i])
                # 重置該 env 的暫存
                self._traj_cache[i] = []
                self._traj_has_success[i] = False

    def _sample_sft_batch(self, bs: int):
        if len(self.success_buffer) == 0:
            return None
        steps = [step for traj in self.success_buffer for step in traj]
        if not steps:
            return None

        idx = np.random.randint(0, len(steps), size=bs)
        pick = [steps[i] for i in idx]

        images   = np.stack([p["obs"] for p in pick], axis=0).astype(np.uint8)    # CPU
        instruct = [p["instruction"] for p in pick]
        actions  = np.stack([p["action"] for p in pick], axis=0).astype(np.int64) # CPU

        obs = dict(
            image=torch.from_numpy(images).pin_memory(),  # CPU + pinned
            task_description=instruct
        )
        actions_t = torch.from_numpy(actions)             # CPU
        return (obs, actions_t)

    # def _train_sft_from_success_buffer(self) -> dict:
    #     steps = [step for traj in self.success_buffer for step in traj]
    #     n = len(steps)
    #     print("n =", n)

    #     idx = np.random.permutation(n)

    #     bs = 8
    #     epochs = 10
    #     grad_accum = 16

    #     self.policy.vla.train()
    #     self.policy.vla_optimizer.zero_grad(set_to_none=True)
    #     # self.policy.vh_optimizer.zero_grad(set_to_none=True)

    #     total_loss = 0.0
    #     total_lp = 0.0
    #     seen = 0
    #     num_batches = 0

    #     for ep in range(epochs):
    #         pbar = tqdm(range(0, n, bs), desc=f"Epoch {ep+1}/{epochs}", unit="batch")
    #         for s in pbar:
    #             e = min(s + bs, n)
    #             pick = idx[s:e]
    #             batch = [steps[k] for k in pick]

    #             images = np.stack([b["obs"] for b in batch], axis=0).astype(np.uint8)   # [B,H,W,C]
    #             instruct = [b["instruction"] for b in batch]
    #             actions = np.stack([b["action"] for b in batch], axis=0)               # [B, A_DIM]

    #             obs = dict(
    #                 image=torch.as_tensor(images, device=self.policy.tpdv["device"]),
    #                 task_description=instruct
    #             )
    #             actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.policy.tpdv["device"])


    #             logprob, entropy, values = self.policy.evaluate_actions(obs, actions_t)
                
    #             ce_loss = -(logprob.mean())  # CE（NLL）

    #             (ce_loss / grad_accum).backward()

    #             if ((num_batches + 1) % grad_accum == 0):
    #                 torch.nn.utils.clip_grad_norm_(self.policy.params_vla, 1)
    #                 self.policy.vla_optimizer.step()
    #                 self.policy.vla_optimizer.zero_grad(set_to_none=True)

    #             bsz = actions_t.shape[0]
    #             total_loss += float(ce_loss.item()) * bsz
    #             total_lp   += float(logprob.mean().item()) * bsz
    #             seen       += bsz
    #             num_batches += 1

    #             pbar.set_postfix({
    #                 "loss": total_loss / max(1, seen),
    #                 "avg_logprob": total_lp / max(1, seen)
    #             })

    #     info = {
    #         "sft/ran": 1,
    #         "sft/num_steps": seen,
    #         "sft/num_batches": num_batches,
    #         "sft/loss": total_loss / max(1, seen),
    #         "sft/avg_logprob": total_lp / max(1, seen),
    #     }

    #     self.policy.vla.eval()
    #     return info



    def _train_sft_from_success_buffer(self) -> dict:
        steps = [step for traj in self.success_buffer for step in traj]
        n = len(steps)
        print("n =", n)

        idx = np.random.permutation(n)

        bs = 4
        epochs = 30
        grad_accum = 32

        # 近鄰控制超參（可在 self.args 覆蓋）
        near_dims = getattr(self.args, "near_token_dims", (0, 1, 2))  # x,y,z 的維度索引
        near_coef = getattr(self.args, "near_coef", 0.8)              # 近鄰 loss 權重
        near_every = getattr(self.args, "near_every", 4)              # 每幾個 batch 算一次近鄰
        near_sample_one = getattr(self.args, "near_sample_one", True) # True: 單一方向；False: 同一維的 ±1 都算

        self.policy.vla.train()
        self.policy.vla_optimizer.zero_grad(set_to_none=True)
        # self.policy.vh_optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_lp = 0.0
        total_near = 0.0
        seen = 0
        num_batches = 0

        # 準備各維 token 的上下界；優先用模型/Tokenizer 提供，否則用資料觀測到的 min/max 當保守界線
        bounds = None
        if hasattr(self.policy, "token_id_minmax"):
            bounds = self.policy.token_id_minmax  # 期望為 [(lo0,hi0), (lo1,hi1), ...]
        elif hasattr(self.policy, "action_tokenizer") and hasattr(self.policy.action_tokenizer, "id_minmax"):
            bounds = self.policy.action_tokenizer.id_minmax

        if bounds is None:
            # fallback：用成功 buffer 估每維的 min/max
            all_actions_np = np.stack([b["action"] for b in steps], axis=0)  # [N, A_DIM]
            lo = all_actions_np.min(axis=0).tolist()
            hi = all_actions_np.max(axis=0).tolist()
            bounds = list(zip(lo, hi))  # [(lo0,hi0), (lo1,hi1), ...]

        def _shift_one_dim_keep_in_bounds(actions_long: torch.Tensor, dim: int, delta: int) -> torch.Tensor:
            """對指定維度做±1，若越界就保留原 token（回到原值）。"""
            lo, hi = bounds[dim]
            lo = int(lo); hi = int(hi)
            col = actions_long[:, dim]
            if delta > 0:
                can_inc = col < hi
                new_col = torch.where(can_inc, col + 1, col)
            else:
                can_dec = col > lo
                new_col = torch.where(can_dec, col - 1, col)
            out = actions_long.clone()
            out[:, dim] = new_col
            return out

        for ep in range(epochs):
            pbar = tqdm(range(0, n, bs), desc=f"Epoch {ep+1}/{epochs}", unit="batch")
            for s in pbar:
                e = min(s + bs, n)
                pick = idx[s:e]
                batch = [steps[k] for k in pick]

                images = np.stack([b["obs"] for b in batch], axis=0).astype(np.uint8)   # [B,H,W,C]
                instruct = [b["instruction"] for b in batch]
                actions = np.stack([b["action"] for b in batch], axis=0)               # [B, A_DIM]

                if getattr(self.args, "aug_enable_sft", False):
                    images = _augment_uint8_batch(
                        images,
                        prob=self.args.aug_prob,
                        brightness=self.args.aug_brightness,
                        contrast=self.args.aug_contrast,
                        noise_std=self.args.aug_noise_std,
                        cutout_p=self.args.aug_cutout_p,
                        cutout_ratio=self.args.aug_cutout_ratio,
                    )

                obs = dict(
                    image=torch.as_tensor(images, device=self.policy.tpdv["device"]),
                    task_description=instruct
                )
                actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.policy.tpdv["device"])

                # 主動作 NLL
                logprob, entropy, values = self.policy.evaluate_actions(obs, actions_t)
                ce_main = -(logprob.mean())

                # 是否在此 batch 計算近鄰
                do_near = (num_batches % near_every == 0)
                ce_near = torch.zeros((), device=ce_main.device)

                if do_near:
                    # 隨機抽 1 維
                    d = int(np.random.choice(list(near_dims)))
                    if near_sample_one:
                        # 只做單一方向（+1 或 -1 擇一）
                        delta = int(np.random.choice([+1, -1]))
                        neigh_actions = _shift_one_dim_keep_in_bounds(actions_t, d, delta)
                        neigh_lp, _, _ = self.policy.evaluate_actions(obs, neigh_actions)
                        ce_near = -(neigh_lp.mean())
                    else:
                        # 同一維同時計算 +1 與 -1（成本 ~2x 主動作）
                        neigh_plus  = _shift_one_dim_keep_in_bounds(actions_t, d, +1)
                        neigh_minus = _shift_one_dim_keep_in_bounds(actions_t, d, -1)
                        lp_plus,  _, _ = self.policy.evaluate_actions(obs, neigh_plus)
                        lp_minus, _, _ = self.policy.evaluate_actions(obs, neigh_minus)
                        ce_near = (-(lp_plus.mean()) + -(lp_minus.mean())) * 0.5

                # 合成總 loss
                ce_loss = ce_main + near_coef * ce_near

                # 反傳 + 梯度累積
                (ce_loss / grad_accum).backward()
                if ((num_batches + 1) % grad_accum == 0):
                    torch.nn.utils.clip_grad_norm_((self.policy.params_vla if hasattr(self.policy, "params_vla") else self.policy.vla.parameters()), 1.0)
                    self.policy.vla_optimizer.step()
                    self.policy.vla_optimizer.zero_grad(set_to_none=True)

                # 統計
                bsz = actions_t.shape[0]
                total_loss += float(ce_loss.item()) * bsz
                total_lp   += float(logprob.mean().item()) * bsz
                total_near += float(ce_near.item()) * bsz
                seen       += bsz
                num_batches += 1

                pbar.set_postfix({
                    "loss": total_loss / max(1, seen),
                    "avg_logprob": total_lp / max(1, seen),
                    "near_ce": total_near / max(1, seen),
                    "near_every": near_every
                })

        info = {
            "sft/ran": 1,
            "sft/num_steps": seen,
            "sft/num_batches": num_batches,
            "sft/loss": total_loss / max(1, seen),
            "sft/avg_logprob": total_lp / max(1, seen),
            "sft/near_loss": total_near / max(1, seen),
            "sft/near_coef": near_coef,
            "sft/near_every": near_every,
            "sft/near_dims": str(tuple(near_dims)),
            "sft/near_sample_one": int(bool(near_sample_one)),
        }

        self.policy.vla.eval()
        return info


    # sft
    def train_sft(self, episode):
        info = self._train_sft_from_success_buffer()

        info2 = {}
        info2["buffer/reward_mean"] = float(np.mean(self.buffer.rewards)) if len(self.buffer.rewards) else 0.0
        info2["buffer/mask_mean"] = float(np.mean(1.0 - self.buffer.masks)) if len(self.buffer.masks) else 0.0

        out = {}
        for k, v in info.items():
            out[f"train/{k}"] = v
        for k, v in info2.items():
            out[k] = v

        self.success_buffer.clear()  # 釋放影像記憶體
        gc.collect()

        return out


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
            sigma = 0.05 * (action_low - action_high)
            action_tokenizer = ActionTokenizer(self.policy.processor.tokenizer)
            
            batch_values, batch_actions, batch_logprobs = self.policy.get_action(obs_batch, deterministic)

            for j in range(self.args.buffer_inferbatch):
                env_idx = i + j
                if self.successed[env_idx]:
                    top_k = B
                else:
                    top_k = 10
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
    def eval_old(self, obj_set: str, object: list[str], receptacle: list[str], episode=-1) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])

        obs_img, instruction, info, _ = self.env.reset(obj_set=obj_set, same_init=self.args.use_same_init, object=object, receptacle=receptacle, set_costmap=False)
        

        viz_writers = []
        for i in range(self.args.num_envs):
            viz_path = f"eval_video_-1_sft_multitask_0.05_guide15/{instruction[0]}/{episode}/costmap_vis_{i:04d}.mp4"
            os.makedirs(os.path.dirname(viz_path), exist_ok=True)
            viz_writer = imageio.get_writer(viz_path, fps=10, codec="libx264")
            viz_writers.append(viz_writer)
            viz_writers[i].append_data(obs_img[i].cpu().numpy())
        print("Evaluating:", instruction[0])

        for _ in tqdm(range(self.args.episode_len), desc="eval"):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img, reward, done, env_info = self.env.step(action)

            # info
            # print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

            for i in range(self.args.num_envs):
                viz_writers[i].append_data(obs_img[i].cpu().numpy())

        for i in range(self.args.num_envs):
            viz_writers[i].close()
        
        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        return env_stats

    @torch.no_grad()
    def eval(
        self,
        obj_set: str,
        object: list[str],
        receptacle: list[str],
        ep,
        save_video: bool = True,
        fps: int = 10,
    ) -> dict:
        """用既有的 self.env（num_envs 平行）進行評估；影片仍用 images_to_video（僅 self.video_env_ids）。"""
        self.policy.prep_rollout()

        num_envs = self.args.num_envs

        # 確保 object/receptacle 長度與 env 數一致（若只給單一元素就複製）
        if len(object) != num_envs:
            object = [object[0]] * num_envs
        if len(receptacle) != num_envs:
            receptacle = [receptacle[0]] * num_envs

        # 直接 reset 既有的 self.env（不建立新環境）
        obs_img, instruction, info, _ = self.env.reset(
            obj_set=obj_set,
            same_init=self.args.use_same_init,
            object=object,
            receptacle=receptacle,
            set_costmap=False
        )
        print(f"Evaluating({num_envs}-env, reuse self.env):", instruction[0])

        env_infos = defaultdict(lambda: [])

        # 僅為選定 env 蒐集影格，避免記憶體暴漲
        frames = {i: [] for i in self.video_env_ids}
        for i in self.video_env_ids:
            frames[i].append(obs_img[i].cpu().numpy())

        last_env_info = None

        # rollout
        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            _, action, _ = self._get_action(obs, deterministic=True)

            obs_img_new, reward, done, env_info = self.env.step(action)
            last_env_info = env_info

            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[k] += v  # 每個 k 是長度 num_envs 的 list

            # 只記錄選定 env 的畫面
            for i in self.video_env_ids:
                frames[i].append(obs_img_new[i].cpu().numpy())

            obs_img = obs_img_new

        # 存影片與統計
        if save_video:
            exp_dir = Path(self.glob_dir) / f"eval_{self.eval_idx}_{obj_set}_{num_envs}env_reuse"
            exp_dir.mkdir(parents=True, exist_ok=True)

            # 讀取 success 標記（最後一步）
            success_flags = [0] * num_envs
            if last_env_info is not None and "success" in last_env_info:
                success_flags = []
                for x in last_env_info["success"]:
                    try:
                        import torch as _torch
                        if _torch.is_tensor(x):
                            success_flags.append(int(x.item()))
                        else:
                            success_flags.append(int(x))
                    except Exception:
                        try:
                            success_flags.append(int(x))
                        except Exception:
                            success_flags.append(0)

            # 逐個選定 env 輸出影片（沿用 images_to_video）
            for i in self.video_env_ids:
                video_name = f"video_ep{ep}_env{i:02d}-{object[i]}_{receptacle[i]}-s_{success_flags[i]}"
                images_to_video(frames[i], str(exp_dir), video_name, fps=fps, verbose=True)

            # 存一份統計與每 env 的最後一步資訊
            per_env_last = {}
            if last_env_info is not None:
                per_env_last = {
                    i: {k: last_env_info[k][i].tolist() for k in last_env_info.keys() if k != "episode"}
                    for i in range(num_envs)
                }

            save_stats = {
                "env_name": self.args.env_id,
                "ep_len": self.args.episode_len,
                "epoch": int(self.eval_idx),
                "stats": {k: float(np.mean(v)) for k, v in env_infos.items()},
                "instruction": {i: instruction[i] for i in range(num_envs)},
                "last_info": per_env_last,
            }
            yaml.dump(save_stats, open(exp_dir / "stats.yaml", "w"))
            self.eval_idx += 1

        # 不需關 env（沿用 self.env）
        gc.collect()
        torch.cuda.empty_cache()

        return {k: np.mean(v) for k, v in env_infos.items()}


    @torch.no_grad()
    def render(self, epoch: int, obj_set: str, object: list[str], receptacle: list[str]) -> dict:
        self.policy.prep_rollout()

        env_infos = defaultdict(lambda: [])
        datas = [{
            "image": [],
            "instruction": "",
            "action": [],
            "info": [],
        } for idx in range(self.args.num_envs)]

        obs_img, instruction, info, _ = self.env.reset(
            obj_set=obj_set, same_init=self.args.use_same_init, object=object, receptacle=receptacle, set_costmap=False
        )
        print("Rendering:", instruction[0])

        # instruction
        for idx in range(self.args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img_new, reward, done, env_info = self.env.step(action)

            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

            post_action = self.env._process_action(action)
            for i in range(self.args.num_envs):
                log_image = obs_img[i].cpu().numpy()
                log_action = post_action[i].cpu().numpy().tolist()
                log_info = {k: v[i].tolist() for k, v in env_info.items() if k != "episode"}
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)

            obs_img = obs_img_new

        # 最後影格
        for i in range(self.args.num_envs):
            datas[i]["image"].append(obs_img[i].cpu().numpy())

        exp_dir = Path(self.glob_dir) / f"vis_{epoch}_{obj_set}"
        print("exp_dir : ", exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 只為選定 env 存影片
        to_save = set(self.video_env_ids)
        for i in to_save:
            images = datas[i]["image"]
            infos = datas[i]["info"]
            assert len(images) == len(infos) + 1

            if self.args.render_info:
                for j in range(len(infos)):
                    images[j + 1] = visualization.put_info_on_image(
                        images[j + 1], infos[j], extras=[f"Ins: {instruction[i]}"]
                    )

            success = int(infos[-1]["success"])
            images_to_video(
                images, str(exp_dir), f"video_{i}-{object[0]}_{receptacle[0]}-s_{success}",
                fps=10, verbose=False
            )

        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats_ret = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        # 存統計（維持原行為）
        last_info = {idx: {k: env_infos[k][idx] for k in env_infos.keys()} for idx in range(self.args.num_envs)}
        save_stats = {
            "env_name": self.args.env_id,
            "ep_len": self.args.episode_len,
            "epoch": epoch,
            "stats": {k: v.item() for k, v in env_stats.items()},
            "instruction": {idx: ins for idx, ins in enumerate(instruction)},
            "last_info": last_info,
        }
        yaml.dump(save_stats, open(exp_dir / "stats.yaml", "w"))

        return env_stats_ret

    def run(self):
        max_episodes = self.args.steps_max // self.args.episode_len // self.args.num_envs
        max_episodes = 100
        instruction_switch_interval = 80
        steps = 0

        num_envs = self.args.num_envs 
        group_size = num_envs // 4   

        # print(f"Evaluating at {steps}")
        # for task in self.task_list:
        #     object, receptacle = self.extract_obj_recep(task)
        #     sval_stats = self.eval_old("train", [object]*self.args.num_envs, [receptacle]*self.args.num_envs)
        #     sval_stats = {f"eval＿put_{object}_in_{receptacle}/{k}": v for k, v in sval_stats.items()}
        #     wandb.log(sval_stats, step=steps) 

        for episode in range(max_episodes):
            env_infos = defaultdict(lambda: [])
            ep_time = time.time()

            # ***** [ADD] COSTMAP-SWITCH init *****
            # subtask_id           = [0] * 64          # 目前 costmap 的索引
            # current_costmap_id   = [-1] * 64         # 用來偵測是否已切換
            # grasp_success_count  = [0] * 64          # 若 costmap 指定 grasp-based switch
            # best_cost            = [1e9] * 64        # 用來記錄該 step 的最佳 cost
            subtask_id           = [0] * num_envs
            current_costmap_id   = [-1] * num_envs
            grasp_success_count  = [0] * num_envs
            best_cost            = [1e9] * num_envs
            self.successed       = [False] * self.args.num_envs
            # =========================================

            objects, receptacles = [], []
            # test
            for i in range(4):
                obj, recep = self.extract_obj_recep(self.task_list[i])
                objects.extend([obj] * group_size)
                receptacles.extend([recep] * group_size)

            # self.task_id = 3
            # obj, recep = self.extract_obj_recep(self.task_list[self.task_id])
            # objects.extend([obj] * num_envs)
            # receptacles.extend([recep] * num_envs)
            ### 


            costmap_dir = f"costmap/rlvla/costmaps2"
            env_reset_options = {
                "obj_set": "train",
                "same_init": self.args.use_same_init,
                "object": objects,
                "receptacle": receptacles,
                "set_costmap": True
            }

            costmap_handler_list = initialize_cost_map(
                self.env, env_reset_options, costmap_dir, "3rd_view_camera", 4, 4
            )

            costmap_handler = []
            for i in  range(self.args.num_envs):
                group = i // group_size
                group_index = (i - group * group_size) // (group_size / 4)
                costmap_handler.append(costmap_handler_list[group + int(group_index) * 4])


            obs_img, instruction, info, _ = self.env.reset(
                obj_set="train",
                same_init=self.args.use_same_init,
                object=objects,
                receptacle=receptacles,
                set_costmap=False
            )

            task_id_map = []
            # test
            for i in range(4):
                task_id_map.extend([(self.task_id + i) % len(self.task_list)] * group_size)
            # task_id_map.extend([self.task_id] * num_envs)
            ### 

            self.buffer.warmup(obs_img.cpu().numpy(), instruction)

            
            # obj, recep = self.extract_obj_recep(self.task_list[self.task_id])
            # obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init, object=[obj]*self.args.num_envs, receptacle=[recep]*self.args.num_envs)
            # self.buffer.warmup(obs_img.cpu().numpy(), instruction)
            rollout_images = [[] for _ in range(self.args.num_envs)]
            # rollout_images = {i: [] for i in self.video_env_ids}

            print("instruction 0 : ", instruction[0])
            print("instruction 8 : ", instruction[8])
            print("instruction 16 : ", instruction[16])
            print("instruction 24 : ", instruction[24])




            cost_map_info = []
            # 只為選定 env 開 writer，其餘保持 None 方便往後索引
            viz_writers = [None] * self.args.num_envs
            # to_save = set(self.video_env_ids)
            # for i in to_save:
            #     viz_path = f"videos_with_costmap/{episode}/costmap_vis_{i:04d}.mp4"
            #     os.makedirs(os.path.dirname(viz_path), exist_ok=True)
            #     viz_writers[i] = imageio.get_writer(viz_path, fps=10, codec="libx264")

            for step_idx in tqdm(range(self.args.training_len), desc="rollout"):
                # costmap_handler = []
                # test
                # for i in  range(self.args.num_envs):
                #     costmap_handler.append(costmap_handler_list[(self.task_id + (i // group_size)) % len(self.task_list)])
                    # costmap_handler.append(costmap_handler_list[self.task_id])
                ###
                for i in range(self.args.num_envs):
                    if len(cost_map_info) != self.args.num_envs:
                        cost_map_info.append(costmap_handler[i].get_costmap_info(subtask_id[i]))
                        current_costmap_id[i] = subtask_id[i]
                    elif subtask_id[i] != current_costmap_id[i]:
                        cost_map_info[i] = costmap_handler[i].get_costmap_info(subtask_id[i])
                        current_costmap_id[i] = subtask_id[i]

                if episode <= 15:
                    value, action, logprob, best_cost = self.collect_costmap(envs=self.env, costmap_handler=costmap_handler, subtask_id=subtask_id, viz_writers=viz_writers)
                else:
                    value, action, logprob = self.collect()
                    best_cost = [1e9] * num_envs
                obs_img, reward, done, env_info = self.env.step(action)
                for idx in range(self.args.num_envs):
                    if env_info["success"][idx]:
                        print(f"[{idx}] success!!!!!!")
                        self.successed[idx] = True
                        # with open("/workspace/AutoRL_SD/log.txt", "a") as f:
                        #     f.write("[{idx}] success!\n")
                for env_i in self.video_env_ids:
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


                self._accumulate_for_success_only(
                    obs_img=obs_img,
                    action=action,
                    logprob=logprob,
                    value=value,
                    reward=reward,
                    done=done,
                    env_info=env_info,
                    instruction=instruction,
                )
                
                if getattr(self.args, "aug_enable_sft", False) and random.random() <= 0.25:
                    obs_img = _augment_uint8_batch(
                        obs_img.detach().cpu().numpy(),
                        prob=self.args.aug_prob,
                        brightness=self.args.aug_brightness,
                        contrast=self.args.aug_contrast,
                        noise_std=self.args.aug_noise_std,
                        cutout_p=self.args.aug_cutout_p,
                        cutout_ratio=self.args.aug_cutout_ratio,
                    )
                    obs_img = torch.from_numpy(obs_img).to(dtype=torch.float64, device="cuda:0")
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

                    # 先丟掉大物件與 CUDA cache，降低 fork ffmpeg 時的峰值記憶體
                    try:
                        del value, action, logprob, reward, done
                    except Exception:
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()

                    # for env_i in self.video_env_ids:
                    #     images = rollout_images[env_i]
                    #     try:
                    #         images_to_video(images, str(render_dir), f"env{env_i}", fps=10, verbose=False)
                    #     except OSError as e:
                    #         # 若記憶體不足（Errno 12），跳過這支影片，訓練不中斷
                    #         if getattr(e, "errno", None) == 12:
                    #             print(f"[WARN] ENOMEM while creating video for env{env_i}; skip this video.")
                    #         else:
                    #             raise
                    for env_i in range(self.args.num_envs):
                        images = rollout_images[env_i]
                        images_to_video(images, str(render_dir), f"env{env_i}", fps=10, verbose=False)

                    # 寫完立刻清空 frame 並再做一次回收
                    rollout_images = [[] for _ in range(self.args.num_envs)]
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.args.alg_ppo_epoch = 1

                    if episode >= 5:
                        self.args.alg_ppo_epoch = 5
                        # self.args.alg_gradient_accum = 40
                        print(f"[alg_ppo_epoch] switch to 10 at episode={episode}")

                    if episode >= 10:
                        self.args.alg_ppo_epoch = 2
                        # self.args.alg_gradient_accum = 40
                        print(f"[alg_ppo_epoch] switch to 3 at episode={episode}")


                    # if episode >= 60:
                    #     self.args.alg_ppo_epoch = 1
                    #     # self.args.alg_gradient_accum = 40
                    #     print(f"[alg_ppo_epoch] switch to 3 at episode={episode}")

                    # train
                    # ==== 只做一次 SFT  ====
                    if not self.did_sft:
                        if len(self.success_buffer) >= self.sft_min_trajs:
                            # 只做這一次 SFT
                            infos = self.train_sft(episode)
                            self.did_sft = True
                        else:
                            infos = {}
                        # infos = self.train(episode)
                    else:
                        infos = self.train(episode)

                    for k, v in env_infos.items():
                        infos[f"env/{k}"] = np.mean(v)

                    # train
                    # infos = self.train(episode)
                    # for k, v in env_infos.items():
                    #     infos[f"env/{k}"] = np.mean(v)
                    # wandb.log(infos, step=step_idx + episode * self.args.training_len)
                    self.buffer.warmup(obs_img.cpu().numpy(), instruction)

                    # Switch Instruction
                    # test 
                    # self.task_id = (self.task_id + 1) % len(self.task_list)
                    objects, receptacles = [], []
                    for i in range(4):
                        obj, recep = self.extract_obj_recep(self.task_list[i])
                        objects.extend([obj] * group_size)
                        receptacles.extend([recep] * group_size)
                    self.env.set_task(objects, receptacles)
                    # objects, receptacles = [], []
                    # objects = [obj]  * self.args.num_envs
                    # receptacles = [recep] * self.args.num_envs
                    # self.env.set_task(objects, receptacles)
                    ###

                    # obj, recep = self.extract_obj_recep(self.task_list[self.task_id])
                    # self.env.set_task([obj]*self.args.num_envs, [recep]*self.args.num_envs)
                    instruction = self.env.get_language_instruction()
                    print(step_idx, "switch instruction to ", instruction[0])
                    self.buffer.update_instruction(instruction)

            # steps
            steps = (episode + 1) * self.args.training_len * self.args.num_envs
            # print(pprint.pformat({k: round(np.mean(v), 4) for k, v in env_infos.items()}))

            for i in self.video_env_ids:
                if viz_writers[i] is not None:
                    viz_writers[i].close()

            # eval
            if episode % self.args.interval_eval == self.args.interval_eval - 1 or episode == max_episodes - 1:
                print(f"Evaluating at {steps}")
                for task in self.task_list:
                    object, receptacle = self.extract_obj_recep(task)
                    sval_stats = self.eval_old("train", [object]*self.args.num_envs, [receptacle]*self.args.num_envs, episode)
                    sval_stats = {f"eval＿put_{object}_in_{receptacle}/{k}": v for k, v in sval_stats.items()}
                    wandb.log(sval_stats, step=steps)

            # if episode % self.args.interval_eval == self.args.interval_eval - 1 or episode == max_episodes - 1:
            #     print(f"Evaluating at {steps}")
            #     for object in self.env.get_object_names()[0]:
            #         for receptacle in self.env.get_receptacle_names()[0]:
            #             sval_stats = self.eval("train", [object]*self.args.num_envs, [receptacle]*self.args.num_envs, episode)
            #             sval_stats = {f"eval_put_{object}_in_{receptacle}/{k}": v for k, v in sval_stats.items()}
            #             print(sval_stats)
            #             wandb.log(sval_stats, step=steps)

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
