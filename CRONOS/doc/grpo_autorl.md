# AutoRL GRPO — 現有實作整理 / Existing implementation review

Index of these documents: [`README.md`](README.md).

**範圍 / Scope:** `AutoRL/SimplerEnv/simpler_env/` 的 GRPO 路徑，唯讀盤點，為「CRONOS 沿用 AutoRL
GRPO」做準備。AutoRL 未被修改。
The GRPO path in `AutoRL/SimplerEnv/simpler_env/`, read-only, in preparation for reusing it in
CRONOS. Nothing in AutoRL was modified.

**狀態 / Status:** 純程式碼閱讀 + numpy 重現，未在硬體上跑過（本機沒有能載入 VLA 的 GPU）。
所有數值宣稱下面都附了可重跑的驗證方式。
Code reading plus a numpy reproduction; not run on hardware (no GPU here large enough for a VLA).
Every numeric claim below states how to re-verify it.

---

## 1. 完整呼叫鏈 / The full call chain

```
train_ms3_ppo.py:606   infos = self.train()
        │
train_ms3_ppo.py:269   def train(self)
        ├── alg_name == "ppo"   :275  → alg.train_ppo(self.prealloc_buffer)
        └── alg_name == "grpo"  :278  → alg.train_grpo(self.buffer)
                                            │
openvla_train.py:530   def train_grpo(self, buffer)
        ├── :534  buffer.compute_returns_grpo()
        ├── :535  minibatch_count = buffer.get_minibatch_count()
        └── loop  train_grpo_step(idx, minibatch_count, batch)
                    │
openvla_train.py:402   def train_grpo_step(...)
```

旗標 / Flags — `train_ms3_ppo.py:109-115`

| 旗標 / Flag | 預設 / Default | 用途 / Purpose |
|---|---|---|
| `alg_name` | `"ppo"` | `ppo` \| `grpo`；`:129` 有 assert |
| `alg_grpo_fix` | `True` | 見 §3 — 只正規化非零 reward / normalize non-zero rewards only |
| `alg_gradient_accum` | `20` | 兩條路徑共用 / shared by both paths |
| `alg_ppo_epoch` | `1` | 兩條路徑共用 / shared |
| `alg_entropy_coef` | `0.0` | 兩條路徑共用；預設 0 使 entropy 項無效 / zero by default, so the entropy term is inert |

`ppo_clip = 0.2`、`ppo_grad_norm = 10.0` 是 `OpenVLAPPO.__init__` 的硬編碼常數
（`openvla_train.py:286-287`），GRPO 沿用同樣的值。
Hard-coded in `OpenVLAPPO.__init__`; GRPO reuses both.

---

## 2. ⚠️ GRPO 與 PPO 吃的不是同一個 buffer / The two paths consume different buffers

**中文** — 這是最容易被忽略、但影響最大的結構差異。

**English** — The most consequential structural difference, and the easiest to miss.

AutoRL 有兩個 buffer / AutoRL keeps two buffers (`train_ms3_ppo.py:169-187`):

| Buffer | 類別 / Class | 內容 / Holds |
|---|---|---|
| `self.buffer` | `SeparatedReplayBuffer`，`(episode_len, num_envs)` | **只有最近一個 segment**（`insert` 用 `% ep_len` 原地繞回，從不 `reset`）/ **only the most recent segment** — `insert` wraps in place via `% ep_len` and is never reset |
| `self.prealloc_buffer` | `PreallocReplayBuffer` 或 `FIFOReplayBuffer` | 每個 segment 邊界 `cat_buffer(self.buffer)` 累積（`:604`），`train()` 後 `reset()`（`:607`）/ accumulated per segment boundary, reset after each update |

於是 / Consequently：

- **PPO** 訓練 `prealloc_buffer` → `training_interval / episode_len` 個 segment
  （預設 `160/80 = 2` 個 segment × 64 envs = 128 條軌跡）。
- **GRPO** 訓練 `self.buffer` → **只有 1 個 segment**（64 條軌跡）。

**中文** — 在 AutoRL 預設設定下，GRPO 每次更新看到的資料量只有 PPO 的一半，且 `cat_buffer` 的累積結果
對 GRPO 完全沒用到。這不是設定選項，是寫死在 `train()` 的分支裡。

**English** — Under AutoRL's defaults GRPO sees half the data PPO does per update, and the
accumulation `cat_buffer` performs is simply unused by GRPO. This is hard-coded in the `train()`
branch, not a configurable choice.

---

## 3. `compute_returns_grpo` 的實際數學 / The actual math

`replay_buffer.py:107-122`

```python
def compute_returns_grpo(self):
    if self.alg_grpo_fix:
        rewards_valid = self.rewards[self.rewards != 0]
        rewards_norm = self.rewards.copy()
        rewards_norm[rewards_norm != 0] -= rewards_valid.mean()
        rewards_norm[rewards_norm != 0] /= (rewards_valid.std() + 1e-5)
    else:
        rewards_norm = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-5)

    returns = 0
    for step in reversed(range(self.rewards.shape[0])):
        returns = rewards_norm[step] + self.masks[step + 1] * returns
        self.returns[step] = returns

    self.advantages = self.returns.copy()
```

逐項拆解 / Item by item:

| 元素 / Element | AutoRL 的做法 / What AutoRL does |
|---|---|
| **分組 / Grouping** | **完全沒有分組。** 正規化的統計量取自整個 `(episode_len, num_envs, 1)` 陣列 —— 所有 env、所有時間步一起算 mean/std。/ **No grouping at all.** Statistics are taken over the entire array. |
| **`alg_grpo_fix`** | `True` 時只用**非零** reward 算 mean/std，並且只把非零項平移縮放，零項保持為零。`RewardShaper` 送出的是 potential *差分*，多數步為 0，所以這一步是為了避免統計量被大量的零稀釋。/ Uses only **non-zero** rewards for the statistics and rescales only those entries. The reward is a potential *difference*, mostly zero, so this prevents the zeros from dominating. |
| **折扣 / Discount** | **無折扣（γ = 1）**。`returns = r_norm[t] + mask[t+1] * returns[t+1]`，`buffer_gamma` 在 GRPO 路徑上完全不使用。/ **Undiscounted.** `buffer_gamma` is unused on the GRPO path. |
| **Baseline** | **沒有**。`advantages = returns`，不減任何 baseline，也不再做一次正規化。/ **None.** No baseline subtraction, no second normalization. |
| **Critic** | 完全不參與。`value_preds` 仍被 rollout 寫入，但 `compute_returns_grpo` 不讀、`train_grpo_step` 不用。/ Not involved at all. |

**中文** — 所以 AutoRL 的「GRPO」實際上是**批次正規化 reward 的無折扣 REINFORCE**，
外加 PPO 的 clipped surrogate。它並不是 DeepSeekMath 意義下的 group-relative：沒有「同一個 prompt
取樣 G 條、組內比較」這一層。真正讓它接近 GRPO 的，是 §5 講的環境端場景廣播。

**English** — AutoRL's "GRPO" is batch-normalized undiscounted REINFORCE wrapped in PPO's clipped
surrogate. It is *not* group-relative in the DeepSeekMath sense — there is no "sample G completions
per prompt and compare within the group" layer. What actually makes it GRPO-like is the
environment-side scene broadcast described in §5.

---

## 4. `train_grpo_step` 與 `train_ppo_step` 的差異 / Diff against the PPO step

`openvla_train.py:402-449` vs `:291-364`

| | `train_ppo_step` | `train_grpo_step` |
|---|---|---|
| Value loss | 有（clipped Huber, δ=10）/ yes | **無 / none** |
| 優化器 / Optimizers | `vh_optimizer` + `vla_optimizer` | **只有 `vla_optimizer`** / `vla_optimizer` only |
| 梯度裁剪 / Grad clip | `params_vla + params_vh` 一起 / jointly | **只有 `params_vla`** / `params_vla` only |
| Policy loss 歸約 / reduction | `-min(surr1, surr2).mean()` | `-min(surr1, surr2).sum(dim=-1, keepdim=True).mean()` |
| Entropy | `- coef * entropy.mean()`（coef 預設 0）/ same, coef 0 by default | 同上 / same |
| BC 項 / BC term | `lambda_bc * bc_loss`，但只在 `demo_batch is not None` 時；**沒有任何呼叫端傳 `demo_batch`**，故為死碼 / dead code — no caller passes `demo_batch` | 無 / none |

Value head 的處理是乾淨的 / The value head is handled cleanly: GRPO 的 loss 從不引用 `values`，
所以沒有梯度流到 `params_vh`，`vh_optimizer.step()` 不呼叫也不會有差別。
`evaluate_actions` 仍然回傳 `values`，只是被丟棄。
GRPO's loss never references `values`, so no gradient reaches `params_vh`; `evaluate_actions` still
returns `values`, they are simply discarded.

### 4.1 ⚠️ `sum(dim=-1)` 造成 policy loss 被放大 `act_dim` 倍 / a ×`act_dim` scale factor

**中文** — 這是閱讀時發現的實際缺陷，會直接影響 GRPO 與 PPO 的可比性。

**English** — A real defect found while reading; it directly affects GRPO-vs-PPO comparability.

三個事實相乘 / Three facts combine:

1. `replay_buffer.py:20` — `action_log_probs` 配置為 `(ep_len, num_env, act_dim)`，`act_dim=7`
   （`train_ms3_ppo.py:186`）。
2. `policy.get_action` 回傳的 logprob 是 **`[B, 1]`**（`openvla_train.py:181` 的 assert），
   `insert` 在 `:74` 直接 `self.action_log_probs[self.step] = action_log_probs.copy()` ——
   numpy 把 `[B,1]` **靜默廣播**成 `[B,7]`，同一個純量複製 7 份。
3. `evaluate_actions` 回傳的 logprob 是 `[B, 1]`（`:225` 的 assert）。

於是 `ratio = exp(logprob - old_logprob)` 廣播成 `[mb, 7]`（7 個相同的欄），
`sum(dim=-1)` 把它加起來 → policy loss 剛好是正確值的 **7 倍**。
`train_ppo_step` 用的是 `.mean()`，對 7 個相同的欄取平均後回到正確值，**不受影響**。

So `ratio` broadcasts to `[mb, 7]` with 7 identical columns, and `sum(dim=-1)` makes the GRPO policy
loss exactly **7×** the intended value. `train_ppo_step` uses `.mean()` over those identical columns
and is unaffected.

實測 / Reproduced (numpy, no GPU needed):

```
grpo policy_loss = -6.600392
ppo  policy_loss = -0.942913
grpo / ppo       = 7.0000   (act_dim = 7)
```

**中文** — 由於 `alg_entropy_coef` 預設為 0 且 GRPO 沒有 value loss，這個 7 倍是**整個 loss 的**
7 倍，等同於把 policy 的有效學習率乘以 7（`clip_grad_norm_(…, 10.0)` 會在梯度夠大時把它截掉一部分，
所以不是乾淨的 7 倍，而是「更常撞到梯度裁剪上限」）。要跟 PPO 做公平比較，這一項必須先對齊。

**English** — With `alg_entropy_coef = 0` and no value loss, the 7× applies to the *whole* GRPO loss —
equivalent to a 7× effective learning rate on the policy. `clip_grad_norm_(…, 10.0)` truncates part of
it once gradients are large, so the practical effect is "hits the grad-norm ceiling far more often"
rather than a clean 7×. This must be reconciled before any fair GRPO-vs-PPO comparison.

**注意 / Note** — SpatialVLA 的 `act_token_len` 是 3，不是 7，所以這個倍率是 policy-dependent。
For SpatialVLA the factor would be 3, not 7 — the bug is policy-dependent.

---

## 5. 讓它成為「group」的其實是環境端 / What actually makes it a group

**中文** — §3 說 `compute_returns_grpo` 完全沒有分組。那它為什麼還算 GRPO？因為在 AutoRL 裡，
**整批 env 本來就跑同一個場景**，所以「整批正規化」等價於「組內正規化」，而那個組就是同一個 prompt 的
64 個樣本。

**English** — §3 established that `compute_returns_grpo` does no grouping. What makes it GRPO-ish is
that in AutoRL **the whole batch runs the identical scene**, so batch normalization *is* group
normalization, and the group is 64 samples of one prompt.

| Codebase | `use_same_init` | `episode_id` | 同批 env 的場景 / Scene across the batch |
|---|---|---|---|
| **AutoRL** | `True`（`train_ms3_ppo.py:84`） | `torch.full((num_envs,), rand_episode_id)` 廣播（`simpler_wrapper.py:36,134`） | **完全一致** — carrot、plate、overlay、xyz、quat 全部相同 / fully identical |
| **RL4VLA** | `False`（`train_ms3_ppo.py:45`） | 不傳 → `randint(0, ltt, (b,))` 每 env 獨立 | **完全不一致** / fully heterogeneous |
| **CRONOS 現況 / today** | — | 廣播後被覆寫 / broadcast then overwritten | **部分一致** — 物件/容器/背景依 YAML group 一致，**姿態每 env 獨立** / partial |

差異出在 `_initialize_episode_pre` 的姿態覆寫分支，兩邊條件是**相反**的 /
The divergence is one inverted condition:

- AutoRL `pick_place_multi.py:719 / 1059 / 1102 / 1474`（四個 env 變體，極性一致 / all four env
  variants share the polarity）— `if obj_set == "fixed"` 才重抽姿態；預設 `obj_set="rand"`
  （`train_ms3_ppo.py:48`）不進這個分支 → 姿態沿用廣播的 `episode_id` → **一致**。
- CRONOS `bridge_multi.py:1018`（唯一一處 / the only site）— `if obj_set != "fixed"` 就重抽；
  預設 `obj_set="rand"`（`main.py:45`）會進這個分支 → **每 env 獨立**。
  （該處註解寫 "matching AutoRL exactly"，但條件其實相反 / the comment claims parity, the condition
  is inverted.）

**中文** — 這代表把 `compute_returns_grpo` 原封不動搬進 CRONOS **不會**得到 AutoRL 的 GRPO 語意：
CRONOS 的批次是異質的，整批正規化會把不同任務、不同初始姿態的 reward 混在一起當基準。
要沿用 AutoRL 的 GRPO，環境端的場景廣播必須一起沿用。

**English** — Porting `compute_returns_grpo` verbatim into CRONOS would **not** reproduce AutoRL's
GRPO semantics: CRONOS's batch is heterogeneous, so batch normalization pools rewards across
different tasks and different initial poses. Reusing AutoRL's GRPO means reusing its scene broadcast
as well.

---

## 6. 盤點結果 / Findings summary

| # | 發現 / Finding | 嚴重度 / Severity |
|---|---|---|
| 1 | GRPO 訓練 `self.buffer`（1 個 segment），PPO 訓練 `prealloc_buffer`（N 個 segment）。預設下 GRPO 每次更新資料量減半 / GRPO trains on half the data | 高 / high — 影響任何 PPO-vs-GRPO 比較 |
| 2 | `sum(dim=-1)` × `[B,1]→[B,7]` 廣播 → policy loss ×`act_dim`（OpenVLA 7、SpatialVLA 3）/ ×`act_dim` policy loss | 高 / high — 等同 7× 有效學習率 |
| 3 | `compute_returns_grpo` 完全不分組；「組」實際上來自環境端的場景廣播 / grouping is environmental, not algorithmic | 中 / medium — 移植時最容易漏掉的一環 |
| 4 | 無折扣（γ=1），`buffer_gamma` 在 GRPO 路徑失效 / undiscounted; `buffer_gamma` inert | 低 / low — 但要寫進文件避免誤解 |
| 5 | 無 baseline、無 KL-to-reference 項（標準 GRPO 有 KL 正則）/ no baseline, no KL term | 低 / low — 設計選擇，非錯誤 |
| 6 | `lambda_bc` / `bc_loss` 是死碼：只在 `train_ppo_step` 的 `demo_batch is not None` 分支，而沒有呼叫端傳 `demo_batch` / dead code | 低 / low |

---

## 7. 沿用到 CRONOS 的對應 / Mapping onto CRONOS

**中文** — 「沿用 AutoRL」在 CRONOS 的具體意思，以及每一項需要決定什麼。

**English** — What "reuse AutoRL" concretely means in CRONOS, and what each item needs decided.

| AutoRL | CRONOS 對應 / Counterpart | 備註 / Note |
|---|---|---|
| `Args.alg_name` / `alg_grpo_fix` | 加到 `main.py` 的 `alg_*` 區塊（`main.py:100-109`）/ add to the existing `alg_*` block | 命名直接沿用 / names carry over verbatim |
| `SeparatedReplayBuffer.compute_returns_grpo` | `CronosReplayBuffer.compute_grpo_returns()` | CRONOS 的 slot 軸是「segment × env 攤平」，`self.rewards[:, :num_env]` 是全部 segment；要決定是否只取最後一個 segment 以對齊 AutoRL 語意 / decide whether to restrict to the last segment to match AutoRL |
| `OpenVLAPPO.train_grpo_step` | `training/grpo.py` 的 `CronosGRPO` | CRONOS 的 `action_log_probs` 已經是 `(…, 1)`，**沒有** §4.1 的廣播問題；因此要決定是重現 AutoRL 的 ×`act_dim` 以求數值可比，還是採用正確的 `.mean()` / CRONOS's buffer is already `(…,1)`, so the ×`act_dim` bug does not reproduce — decide whether to match AutoRL numerically or fix it |
| `Runner.train()` 的 if/elif | `main.py:867` `_run_ppo_update` 內的分支 | 方法名保持 `_run_ppo_update`（`tools/bench_rollout.py:218` 靠名字 monkey-patch）/ keep the name |
| `use_same_init` + 廣播的 `episode_id` | `wrapper._build_options()` + `bridge_multi._initialize_episode_pre` | §5 — 不做這個就沒有 group / without this there is no group |
| `prealloc_buffer` vs `buffer` | CRONOS 只有一個 buffer | 需要明確決定 GRPO 的更新視窗 / the GRPO update window needs an explicit decision |

### 已實作 / Implemented

| 項目 / Item | 決定 / Decision |
|---|---|
| 分組 / Grouping | `--grpo-group-scope batch\|scene\|task` —— 三層由寬到窄，見 §10。`batch` = AutoRL（已驗證與 `compute_returns_grpo` **逐元素相同**，`fix` 兩種分支皆是）。 |
| §4.1 的 ×`act_dim` | **不重現**。CRONOS 的 `action_log_probs` 寬度本來就是 1，那個廣播在此不可能發生；`training/grpo.py` 用 `.mean()`。**後果：CRONOS 的 GRPO loss 數值是 AutoRL 的 1/`act_dim`，跨 codebase 請比 grad-norm 不要比 loss。** |
| 更新視窗 / Update window | **訓練資料**沿用 CRONOS 的整個 `ppo_update_len` 視窗（預設 2 個 segment），比 AutoRL 多（它的 GRPO 丟掉除最後一個以外的 segment）。**正規化**則逐 segment 進行，與 AutoRL 的統計量一致。三個 scope 的 group key 都含 segment index。 |
| 空組 / Empty group | 組內完全沒有非零 reward 時整組維持 0（不產生 nan / RuntimeWarning）。AutoRL 只取整批所以碰不到；`task` scope 早期訓練很常見。 |

### 仍待決 / Still open

- **場景廣播 / Scene broadcast**（§5）— 尚未移植。目前 `task` scope 給的是
  *task-conditioned* baseline（同指令、同物件、同背景、**不同初始姿態**），不是 same-state group。
  Not yet ported: `task` scope currently gives a task-conditioned baseline, not a same-state group.

---

## 9. 是否需要 `/std`：取捨 / Whether to divide by std

**中文** — `--grpo-std-scope {group, global, none}`。這一節是那個旗標的依據。

**English** — The rationale behind `--grpo-std-scope {group, global, none}`.

### 9.1 問題：per-group std 同時也是 per-group 權重 / std doubles as a weight

**中文** — `A = (r − mean_g) / (std_g + 1e-5)`。除以 `std_g` 不只是縮放，它還把**組內離散度**變成該組的
梯度權重：離散度小的組被放大。這正是 Dr. GRPO 指出的 difficulty bias。

**English** — Dividing by `std_g` is not just scaling: it turns the group's *dispersion* into that
group's gradient weight, amplifying low-dispersion groups. This is Dr. GRPO's difficulty bias.

用 CRONOS 的 reward 量級實測（非零 `r_t` ∈ {0.1, 0.2, 1.0}，G=4）/
Measured at CRONOS's reward scale:

| 組內非零 `r_t` | `std_g` | `max\|A\|` (÷std_g) | 相對權重 | `max\|A\|` (不除) | 相對權重 |
|---|---|---|---|---|---|
| 1/4 成功 `[1.0, .1, .1, .1]` | 0.3897 | 1.732 | **1.00×** | 0.675 | 1.00× |
| 2/4 成功 `[1.0, 1.0, .1, .1]` | 0.4500 | 1.000 | **0.58×** | 0.450 | 0.67× |
| 3/4 成功 `[1.0, 1.0, 1.0, .1]` | 0.3897 | 1.732 | **1.00×** | 0.675 | 1.00× |
| 全同 `[.1, .1, .1, .1]` | 0.0000 | 0.000 | 0.00× | 0.000 | 0.00× |

**中文** — 除以 `std_g` 讓**最不平衡**的組（1/4、3/4）拿到最大權重，而訊號最豐富的平衡組（2/4）
只有 0.58×。這是反的。不除的話權重只反映實際 reward 差距。

**English** — Per-group std gives the *most imbalanced* groups the largest weight and the most
informative balanced group only 0.58×. That is backwards. Without it, the weight tracks the actual
reward gap.

### 9.2 但完全不除會改變梯度尺度 / but dropping it changes the gradient scale

實測（`training/buffer.py` 的三個模式，同一份 fixture）/ measured on one fixture:

| `std_scope` | Σ\|advantage\| | 說明 / Note |
|---|---|---|
| `group` | 30.78 | AutoRL / 教科書 GRPO |
| `global` | 30.92 | 尺度幾乎不變，偏誤消失 / scale preserved, bias gone |
| `none` | 10.53 | **約 1/3**，等同把 policy 的有效學習率砍成三分之一 / ~3× smaller effective LR |

### 9.3 結論 / Conclusion

| `grpo_group_scope` | 建議 `std_scope` | 理由 / Reason |
|---|---|---|
| `batch` | **`group`** | 組就是整個 update（數千個非零項），`std` 只是一個穩定的全域尺度因子，不構成 per-group 權重；而且這才是 AutoRL 的行為，是數值對齊的前提。With one batch-wide group the std is a stable global scale factor, not a per-group weight — and it is what AutoRL does. |
| `scene` / `task` | **`global`** | 每組只有 4–32 個 env，`std_g` 既噪聲大又帶 §9.1 的偏誤；`global` 保留尺度、去掉偏誤。At 4–32 envs per group `std_g` is noise-dominated *and* biased; `global` keeps the scale and drops the bias. |
| 任何 / any | `none` | 只在想完全排除尺度因素的消融實驗使用，記得同步調高 `--vla-lr`。Ablation only; compensate with `--vla-lr`. |

`scripts/train.sh` 的 `grpo` / `grpo-scene` / `grpo-task` 三個模式已分別預設成
`group` / `global` / `global`，可用 `GRPO_STD_SCOPE=` 覆寫（覆寫時會加進 RUN_TAG，所以不會蓋到別的 run）。
Wired into `scripts/train.sh`; override with `GRPO_STD_SCOPE=`, which is reflected in the run tag.

**注意 / Caveat** — `group` 與 `global` 在 `grpo_group_scope=batch` 底下是**同一件事**（組就是全體），
所以那個組合只有 `none` 是真正不同的選項。
Under `batch` scope `group` and `global` are identical by construction.

---

## 10. 三層分組 / The three grouping levels

**中文** — `--grpo-group-scope` 的三個值對應 CRONOS 既有的三層巢狀結構，由寬到窄。
下表以 `four_group_sequential_2x2` 為例（64 envs = 4 個 YAML group × 16 envs，每組 4 個 unique task），
一次 PPO update 涵蓋 2 個 segment，共 128 個 buffer slot。

**English** — The three values map onto CRONOS's existing nesting, widest to narrowest. Numbers below
are for `four_group_sequential_2x2`; one PPO update spans 2 segments = 128 buffer slots.

**三層一律以 segment 為界**，group key 是 `(segment, block)`。
All three are scoped to one segment; the key is `(segment, block)`.

| `scope` | 組大小 | 組/segment | 組/update | 組內共享 / Shared within a group | 對應 |
|---|---|---|---|---|---|
| `batch` | 64 | 1 | 2 | 什麼都不保證 —— 4 種物件、4 種背景、4 種任務混在一起 / nothing guaranteed | AutoRL 的統計量 |
| `scene` | 16 | 4 | 8 | 物件、容器、背景相同；**任務可能不同** / same objects, receptacles, overlay; tasks may differ | YAML `groups:` |
| `task` | 4 | 16 | 32 | 物件、容器、背景、**任務**全部相同 / everything above plus the same task | fan-out 子區段 |

`batch` 的組是 **`num_envs` 條軌跡**，不是整個 `ppo_update_len` 視窗 —— 因為 AutoRL 的
`train_grpo(self.buffer)` 吃的 buffer 就只裝一個 segment（`ep_len = episode_len`，`num_env = num_envs`），
所以 `compute_returns_grpo` 的統計量本來就是每 `num_envs` 條算一次。把整個視窗（預設 2 個 segment）
pool 在一起會改變統計量：實測在兩個 segment 的 reward 尺度不同時，returns 最大差 1.6。
`batch` groups `num_envs` trajectories, not the whole `ppo_update_len` window: AutoRL's
`train_grpo` consumes a buffer holding exactly one segment, so its statistic is per-`num_envs`.
Pooling the window changes it — measured max difference 1.6 when the two segments differ in scale.

CRONOS 仍然用整個視窗的資料訓練（比 AutoRL 多，AutoRL 的 GRPO 丟掉除了最後一個以外的 segment），
只是每個 segment 各自正規化。
CRONOS still *trains* on the whole window — more data than AutoRL's GRPO, which discards all but the
last segment — it just normalizes each segment independently.

`task` 的組大小是 `group_num_envs / n_unique_tasks`，隨 config 變動：
`four_group_sequential_2x2` 是 4、`two_group_sequential_2x2` 是 8、`one_group_*_2x2` 是 16、
`one_group_half_train_2x2` 是 32。`main.py` 啟動時會印出實際的組大小，小於 8 會警告。
Group width for `task` depends on the config (4 / 8 / 16 / 32 across the shipped ones); `main.py`
prints the real sizes at startup and warns below 8.

### 為什麼 key 一定要含 segment index / Why the key must include the segment

`TaskScheduler.update_index` 每個 segment 旋轉 `_fan_out_offsets`，所以同一個 task block 的
**成員固定但任務會換**。實測（`four_group_sequential_2x2`，group_A）/ measured:

```
env  0 (task block 0): seg0 = put obj1 on recep1   ->   seg1 = put obj1 on recep2
env  4 (task block 1): seg0 = put obj1 on recep2   ->   seg1 = put obj2 on recep1
env  8 (task block 2): seg0 = put obj2 on recep1   ->   seg1 = put obj2 on recep2
env 12 (task block 3): seg0 = put obj2 on recep2   ->   seg1 = put obj1 on recep1
```

若 key 只用 block id，seg0 的 `put obj1 on recep1` 就會跟 seg1 的 `put obj1 on recep2` 被當成同一組
互相比較。`scene` 其實沒有這個問題（一個 scene block 每個 segment 跑的任務*集合*不變），`batch` 也沒有，
但三者仍一律以 `(segment, block)` 為 key —— 一致，而且在非 episodic 連續性下 segment N+1 的起始狀態是
segment N 的結束狀態，兩者本來就不同分佈。

Keying on block id alone would compare `put obj1 on recep1` against `put obj1 on recep2`. `scene`
does not strictly need the segment (its task *set* is stable) but uses it anyway, for consistency and
because under non-episodic continuity segment N+1 starts from segment N's end state.

---

## 8. 重跑驗證 / Re-verification

不需要 GPU / No GPU required:

```bash
# §4.1 的 ×act_dim：重現 [B,1] → [B,7] 廣播與 sum(dim=-1) 的交互作用
python3 - <<'PY'
import numpy as np
alp = np.zeros((2, 3, 7), np.float32)
alp[0] = np.array([[-1.], [-2.], [-3.]], np.float32)      # replay_buffer.py:74
print(alp[0])                                              # 每列 7 個相同值 / 7 identical columns
old, new = alp[0], np.array([[-.9], [-2.1], [-3.2]], np.float32)
r = np.exp(new - old); s = np.minimum(r, np.clip(r, .8, 1.2))
print("grpo/ppo =", (-s.sum(-1, keepdims=True).mean()) / (-s.mean()))
PY

# §2 的 buffer 分歧：確認兩條路徑吃不同物件
grep -n "train_ppo(self\|train_grpo(self" AutoRL/SimplerEnv/simpler_env/train_ms3_ppo.py

# §5 的場景廣播：確認兩邊條件相反
grep -n 'obj_set == "fixed"' AutoRL/ManiSkill/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/pick_place_multi.py
grep -n 'obj_set != "fixed"' Benchmark/CRONOS/envs/bridge_multi.py
```
