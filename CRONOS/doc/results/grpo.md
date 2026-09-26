# GRPO in CRONOS — review and failure analysis

Index of the documentation: [`../README.md`](../README.md).

**Summary.** `--alg-name grpo` is implemented to match AutoRL bit-for-bit (Part 1),
and it collapses: in every measured run grasp and success fall to ≈ 0 while the PPO
controls improve (Part 2, and Q5 in [`paper_experiments.md`](paper_experiments.md)).
The cause is the objective itself — per-step normalization with no group baseline,
and `alg_grpo_fix` leaving idle trajectories at zero advantage — not a CRONOS port
error. The trajectory-level fix proposed in Part 2 §6 is **not implemented in V0.99**;
treat GRPO as experimental and do not cite GRPO-vs-PPO numbers.

Part 2 §5 corrects three claims of Part 1 (§4.1, §5, §6); Part 1 is kept unedited
as the record of what was believed at V0.93.

---

## Part 1 — AutoRL GRPO review (V0.93)

*Originally `grpo_autorl.md`: AutoRL GRPO — 現有實作整理 / Existing implementation review.*


**範圍 / Scope:** `AutoRL/SimplerEnv/simpler_env/` 的 GRPO 路徑，唯讀盤點，為「CRONOS 沿用 AutoRL
GRPO」做準備。AutoRL 未被修改。
The GRPO path in `AutoRL/SimplerEnv/simpler_env/`, read-only, in preparation for reusing it in
CRONOS. Nothing in AutoRL was modified.

**狀態 / Status:** 純程式碼閱讀 + numpy 重現，未在硬體上跑過（本機沒有能載入 VLA 的 GPU）。
所有數值宣稱下面都附了可重跑的驗證方式。
Code reading plus a numpy reproduction; not run on hardware (no GPU here large enough for a VLA).
Every numeric claim below states how to re-verify it.

---

### 1. 完整呼叫鏈 / The full call chain

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

### 2. ⚠️ GRPO 與 PPO 吃的不是同一個 buffer / The two paths consume different buffers

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

### 3. `compute_returns_grpo` 的實際數學 / The actual math

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

### 4. `train_grpo_step` 與 `train_ppo_step` 的差異 / Diff against the PPO step

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

#### 4.1 ⚠️ `sum(dim=-1)` 造成 policy loss 被放大 `act_dim` 倍 / a ×`act_dim` scale factor

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

### 5. 讓它成為「group」的其實是環境端 / What actually makes it a group

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

### 6. 盤點結果 / Findings summary

| # | 發現 / Finding | 嚴重度 / Severity |
|---|---|---|
| 1 | GRPO 訓練 `self.buffer`（1 個 segment），PPO 訓練 `prealloc_buffer`（N 個 segment）。預設下 GRPO 每次更新資料量減半 / GRPO trains on half the data | 高 / high — 影響任何 PPO-vs-GRPO 比較 |
| 2 | `sum(dim=-1)` × `[B,1]→[B,7]` 廣播 → policy loss ×`act_dim`（OpenVLA 7、SpatialVLA 3）/ ×`act_dim` policy loss | 高 / high — 等同 7× 有效學習率 |
| 3 | `compute_returns_grpo` 完全不分組；「組」實際上來自環境端的場景廣播 / grouping is environmental, not algorithmic | 中 / medium — 移植時最容易漏掉的一環 |
| 4 | 無折扣（γ=1），`buffer_gamma` 在 GRPO 路徑失效 / undiscounted; `buffer_gamma` inert | 低 / low — 但要寫進文件避免誤解 |
| 5 | 無 baseline、無 KL-to-reference 項（標準 GRPO 有 KL 正則）/ no baseline, no KL term | 低 / low — 設計選擇，非錯誤 |
| 6 | `lambda_bc` / `bc_loss` 是死碼：只在 `train_ppo_step` 的 `demo_batch is not None` 分支，而沒有呼叫端傳 `demo_batch` / dead code | 低 / low |

---

### 7. 沿用到 CRONOS 的對應 / Mapping onto CRONOS

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

#### 已實作 / Implemented

| 項目 / Item | 決定 / Decision |
|---|---|
| 分組 / Grouping | `--grpo-group-scope batch\|scene\|task` —— 三層由寬到窄，見 §10。`batch` = AutoRL（已驗證與 `compute_returns_grpo` **逐元素相同**，`fix` 兩種分支皆是）。 |
| §4.1 的 ×`act_dim` | **不重現**。CRONOS 的 `action_log_probs` 寬度本來就是 1，那個廣播在此不可能發生；`training/grpo.py` 用 `.mean()`。**後果：CRONOS 的 GRPO loss 數值是 AutoRL 的 1/`act_dim`，跨 codebase 請比 grad-norm 不要比 loss。** |
| 更新視窗 / Update window | **訓練資料**沿用 CRONOS 的整個 `ppo_update_len` 視窗（預設 2 個 segment），比 AutoRL 多（它的 GRPO 丟掉除最後一個以外的 segment）。**正規化**則逐 segment 進行，與 AutoRL 的統計量一致。三個 scope 的 group key 都含 segment index。 |
| 空組 / Empty group | 組內完全沒有非零 reward 時整組維持 0（不產生 nan / RuntimeWarning）。AutoRL 只取整批所以碰不到；`task` scope 早期訓練很常見。 |

#### 仍待決 / Still open

- **場景廣播 / Scene broadcast**（§5）— 尚未移植。目前 `task` scope 給的是
  *task-conditioned* baseline（同指令、同物件、同背景、**不同初始姿態**），不是 same-state group。
  Not yet ported: `task` scope currently gives a task-conditioned baseline, not a same-state group.

---

### 9. 是否需要 `/std`：取捨 / Whether to divide by std

**中文** — `--grpo-std-scope {group, global, none}`。這一節是那個旗標的依據。

**English** — The rationale behind `--grpo-std-scope {group, global, none}`.

#### 9.1 問題：per-group std 同時也是 per-group 權重 / std doubles as a weight

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

#### 9.2 但完全不除會改變梯度尺度 / but dropping it changes the gradient scale

實測（`training/buffer.py` 的三個模式，同一份 fixture）/ measured on one fixture:

| `std_scope` | Σ\|advantage\| | 說明 / Note |
|---|---|---|
| `group` | 30.78 | AutoRL / 教科書 GRPO |
| `global` | 30.92 | 尺度幾乎不變，偏誤消失 / scale preserved, bias gone |
| `none` | 10.53 | **約 1/3**，等同把 policy 的有效學習率砍成三分之一 / ~3× smaller effective LR |

#### 9.3 結論 / Conclusion

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

### 10. 三層分組 / The three grouping levels

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

#### 為什麼 key 一定要含 segment index / Why the key must include the segment

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

### 8. 重跑驗證 / Re-verification

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


---

## Part 2 — Why GRPO collapses (V0.95)

*Originally `grpo_failure.md`: GRPO 失效診斷 / Why GRPO collapses.*

Companion: [Part 1](#part-1--autorl-grpo-review-v093) — 該文件的 §4.1、§5、§6 有三處需要修正，見本文 §5。
Three claims in that document need correcting; see §5 below.

**範圍 / Scope:** `--alg-name grpo` 在 CRONOS 上為何不但學不起來、還會把 SFT 初始策略破壞掉。
包含七個 GRPO run 的實測、與 `$AUTORL_ROOT` 的逐項對拍，以及修法建議。
Why `--alg-name grpo` not only fails to learn but actively destroys the SFT initialization.
Covers seven measured GRPO runs, an item-by-item comparison against `$AUTORL_ROOT`,
and the recommended fix.

**狀態 / Status:** 本文的每一項數值都來自 (a) 真實 run 的 `glob/` 產出，或 (b) 直接 import
`training/buffer.py` 的 `CronosReplayBuffer` 跑出來的數值重現——**不是**重寫的近似實作。
§7 給了可重跑的腳本。唯一未在硬體上驗證的是 §6 的修法建議。
Every number below comes from (a) a real run's `glob/` output, or (b) a numeric reproduction that
imports the actual `CronosReplayBuffer` from `training/buffer.py` — **not** a reimplementation.
§7 gives re-runnable scripts. The only thing not confirmed on hardware is the fix proposed in §6.

---

### 1. 結論 / Findings

| # | 發現 / Finding | 嚴重度 / Severity |
|---|---|---|
| 1 | 這個實作**沒有 group baseline**。正規化作用在**單步 reward** 上再累積，不是 GRPO 的「每條軌跡算 return、組內比較」。/ No group baseline: normalization is applied per-step then accumulated, not per-trajectory. | **致命 / fatal** |
| 2 | `fix=True` 把零 reward 排除在統計外**也不縮放**，於是「全程沒碰到物體」的軌跡 advantage 恆為 0——**永遠不會被懲罰**。/ `fix=True` exempts inaction from the baseline: an all-zero trajectory gets advantage exactly 0, forever unpunished. | **致命 / fatal** |
| 3 | 綜合 1+2，目標函數的排序是 `success > 不作為 > 嘗試但失敗`。由於 `P(success \| 嘗試)` ≈ 2–20%，「嘗試」的期望梯度為負，**唯一穩定解是停止互動**。/ The objective ranks `success > inaction > attempt-and-fail`; inaction is the unique stable optimum. | **致命 / fatal** |
| 4 | 非零 reward 的均值 μ ≈ 0.04–0.12，恰好就是 grasp 事件的 +0.1。均值相減把唯一稠密的 shaping 訊號抵銷掉，有時還翻轉符號。/ μ(non-zero) ≈ the grasp reward itself, so centering annihilates the only dense signal. | 高 / high |
| 5 | 98.8% 的樣本 advantage 為 0；**41–92% 的 optimizer step 收到的有效樣本數是 0**（中位數 0/160）。/ 98.8% of samples are dead; 41–92% of optimizer steps see zero live samples. | 高 / high |
| 6 | 上述 1–4 **AutoRL 完全相同**（數值 bit-identical）。切回 AutoRL 精確設定救不了。/ All of the above are shared with AutoRL bit-for-bit; reverting to AutoRL's exact setting does not help. | — |
| 7 | AutoRL 的 GRPO 路徑**從未跑過**（41 個 wandb run 全是 PPO）。所謂「對齊 AutoRL」對齊的是未驗證的程式碼。/ AutoRL's GRPO path was never run; "matching AutoRL" matched unvalidated code. | — |

---

### 2. 實驗證據 / The empirical evidence

#### 2.1 七個 GRPO run 全部崩潰 / All seven runs collapse

七個 run 全用 `grpo_group_scope=scene`、`grpo_std_scope=global`、`alg_grpo_fix=True`、
`num_envs=64`、`buffer_minibatch=8`、`alg_gradient_accum=20`、`alg_ppo_epoch=1`、
`alg_entropy_coef=0.0`、`vla_lr=1e-4`、`vla_grad_norm=10.0`。
`compute_grpo_returns` 自 `e65d33f` (V0.93) 起未再改動，七個 run 的 GRPO 數學完全相同。
All seven share the settings above; the GRPO math has not changed since `e65d33f`, so all seven
runs used identical code.

（`grasp` = `is_src_obj_grasped`，取前 1/5 與後 1/5 episode 的平均 / mean over the first and last
fifth of episodes）

| dir / run | T | seed | episodes | grasp | success | 最後一個 episode 的 grasp |
|---|---|---|---|---|---|---|
| `GRPO/41unprib` | 2560 | 0 | 1–4 | 0.0845 → 0.0347 | 0.1338 → 0.0059 | 0.0347 |
| `GRPO/80pznnd9` | 2560 | 1 | 1–4 | 0.0806 → **0.0005** | 0.1025 → 0.0005 | 0.0005 |
| `GRPO_80/fdh5hwcw` | 80 | 0 | 1–45 | 0.2448 → 0.0243 | 0.0434 → 0.0017 | 0.0312 |
| `GRPO_80/0fddmf8i` | 80 | 0 | 33–121 | 0.0165 → 0.0129 | 0.0028 → 0.0009 | 0.0312 |
| `GRPO_80/5f7llcd0` | 80 | 0 | 97–136 | 0.0117 → **0.0000** | 0.0000 → 0.0039 | **0.0000** |
| `GRPO_80/jae20zns` | 80 | 1 | 1–46 | 0.1823 → 0.1059 | 0.0156 → 0.0000 | 0.1250 |
| `GRPO_80/5st8d9e6` | 80 | 1 | 33–131 | 0.0995 → 0.0469 | 0.0000 → 0.0025 | 0.0312 |

**中文** — `GRPO_80` 的 seed0 是一條三段續跑的鏈（ep1–45 → ep33–121 → ep97–136），
grasp 從 0.328（ep1）單調掉到 0.0000，最後 20 個 episode 幾乎每個都是 0。seed1 是兩段鏈，
從 0.250 掉到 0.047。**grasp 在 7/7 個 run 下降。** `GRPO/41unprib` 的最終 in-domain eval
`mean_success = 0`。

**English** — `GRPO_80` seed0 is a three-run chain; grasp falls monotonically from 0.328 to
0.0000, with almost every one of the last 20 episodes at exactly zero. **Grasp declines in 7/7
runs.** The final in-domain eval of `GRPO/41unprib` reports `mean_success = 0`.

#### 2.2 PPO 對照組：同環境、同 config / The PPO control

| | grasp 首 → 末 | 倍率 |
|---|---|---|
| 8 個 ep1–4 的 PPO run（HSR/LSR/PER/noep × 2 seeds） | 0.106–0.149 → 0.127–0.435 | **7/8 上升，1.47–3.22×**；1 個持平（0.94×） |
| 3 個 ep5–7 的續跑 run | 起點已在 0.44–0.72 | 高檔持平（0.94–1.15×） |
| **7 個 GRPO run** | 0.012–0.245 → **0.000–0.106** | **7/7 下降** |

**中文** — PPO 在同一個環境、同一份 reward、同一個 SFT 起點下，把 grasp 率推到 SFT 基線的
1.5–3.2 倍並在 0.44–0.72 高檔穩住；GRPO 則無一例外地把它推向 0。**這不是環境或 reward 的問題，
是 GRPO 路徑特有的。** 注意 success 率在 PPO 下也不是單調上升（LSR 甚至下降），所以此處用 grasp
——reward shaping 直接作用的行為——作為判準。

**English** — Under an identical environment, reward and SFT starting point, PPO raises grasp to
1.5–3.2× the baseline and holds it at 0.44–0.72; GRPO drives it to zero in every run. The failure
is specific to the GRPO path. Note that success is *not* monotone under PPO either (LSR declines),
which is why grasp — the behaviour the shaping term acts on directly — is the discriminator here.

#### 2.3 訊號飢餓 / Signal starvation

取自各 run 的 `glob/ppo_log.txt`（`training/grpo.py` 每個 minibatch 印一行）：

| run | zero-adv frac (mean) | 100% 死掉的 minibatch | 每 minibatch 有效樣本 | **0 有效樣本的 optimizer step** |
|---|---|---|---|---|
| `GRPO/41unprib` | 0.9878 | 92.4% | 0.097 / 8 | 68.9%（中位數 0/160） |
| `GRPO/80pznnd9` | 0.9946 | 96.4% | 0.043 / 8 | 84.6%（0/160） |
| `GRPO_80/fdh5hwcw` | 0.9587 | 76.0% | 0.330 / 8 | 41.0%（4/160） |
| `GRPO_80/jae20zns` | 0.9906 | 94.3% | 0.075 / 8 | 83.6%（0/160） |
| `GRPO_80/0fddmf8i` | 0.9981 | 98.6% | 0.015 / 8 | 89.2%（0/160） |
| `GRPO_80/5st8d9e6` | 0.9969 | 97.9% | 0.025 / 8 | 92.2%（0/160） |
| `GRPO_80/5f7llcd0` | 0.9986 | 99.0% | 0.011 / 8 | 92.4%（0/160） |

**中文** — `alg_entropy_coef = 0`，所以 advantage 為 0 的樣本 loss 恰好是 0，梯度也**恰好**是 0
（`-min(r·0, clamp(r)·0)` 對 `r` 的導數是 `-0`）。配上 `minibatch=8 × gradient_accum=20`，
模型走了數千個完全沒有梯度的 no-op step，再被少數幾個極大、極吵的梯度推一把——
而 `clip_grad_norm_(…, 10.0)` 讓那少數幾步的方向完全主導更新。

**English** — With `alg_entropy_coef = 0`, a zero-advantage sample contributes exactly zero
gradient. Combined with `minibatch=8 × gradient_accum=20`, the model takes thousands of no-op steps
punctuated by a handful of very large, very noisy ones, whose direction then dominates the update.

---

### 3. 機制 / The mechanism

#### 3.1 沒有 group baseline / There is no group baseline

`training/buffer.py::compute_grpo_returns` 做的是：把**個別 reward 事件**在組內做
`(r − μ) / σ`，再沿時間累積成 undiscounted reward-to-go，然後 `advantages = returns`。

```python
stat_mask = (rewards != 0) if fix else np.ones_like(rewards, dtype=bool)
...
sub -= vals.mean()          # 只作用在非零項 / non-zero entries only
sub /= (vals.std() + 1e-5)
...
acc = norm[step] + self.masks[step + 1, :n] * acc
self.advantages[:, :n] = self.returns[:, :n]
```

**中文** — GRPO 的定義性機制是「同一個 prompt 取樣 G 條軌跡，用**組內 return 的平均**當 baseline」。
這裡沒有這一層：被減掉的 μ 是**跨時間、跨 env 的單步 reward 均值**，不是軌跡 return 的組平均。
Part 1 §3 已經指出「這其實是 batch-normalized undiscounted REINFORCE」——本文要補的是
它的後果比當時估計的嚴重得多。

**English** — GRPO's defining mechanism is a baseline formed from the *group mean of trajectory
returns*. That layer is absent: μ is the mean of individual per-step rewards pooled over time and
envs. Part 1 §3 already called this "batch-normalized undiscounted REINFORCE"; what
follows is why that is fatal rather than merely imprecise.

#### 3.2 `fix=True` 讓「不作為」豁免於 baseline / Inaction is exempt

**中文** — `stat_mask = (rewards != 0)`：零項既不參與統計，**也不被平移縮放**。
於是一條「從頭到尾沒碰到物體」的軌跡，每一步 reward 都是 0 → 累積出的 return 每一步都是 0
→ advantage 恆為 0 → 梯度恆為 0。**它永遠不可能被懲罰。**
正確的 GRPO 會給它 `(0 − mean_g R)/σ_g < 0`（只要組內有任何一條做得比較好）。

**English** — Zeros neither enter the statistics nor get rescaled, so a trajectory that never
touches the object has advantage exactly zero at every step and is structurally immune to
punishment. Correct GRPO would assign it `(0 − mean_g R)/σ_g < 0` whenever any peer did better.

#### 3.3 均值相減抵銷 grasp 訊號 / Centering annihilates the grasp signal

**中文** — reward 是 potential 差分（`envs/reward.py`），Φ = `0.1·grasped + 0.1·consec + 1.0·(success ∧ grasped)`，
所以最常見的非零事件就是 **±0.1**（grasp 開關）。實測非零 reward 的均值 μ 落在 0.04–0.12，
**和 grasp 事件本身同量級**：

| 訓練階段 | μ (非零 reward 均值) | grasp 事件 +0.1 正規化後 |
|---|---|---|
| 早期 (grasp 30%) | +0.0681 | +0.0319 |
| 中期 (grasp 10%) | +0.1200 | **−0.0200（符號翻轉）** |
| 後期 (grasp 3%) | +0.0400 | +0.0600 |

於是「教它去抓」的那一項被減成 ~0 甚至變負，而 ±0.2（掉落）、±1.0（成功）這些稀有事件
完整保留。**扣分的留著，加分的被消掉。**

**English** — The reward is a potential difference, so its most common non-zero event is exactly
±0.1 (the grasp toggle). Measured μ sits at 0.04–0.12 — the same magnitude — so centering shrinks
the grasp term to ~0 and at mid-training flips its sign, while the rarer ±0.2 (drop) and ±1.0
(success) events survive at full magnitude. The penalty survives; the reward does not.

#### 3.4 結果：目標函數獎勵「不要動」/ The objective rewards inaction

用**真實的** `CronosReplayBuffer.compute_grpo_returns` 跑一份符合實測事件率的 reward
（grasp 30%、`P(success|grasp)` 20%、scene 分組 16 envs、`std_scope=global`），
每條軌跡的平均 advantage：

| 軌跡結果 | 現行實作 | 正確的 trajectory-level GRPO |
|---|---|---|
| **idle** — 全程沒碰到物體 | **+0.000（恆為 0）** | **−0.462**（被懲罰） |
| **hold** — 抓到並抓住，未完成 | −0.141 | **+0.228** |
| **drop** — 抓到又掉 | **−0.859** | −0.375 |
| **success** | +0.952 | +2.294 |
| 死樣本比例 / dead fraction | 78.1% | 0.0% |

**中文** — 兩個符號都是反的：**不作為完全免罰，而「嘗試但失敗」被重罰。**
由於 `P(success | 嘗試)` 只有 2–20%（且隨訓練下降），「嘗試」的期望 advantage 是負的。
對這個目標函數做梯度上升，**唯一穩定解就是停止與物體互動**——這正是 §2.1 觀察到的 grasp → 0.0000。
這是一個自我強化的迴圈：grasp 越少 → 非零 reward 越少 → 死樣本越多 → 訊號越噪 → grasp 更少。

**English** — Both signs are inverted: inaction is unpunished, attempting and failing is heavily
punished. With `P(success | attempt)` at 2–20% and falling, the expected advantage of attempting is
negative, so the unique stable optimum of this objective is to stop interacting with the object —
exactly the grasp → 0.0000 observed in §2.1. The loop is self-reinforcing: fewer grasps → fewer
non-zero rewards → more dead samples → noisier signal → fewer grasps.

---

### 4. 與 AutoRL 的對照 / Comparison against AutoRL

比對對象 / Compared against: `$AUTORL_ROOT/SimplerEnv/simpler_env/`
（`utils/replay_buffer.py:109`、`policies/openvla/openvla_train.py:411,539`、`train_ms3_ppo.py:271`）

#### 4.1 完全一致的部分 / Bit-identical

| 項目 / Item | 結果 / Result |
|---|---|
| `compute_returns_grpo` 數學 | `group_ids=None, std_scope="group"` 下 **max\|AutoRL − CRONOS\| = 0.000e+00**，`fix=True`/`False` 皆然 |
| reward function | `RewardShaper.compute_reward` 與 `env/simpler_wrapper.py:52 get_reward` 逐行相同 |
| clip / grad_norm / entropy / epoch / accum | 0.2 / 10.0 / 0.0 / 1 / 20（`openvla_train.py:295-297`、`train_ms3_ppo.py:112-116`），全同 |
| 只 clip + step `params_vla`，無 value loss | 相同 |
| gradient accum 邊界 | `idx % a == a-1 or idx == total-1` ≡ CRONOS 的 `(idx+1) % a == 0 or (idx+1) == total` |
| minibatch generator（含 `indices % n_rollout_threads` 的 instruction 索引） | 相同 |
| pose 隨機化 | `bridge_multi.py:1018` 與 `pick_place_multi.py:1529` **逐字相同**（含 `rand_8` 的 `b//8` + `repeat(8)`） |

#### 4.2 真正的差異 / Actual divergences

| 項目 | AutoRL | CRONOS | 影響 |
|---|---|---|---|
| 分組 / grouping | 無（整批） | `--grpo-group-scope {batch,scene,task}` + `--grpo-std-scope` | 七個 run 都用 `scene`+`global`，**不是 AutoRL 語意** |
| update window | GRPO 只吃 `self.buffer` = 1 segment（`train_ms3_ppo.py:280`） | 整個 `ppo_update_len` 視窗 | `GRPO_80`（`ppo_update_len=80=segment_len`）**恰好等同 AutoRL**；`GRPO` T2560（160）是 2 segment |
| 歸約 / reduction | `.sum(dim=-1, keepdim=True).mean()` | `.mean()` | **數值相同**，見 §5.1 |

#### 4.3 切回 AutoRL 設定救不了 / Reverting to AutoRL's setting does not help

同一份 reward 下 / On one fixture:

| 設定 | 與 AutoRL 的 max diff | zero-adv frac |
|---|---|---|
| AutoRL（`batch` + `group`） | 0（定義上） | **81.09%** |
| 七個 run 實際用的（`scene` + `global`） | 0.6061（corr 0.9897） | **82.71%** |

**中文** — 差異微不足道。`scene` 分組（16 envs）確實讓 dead group 稍微更常見，但那是**加重因子，
不是根因**。根因（§3.1–3.4）是雙方共有的那段數學。

**English** — The divergence is marginal. The narrower `scene` grouping makes dead groups slightly
more frequent, but it is an aggravating factor, not the cause; the cause is the math both share.

---

### 5. Part 1 需要修正的三處 / Three corrections to Part 1

#### 5.1 §4.1 / Finding #2 的 ×`act_dim` 不適用於 `$AUTORL_ROOT`

**中文** — 文件說 `replay_buffer.py:20` 把 `action_log_probs` 配置成寬度 `act_dim`，
使 `[B,1]` 被廣播成 `[B,7]`、`sum(dim=-1)` 把 policy loss 放大 7 倍，並列為「高」嚴重度。
**在 `$AUTORL_ROOT` 不成立**：三個 buffer class 的 `action_log_probs` 都是**寬度 1**
（`replay_buffer.py:20`、`:217`、`:273`），且 `evaluate_actions` 有
`assert logprobs.shape[1] == 1`（`openvla_train.py:234`）。
`.sum(dim=-1, keepdim=True)` 作用在 `[mb,1]` 上是 **no-op**，與 `.mean()` 完全相同。

寬度 `act_dim` 的版本在 **`$RL4VLA_ROOT/SimplerEnv/simpler_env/utils/replay_buffer.py:18`**
——文件把 RL4VLA 的 bug 記到 AutoRL 頭上了。連帶地 `training/grpo.py` docstring 那句
「CRONOS GRPO losses are `act_dim` times smaller than AutoRL's … Compare gradient norms, not raw
loss values」也不成立：**兩邊 loss 可以直接比較。**

**English** — The ×`act_dim` inflation does not exist in `$AUTORL_ROOT`: all three buffer
classes allocate `action_log_probs` at width 1, and `evaluate_actions` asserts `[B,1]`, so
`sum(dim=-1)` is a no-op equal to `.mean()`. The width-`act_dim` allocation is in **RL4VLA**
(`replay_buffer.py:18`); the bug was misattributed. The note in `training/grpo.py`'s docstring is
consequently wrong — the two losses are directly comparable.

#### 5.2 §5「條件相反」是誤讀，且結論相反 / The "inverted condition" claim

**中文** — 文件引用 `pick_place_multi.py:719/1059/1102/1474` 作為 pose 重抽的四個位置，
並說「四個 env 變體極性一致」。實際上 **719/1059/1474/1976 是選資產清單長度的分支**
（`lc`/`lp`/`le`），不是 pose。真正的 pose 分支是：

| AutoRL env | pose 分支 | 預設 `obj_set="rand"` 下 |
|---|---|---|
| `TwoObjectOneReceptacle` | :753（無分支） | 沿用廣播 → **同構** |
| `OneObjectTwoReceptacle` | :1102 `== "fixed"` | 不重抽 → **同構** |
| `TwoObjectTwoReceptacle` | **:1529 `!= "fixed"`** | 每 env 重抽 → **異構** |
| `ThreeObjectThreeReceptacle` | **:2035 `!= "fixed"`** | 每 env 重抽 → **異構** |

四個變體極性**並不一致**。CRONOS 用的是 `PickPlaceNxM-v1`（`bridge_multi.py:1301`），
對應 `TwoObjectTwoReceptacle`，而 `bridge_multi.py:1018` 與 `pick_place_multi.py:1529` 逐字相同。
**那句 "matching AutoRL exactly" 的註解是對的。**

⇒ 「scene broadcast 未移植」這個 open item，對 2x2 config 而言**不是 CRONOS 與 AutoRL 的差異**：
AutoRL 自己的 2x2 env 一樣是異構的。也就是說 §5 那個「AutoRL 整批同場景，所以 batch norm ≡
group norm」的論證，**對 AutoRL 自己的 2x2 env 也不成立**。

**English** — The cited lines are asset-list-size branches, not pose branches. The real pose
branches split two ways across the four variants, and CRONOS's `PickPlaceNxM-v1` matches its
counterpart (`TwoObjectTwoReceptacle`, `:1529`) verbatim — the "matching AutoRL exactly" comment is
correct. Consequently the "scene broadcast not ported" open item is **not** a CRONOS-vs-AutoRL
divergence for the 2x2 config: AutoRL's own 2x2 env is equally heterogeneous, so §5's argument that
"AutoRL runs one scene batch-wide, therefore batch norm ≡ group norm" fails for AutoRL too.

#### 5.3 AutoRL 的 GRPO 路徑從未跑過 / AutoRL never ran GRPO

**中文** — `$AUTORL_ROOT/wandb` 有 41 個 run，能讀出 `alg_name` 的 35 個**全部是 `ppo`**，
另 6 個是空的或中斷的 run。config 裡出現的 `grpo` 字串都是 `alg_grpo_fix` 這個 key。
**沒有任何一個 GRPO run。** 所以「對齊 AutoRL」對齊的是一段從未產生過結果的程式碼——
忠實移植從來就不等於正確。

**English** — Of 41 wandb runs, the 35 with a readable `alg_name` are all `ppo`; the remaining 6 are
empty or crashed. There is no GRPO run. "Matching AutoRL" matched a code path that never produced a
result, so fidelity was never evidence of correctness.

---

### 6. 建議修法 / Recommended fix

**中文** — 主要修正只有一處：**把 advantage 改成 trajectory-level**。

因為 reward 是 potential 差分，`Σ_t r_t = Φ_T − Φ_0 = Φ_T`，所以每條軌跡的 return **就是最終
potential**。作法：

1. 每條軌跡（buffer slot）算 `R_i = Σ_t r_t`；
2. 組內正規化 `A_i = (R_i − mean_g R) / (σ_g + ε)`——**分母用組內 return 的 std，不是單步 reward 的**；
3. 把 `A_i` 這個常數廣播到該軌跡的所有 timestep。

這就是標準 GRPO 配 outcome reward，一次解掉 §3.1（有了真正的 group baseline）、§3.2（idle 的
`R=0` 會參與統計因而被懲罰）、§3.3（不再對單步事件做平移）、§3.5（死樣本率 → 0）。
建議用 `--grpo-adv-level {step,trajectory}` 加上去，保留 `step` 以維持與現有 run 的可比性。

**English** — One change: compute the advantage at trajectory level. Since the reward is a potential
difference, each trajectory's return is its final potential. Take `R_i = Σ_t r_t` per slot,
normalize within the group, and broadcast the scalar to every timestep. That is textbook GRPO with
an outcome reward and fixes all four mechanisms at once. Suggested as
`--grpo-adv-level {step,trajectory}`, keeping `step` for comparability with existing runs.

**已測過但不足以修好的 / Tried and insufficient**（同一份 fixture，見 §7）：

| 變體 | idle | hold | drop | success | 判定 |
|---|---|---|---|---|---|
| 現行 `fix=True` | +0.000 | −0.141 | −0.859 | +0.952 | ✗ idle 免罰 |
| `fix=False` | −2.261 | +0.066 | **−3.509** | +13.747 | ✗ drop 仍比 idle 差；尺度爆掉 |
| `fix=False` + `std_scope=none` | −0.098 | +0.003 | −0.153 | +0.598 | ✗ 排序仍錯；等效 LR 砍掉 |
| **trajectory-level** | **−0.462** | **+0.228** | −0.375 | **+2.294** | ✓ |

**次要 / Secondary**（都不是根因，但值得一起調）：
- `scene` 分組只有 16 envs，success 率 2–5% 時多數組完全沒有成功案例 → 整組 dead。
  改 trajectory-level 後仍建議放大組別或提高每組 env 數，讓組內同時含成功與失敗。
- `alg_entropy_coef = 0` 對「塌縮成不動」沒有任何抵抗力，建議給小正值。
- `--vla-vhlr` / `--buffer-gamma` / `--buffer-lambda` 在 GRPO 路徑上是 inert 的，文件已記錄，
  但 `run_config.json` 仍會寫出它們，讀 config 時別誤判。

---

### 7. 重跑驗證 / Re-verification

不需要 GPU / No GPU required. `$PY` = 任一有 numpy 的環境，例如
`$(conda run -n cronos_tf447_cu121 which python)`。

```bash
# §2.1 / §2.2 — 每個 run 的 grasp / success 首末比較
$PY - <<'PY'
import csv,glob,json,os,numpy as np
from collections import defaultdict
for f in sorted(glob.glob('*/wandb/run-*/glob/rollout_success.csv')):
    c=json.load(open(os.path.join(os.path.dirname(f),'run_config.json')))
    d=defaultdict(list)
    for r in csv.DictReader(open(f)):
        d[int(r['episode'])].append((float(r['success']),float(r['is_src_obj_grasped'])))
    if len(d)<3: continue
    eps=sorted(d); k=max(1,len(eps)//5)
    g=lambda E: np.mean([x[1] for e in E for x in d[e]])
    print(f"{c['alg_name']:>5} T{c['episode_len']:<5} seed{c['seed']} {f.split('/')[0]:<9}"
          f" ep{eps[0]}-{eps[-1]:<4} grasp {g(eps[:k]):.4f} -> {g(eps[-k:]):.4f}")
PY

# §2.3 — 死樣本率與「0 有效樣本的 optimizer step」
$PY - <<'PY'
import re,glob,numpy as np
pat=re.compile(r"\[GRPO Step \d+/\d+\].*Zero-adv frac: ([\d.]+)")
for f in sorted(glob.glob('GRPO*/wandb/run-*/glob/ppo_log.txt')):
    z=np.array([float(m.group(1)) for m in (pat.match(l.strip()) for l in open(f)) if m])
    n=(len(z)//20)*20; live=(1-z[:n]).reshape(-1,20).sum(axis=1)*8
    print(f"{f.split('/')[0]}/{f.split('/')[2][-8:]}: zero-adv {z.mean():.4f}, "
          f"100%-dead {(z==1).mean()*100:.1f}%, 0-live steps {(live<1e-9).mean()*100:.1f}%")
PY
```

```bash
# §4.1 — CRONOS 與 AutoRL 的 compute_returns_grpo 是否 bit-identical
# §3.4 / §6 — 各變體的 advantage 排序（用真實的 CronosReplayBuffer）
$PY - <<'PY'
import sys, os, numpy as np
sys.path.insert(0,'<repo>/CRONOS'); os.chdir('/tmp')
from training.buffer import CronosReplayBuffer
class A: pass
a=A(); a.segment_len=80; a.buffer_gamma=.99; a.buffer_lambda=.95
a.buffer_minibatch=8; a.num_envs=64; a.episode_len=80
rng=np.random.default_rng(11)

def autorl(rewards, masks, fix):          # verbatim replay_buffer.py:109
    if fix:
        rv=rewards[rewards!=0]; rn=rewards.copy()
        rn[rn!=0]-=rv.mean(); rn[rn!=0]/=(rv.std()+1e-5)
    else: rn=(rewards-rewards.mean())/(rewards.std()+1e-5)
    ret=0; out=np.zeros_like(rewards)
    for s in reversed(range(rewards.shape[0])):
        ret=rn[s]+masks[s+1]*ret; out[s]=ret
    return out

r=np.zeros((80,64,1),np.float32); meta=[]
for e in range(64):
    if rng.random()>=.30: meta.append('idle'); continue
    k=int(rng.integers(10,50)); r[k,e,0]+=.1; r[k+1,e,0]+=.1
    if rng.random()<.20: r[int(rng.integers(k+2,80)),e,0]+=1.0; meta.append('success')
    elif rng.random()<.6: r[int(rng.integers(k+2,80)),e,0]-=.2; meta.append('drop')
    else: meta.append('hold')
meta=np.array(meta); m=np.ones((81,64,1),np.float32)

def run(label, fix, scope, gsz):
    b=CronosReplayBuffer(a,obs_dim=(1,1,1),act_dim=7); b.num_env=64
    b.rewards[:,:64]=r; b.masks[:,:64]=m
    b.compute_grpo_returns(group_ids=np.arange(64)//gsz, fix=fix, std_scope=scope)
    adv=np.asarray(b.advantages[:,:64,0]).copy(); per=adv.mean(axis=0)
    print(f"{label:<32}" + "".join(f"{per[meta==k].mean():+9.3f}" for k in
          ('idle','hold','drop','success')) + f"   dead {(adv==0).mean()*100:5.1f}%")
    b.cleanup(); return adv

print(f"{'variant':<32}{'idle':>9}{'hold':>9}{'drop':>9}{'success':>9}")
got = run("CURRENT (fix=True, global, 16)", True,  'global', 16)
run("fix=False",                            False, 'global', 16)
run("fix=False + std_scope=none",           False, 'none',   16)
R=r.sum(axis=0)[:,0]; g=np.arange(64)//16; Ac=np.zeros(64)
for gg in np.unique(g):
    c=g==gg; Ac[c]=(R[c]-R[c].mean())/(R[c].std()+1e-5)
print(f"{'TRAJECTORY-level (correct)':<32}" + "".join(
      f"{Ac[meta==k].mean():+9.3f}" for k in ('idle','hold','drop','success')))

# bit-equality against AutoRL (batch scope + own std)
for fix in (True, False):
    b=CronosReplayBuffer(a,obs_dim=(1,1,1),act_dim=7); b.num_env=64
    b.rewards[:,:64]=r; b.masks[:,:64]=m
    b.compute_grpo_returns(group_ids=None, fix=fix, std_scope='group')
    d=np.abs(autorl(r,m,fix)-np.asarray(b.advantages[:,:64])).max()
    print(f"fix={fix!s:<5} max|AutoRL - CRONOS| = {d:.3e}")
    b.cleanup()
PY
```

```bash
# §5.1 — AutoRL 的 action_log_probs 寬度是 1，不是 act_dim（RL4VLA 才是）
grep -n 'action_log_probs = \(np.zeros\|create_memmap\)' \
  $AUTORL_ROOT/SimplerEnv/simpler_env/utils/replay_buffer.py \
  $RL4VLA_ROOT/SimplerEnv/simpler_env/utils/replay_buffer.py
grep -n 'assert len(logprobs.shape)' \
  $AUTORL_ROOT/SimplerEnv/simpler_env/policies/openvla/openvla_train.py

# §5.2 — pose 分支的真實位置與極性
grep -n 'select_pos_ids =\|if obj_set' \
  $AUTORL_ROOT/ManiSkill/mani_skill/envs/tasks/digital_twins/bridge_dataset_eval/pick_place_multi.py

# §5.3 — AutoRL 的 41 個 run 全是 ppo
$PY - <<'PY'
import glob,re,os
for d in sorted(glob.glob(os.path.expandvars('$AUTORL_ROOT/wandb/*run-*/'))):
    hit=None
    for f in glob.glob(d+'*.wandb'):
        b=open(f,'rb').read(4_000_000)
        for m in re.finditer(rb'alg_name', b):
            v=re.search(rb'(ppo|grpo)', b[m.start():m.start()+80])
            if v: hit=v.group(1).decode(); break
    print(os.path.basename(d.rstrip('/')), hit)
PY
```

---

### 8. 這份文件讓哪些既有數據失效 / What this invalidates

**中文** — 七個 GRPO run 的 `rollout/*` 與 `eval_*` 數據**本身是有效的**（它們忠實記錄了發生的事），
但**不能**被解讀為「GRPO 在這個任務上的表現」——它們記錄的是一個獎勵不作為的目標函數的收斂結果。
任何「GRPO vs PPO」的比較在 §6 的修法落地之前都不應該寫進論文。
Part 1 §4.1、§5、§6 的三項在被修正前不應被引用。

**English** — The seven runs' logged metrics are valid records of what happened, but must not be
read as "GRPO's performance on this task": they record the convergence of an objective that rewards
inaction. No GRPO-vs-PPO comparison should be published before the §6 fix lands, and the three
claims in Part 1 corrected here should not be cited until that document is updated.

