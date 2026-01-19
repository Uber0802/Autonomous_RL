# Non-Episodic RL

## Table of Contents
- [Install](#install)
- [Training](#training)
  - [Configs](#basic-configs)
  - [Tools](#tools)
- [Evaluation](#evaluation)
  - [Configs](#configs)
  - [Tools](#tools)
- [Others](#others)
  - [Code Structure](#code-structure)
  - [Object List](#object-list)
  - [Plate List](#plate-list)

## Install
1. Clone the repository.
    ```bash
    git clone git@github.com:Uber0802/Autonomous_RL.git
    cd Autonomous_RL
    ```
2. Create conda env: nerl_env.
    ```bash
    conda create -n nerl_env -y python=3.10
    conda activate nerl_env
    ```
3. Run installation.
    ```bash
    chmod +x *.sh
    ./setup.sh
    ```
4. Optional: For ubuntu 2204
    ```bash
    sudo apt-get update
    sudo apt-get install -y libglvnd-dev
    ```

## Training
To train a nerl baseline models, run the following command:
<!--TODO: DEFAULT ARGS-->
```bash
python train_ms3_ppo.py
```
### Basic Configs
- `--seed`: Random seed for reproducibility.
- `--vla-path`: (Default: "openvla/openvla-7b")
- `--vla-unnorm-key`: (Default: "bridge_orig")
- `--vla-load-path`: Path to a pretrained VLA checkpoint. Use this to resume training from saved weights.

### Logging Configs
<!--TODO: RENAME-->
- `--name`: WandB run name.
- `--log`: Path to a `.txt` log file for PPO training output.
- `--wandb`: Enable logging to Weights & Biases. (Default: True)

### Environment Configs
- `--obj-set`: <!--TODO: RENAME-->
    - "fixed": Use identical scene and object layout across all environments.
    - "rand": Use random scene and object layout across all environments.
    - "rand_ood": For evaluating OOD.
- `--obj1-index` and `--obj2-index`: Choose an object using its index from the [Object List](#object-list). (Default: 7 and 2)
- `--plate1-index` and `--plate2-index`: Choose an plate using its index from the [Plate List](#plate-list).(Default: 1 and 2)

### Training Configs
<!--TODO: RENAME-->
- `--max-episodes`: Total number of training episodes.
- `--max-reset`: Total number of resets. (Default: 8192 = 128 episodes * 64 environments, 655360 steps for training_len=80)
- `--training-len`: Rollout length (steps per episode). (Default: 320)
- `--training-interval`: Number of rollout steps between VLA training updates. (Default: 160)
- `--instruction-switch-interval`: Number of rollout steps before switching to a new instruction. (Default: 80)
- `--interval-eval`: Number of episodes between evaluation.
- `--interval-save`: Number of episodes between saving model weights.
- `--eval-at-start`: Enable evaluation at step 0.

### Reset Gripper
- `--reset-robot` : Reset robot every 80 steps. (Default: `False`)

### Forward Backward
<!--TODO: RENAME-->
- `enable-backward`: Enable forward backward training. (Default: False)
- `backward-interval`: Number of forward instructions between backward instruction. Set to 1 for interleaved switch. (Default: 1)

### Reset Unsuitable
- `reset-unsuitable`: Reset environments that fail to complete the current task after an instruction switch. (Default: False)

### FIFO Buffer
<!--TODO: DEBUG (deleting dir)-->
Online + Offline Training
- `fifo_buffer`: Enable FIFO replay buffer.
- `fifo_length`: Maximum number of trajectories the buffer can store.

```
### Tools
We have provided a bash script, you can modify the script as needed.
#### Bash Script
#### Example
Example for 1280 Forward Backward with Reset Unsuitable.
```bash
python simpler_env/train_ms3_ppo.py \
  --name="bottle_shovel-1280-rand_scene-FB-seed_2" \
  --log="user/Autonomous_RL/bottle_shovel-1280-rand_scene-FB-seed_2.txt" \
  --env-id="TwoObjectTwoReceptacle-v1" \
  --vla-path="openvla/openvla-7b" --vla-unnorm-key="bridge_orig" \
  --training-len=1280 --max-episodes=8 \
  --interval-eval=1 --interval-save=1 \
  --enable-backward --backward-interval=1 \
  --reset-unsuitable \
  --seed=2 --obj_set="rand"
```

## Evaluation
To evaluate a trained nerl baseline agent, run the following command:
```bash
python eval_nerl_baseline.py
```
<!--TODO: ANOTHER FUNCTION-->
### Configs
- `--only_render` : For evaluating single task.
- `--only_render_seq` : For evaluating sequential task.
- `--task_order_config` : Path to task order config file for sequential task. <!--TODO: ADD THIS PARAMETER-->
- `--obj-set`: <!--TODO: RENAME-->
  - "rand": For evaluating ID.
  - "rand_ood": For evaluating OOD.
### Tools
We have provided a bash script, you can modify the script as needed.
<!--TODO: REFACTOR or REMOVE-->
Modify the `vla_load_paths` in `eval_single.sh` or `eval_seq.sh` to point to your model weights.
```bash
vla_load_paths=(
/path/to/your/first/model/weights
/path/to/your/second/model/weights
...
)
```
Make sure `seed` and `obj_set` in `./eval_ood.sh` or `./eval_seq.sh` are consistent with the training settings.
#### Collect Results
<!--TODO: REFACTOR-->
Collect success rates from evaluations.
1. Modify the `paths` in `collect_success_rates.sh` to point to your evaluation runs.
    ```bash
    paths=(
    /path/to/wandb/offline-run-1
    /path/to/wandb/offline-run-2
    ...
    )
    ```
2. Modify the `output_file` in `collect_success_rates.sh` to point to summary output file.
    ```bash
    output_file="success_summary.txt"
    ```
3. Run:
    ```bash
    bash collect_success_rates.sh
    ```

## Others
### Code Structure
#### Autonomous_RL/ManiSkill/mani_skill/envs/tasks/digital_twins/
- bridge_dataset_eval/pick_place_multi.py: All environment implemented here.
#### Autonomous_RL/SimplerEnv/simpler_env/
- train_ms3_ppo.py: All training and evaluation code.
- env/simpler_wrapper.py: Agent interact with the environment through here. 
- policies/openvla/openvla_train.py: PPO training algorithm.
- utils/replay_buffer.py: Replay buffer.

### Object List
Bridge dataset: carrot, plastic bottle, 7up can, kitchen spoon, cup
| Index | Object Name       |
|-------|-------------------|
| 1     | carrot            |
| 2     | kitchen shovel    |
| 3     | bread             |
| 4     | plastic bottle    |
| 5     | 7up can           |
| 6     | zuchinni          |
| 7     | ketchup bottle    |
| 8     | watering can      |
| 9     | pipe              |
| 10    | toy bear          |
| 11    | fast food cup     |
| 12    | plant             |
| 13    | banana            |
| 14    | hamburger         |
| 15    | golf ball         |
| 16    | BBQ sauce         |
| 17    | travel cup        |
| 18    | pepper            |
| 19    | nonstop can       |
| 20    | potato            |
| 21    | baguette          |
| 22    | champagne glass   |
| 23    | kitchen spoon     |
| 24    | onion             |
| 25    | cup               |

### Plate List
Bridge dataset: yellow_plate, cloth

| Index | Plate Name       |
|-------|-------------------|
| 1     | yellow_plate      |
| 2     | cloth             |
| 3     | carpet            |
| 4     | newspaper         |
| 5     | sheet metal       |
| 6     | drawing tablet    |
| 7     | tomato slice      |
| 8     | pizza             |
| 9     | flat bowl         |
| 10    | gramophone disk   |
| 11    | frying pan        |
| 12    | mouse pad         |
| 13    | cutting board     |
| 14    | chess board       |
| 15    | manhole cover     |
| 16    | envelope          |
| 17    | notepad           |
| 18    | black_plate       |
