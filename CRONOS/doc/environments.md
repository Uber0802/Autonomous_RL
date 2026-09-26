# Conda environments and checkpoint portability

Index of the documentation: [`README.md`](README.md). How to install is in the
top-level README (*Installation*); this document says why there are four
environments, what they do and do not reproduce, and how checkpoints move between
them.

## Why four environments

**Memory budget — pick the right env for your GPU.** The dual-VLA stack lifts OpenVLA-7B PPO peak memory from ~40 GB → ~55 GB (`transformers==4.47` HybridCache + `peft==0.14` fast path + newer torch caching), which **does not fit on Ada-class GPUs (48 GB)**. If you only need OpenVLA on Ada, the lightweight stack (`torch==2.2.0+cu121` + `transformers==4.40.1`, ~40 GB peak) still fits 1 OpenVLA-7B PPO on a 48 GB Ada. See the lightweight env below.

The `tf440` split is a workaround for that regression, not a design goal. `tools/bench_rollout.py` measures throughput and peak memory per stack with a phase breakdown, so the ~15 GB can be attributed and ideally removed — at which point `tf440` retires and the matrix collapses to one env per torch channel. The memory figures in the env table of the top-level README are historical reports, not `bench_rollout.py` output; re-measure on your own hardware before relying on them.

## Bit-exactness

**Bit-exact note:** Only `cronos_tf440_cu121` reproduces V0.1 baseline PPO logs bit-for-bit. `cronos_tf440_cu128` upgrades torch (no V0.1-era cu128 wheels exist), so it shares V0.1's transformers/peft *ABI* but not its exact cuBLAS/attention kernels. The dual-VLA (`tf447`) envs upgrade both torch and transformers, so their PPO logs drift ~10⁻² from V0.1 in the first 1000 minibatches and converge to <0.2% by PPO step 100 — algorithmically correct, numerically different. For bit-exact ablations, run the baseline arm in the **same env** as the test arm. Multi-seed mean±std comparisons are unaffected (drift ≪ seed-to-seed variance).

## Blackwell

Both variants pin `torch==2.7.0+cu128` (the lowest stable cu128 build with sm_120). What differs is the LM stack:

| Env | Transformers/peft/tokenizers | OpenVLA peak | Bit-exact to V0.1 cu121? |
|---|---|---|---|
| `cronos_tf447_cu128` | V0.4 (4.47 / 0.14 / 0.21) | ~55 GB | No — also drifts from V0.1 |
| `cronos_tf440_cu128` | V0.1 (4.40.1 / 0.11.1 / 0.19.1) | ~45 GB | No — cu128 changes cuBLAS/attention kernels, but transformers ABI matches V0.1 |

**No torch build is simultaneously V0.1-era *and* Blackwell-compatible.** The cu128 channel does not ship `torch==2.2.0` (cu128 wheels start at torch 2.7), and torch 2.2.0+cu121 has no `sm_120` SASS or PTX. Bit-exact V0.1 baseline replication is therefore Ada/Hopper-only by physics of GPU release dates.

*Reproducibility note:* PTX is JIT-compiled on first CUDA op on a new arch, so SASS may differ slightly between a Blackwell run and a Hopper run even within the same env. Training curves on Blackwell are *statistically* equivalent to Hopper, not bit-exact.

## Lightweight OpenVLA-only env (Ada)

For Ada-class GPUs (L40S, RTX 6000 Ada, A6000 — 48 GB), the dual-VLA stack's ~55 GB OpenVLA-7B PPO peak does **not** fit. The lightweight `openvla_v01` mode pins the lightweight stack (`transformers==4.40.1` + `peft==0.11.1` + `tokenizers==0.19.1`) and — on `cu121` — pins `torch==2.2.0` to match V0.1 exactly. OpenVLA-7B PPO peak stays at ~40 GB, fitting one PPO on a 48 GB Ada with headroom.

Tradeoffs:
- ✅ Fits on Ada (48 GB) — restores parity with V0.1's running memory profile.
- ✅ Numerically bit-exact against V0.1 baseline runs (same cuBLAS GEMM tile order + attention kernels).
- ❌ Cannot run `--policy spatialvla` — transformers ≥ 4.43 needed for the `HybridCache` import in SpatialVLA's `model/modeling_gemma2.py`. `setup.sh openvla_v01` skips the `../SpatialVLA` editable install entirely; the policy's lazy import in [main.py](../main.py) and [eval_only.py](../eval_only.py) (the `--policy spatialvla` branch of the policy construction) is gated by `--policy spatialvla` so it never fires under OpenVLA-only runs.
- ❌ Will not run on Blackwell as-is — pass `blackwell` as the 2nd arg to install the Blackwell variant: `./setup.sh openvla_v01 blackwell` produces `cronos_tf440_cu128` (lightweight stack on `torch==2.7.0+cu128`; loses cu121 bit-exactness but keeps V0.1 transformers ABI).

How `setup.sh` picks the install: the 1st positional arg picks the LM stack + which sibling pillars get installed, the 2nd picks the torch wheel channel. See the header comment in `setup.sh` for the full pin rationale and the 4-env recommended workflows.

## Checkpoint portability

A checkpoint is written by whichever env trained it, and the two LM stacks do not
serialize the LoRA adapter the same way. **`tf447` (peft 0.14) writes three
`LoraConfig` fields that `tf440` (peft 0.11.1) has no field for** — `eva_config`,
`exclude_modules`, `lora_bias`. `LoraConfig` is a dataclass, so peft 0.11.1 does
not ignore the extras; `PeftModel.from_pretrained` dies on the constructor:

```
TypeError: LoraConfig.__init__() got an unexpected keyword argument 'eva_config'
```

This cannot be pinned away. peft 0.14 needs `transformers>=4.43` (for
`EncoderDecoderCache`), which breaks OpenVLA's `transformers<4.43` pin, and peft
0.11.1 is the newest release that works against transformers 4.40.1 — so it is
handled at the load site instead, in
[`SimplerEnv/simpler_env/policies/peft_compat.py`](../../SimplerEnv/simpler_env/policies/peft_compat.py).
`load_peft_adapter` filters out fields the *installed* peft cannot represent
before building the config, and both VLA pillars load through it.

The filter is behaviour-neutral, and enforces that rather than assuming it: a key
is dropped **only** when it holds the value that means "feature off"
(`eva_config: null`, `exclude_modules: null`, `lora_bias: false`) — which is what
`tf447` checkpoints actually contain, since nobody turned those features on. A key
that is genuinely set, or one no peft release in the table accounts for, raises
instead of being silently discarded. The reverse direction (`tf440` checkpoint,
`tf447` reader) needs no repair and is the identity transform.

The LoRA *weights* (`adapter_model.safetensors`) have the same layout in both peft
releases and need no translation. `training_state.pt` is a torch pickle, and its
own cross-version hazard is torch 2.6 flipping the `torch.load` default to
`weights_only=True`; every load site passes `weights_only=False` explicitly, since
CRONOS spans torch 2.2 / 2.5 / 2.7 and writes these files itself.

## Env names

Renamed from `cronos_env` / `cronos_env_blackwell` / `cronos_env_lite` /
`cronos_env_lite_blackwell`, which did not say which stack they carried. Nothing
reads the env name programmatically, so existing envs keep working — rename with
`conda rename -n cronos_env cronos_tf447_cu121` when convenient.
