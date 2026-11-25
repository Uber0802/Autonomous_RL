"""Cross-stack PEFT adapter loading — read a checkpoint under a *different* peft.

CRONOS ships two package stacks (see `CRONOS/setup.sh`, and `README.md`
§"Checkpoint portability across envs"):

- `tf447` — transformers 4.47 / **peft 0.14.0** (`cronos_tf447_cu121/cu128`,
  historically `cronos_envV0.4` / `CE_SpVLA`). Serves both VLA pillars.
- `tf440` — transformers 4.40.1 / **peft 0.11.1** (`cronos_tf440_cu121/cu128`,
  historically `CE_opVLA`). OpenVLA only, and the only stack that reproduces
  the V0.1 baseline bit-for-bit.

The split is not a preference: OpenVLA's model code pins `transformers<4.43`,
peft 0.14 needs `transformers>=4.43` (it imports `EncoderDecoderCache`), and
peft 0.11.1 is the newest release that works against transformers 4.40.1. So no
single pin set satisfies both, and **no amount of version juggling fixes this** —
it has to be handled where the checkpoint is read.

The failure it causes
---------------------
`LoraConfig` is a dataclass, and `PeftConfig.from_pretrained` ends in
`config_cls(**json.load(adapter_config.json))`. A dataclass rejects keywords it
has no field for, so an adapter serialized by peft 0.14 and loaded under peft
0.11.1 dies with:

    TypeError: LoraConfig.__init__() got an unexpected keyword argument 'eva_config'

peft 0.14 added three `LoraConfig` fields that 0.11.1 does not know:
`eva_config`, `exclude_modules`, `lora_bias`. Every adapter saved by a `tf447`
env therefore carries all three, and every one of those checkpoints is
unreadable in a `tf440` env — even though the LoRA weights themselves are
byte-identical in format and load fine.

Why dropping the keys is safe
-----------------------------
In the affected checkpoints the three keys always hold their *default* value
(`None`, `None`, `False`) — i.e. nobody ever turned the peft-0.14-only features
on; they are present only because 0.14's `save_pretrained` serializes every
field. Removing a key whose value means "feature off" from the kwargs of a
`LoraConfig` that has no such feature is behaviour-neutral by construction.

That is the whole licence for this module, so it is enforced rather than
assumed: a forward key is dropped **only** when it carries the inert value
recorded in `INERT_FORWARD_KEYS`. A key that is genuinely set — say
`lora_bias: true` — or a key nobody has vetted raises, because silently
discarding it would build a model that differs from the one that was trained.

Direction and scope
-------------------
Only `tf447 → tf440` (new writer, old reader) needs repair. The reverse is a
no-op: a `tf440`-written config has no keys peft 0.14 lacks, so nothing is
filtered and `LoraConfig` fills the new fields from its own defaults. Both
pillars therefore route through `load_peft_adapter` unconditionally — it is the
identity transform in the stack that does not need it.

Not covered here: the LoRA *weights* (`adapter_model.safetensors` — same layout
in both peft releases) and `training_state.pt` (a torch pickle; its cross-torch
hazard is the `weights_only` default that flipped in torch 2.6, handled at the
`torch.load` call sites).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Tuple

CONFIG_NAME = "adapter_config.json"

# `LoraConfig` fields a newer peft may serialize that an older one has no field
# for, mapped to the value that means "this feature is off". A key is safe to
# drop only when the checkpoint holds exactly this value; anything else is a
# setting that would be lost, and raises instead.
#
# All three were added in peft 0.14.0 (the `tf447` stack) and are absent from
# peft 0.11.1 (the `tf440` stack):
INERT_FORWARD_KEYS: Dict[str, Any] = {
    "eva_config": None,       # EVA (SVD-driven) LoRA init; None = plain init
    "exclude_modules": None,  # subtracted from target_modules; None = subtract nothing
    "lora_bias": False,       # bias term on lora_B; False = no bias
}


class AdapterConfigCompatError(RuntimeError):
    """An adapter_config.json cannot be honoured by the installed peft."""


def split_forward_keys(
    raw: Dict[str, Any], accepted: set[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a serialized adapter config into (usable kwargs, dropped keys).

    `accepted` is the installed config dataclass's field names. Keys outside it
    were written by a newer peft; each is dropped only if `INERT_FORWARD_KEYS`
    vouches for its value. Pure dict-in/dict-out so it is testable without peft
    or torch installed.

    Raises `AdapterConfigCompatError` if any unknown key carries a value that
    would actually change the adapter.
    """
    unknown = {k: v for k, v in raw.items() if k not in accepted}

    # Two ways an unknown key can be unsafe, and they want different answers:
    # one the table vouches for but the checkpoint actually turned on, and one
    # nobody has looked at yet.
    enabled = {k: v for k, v in unknown.items()
               if k in INERT_FORWARD_KEYS and v != INERT_FORWARD_KEYS[k]}
    unvetted = {k: v for k, v in unknown.items() if k not in INERT_FORWARD_KEYS}

    if enabled or unvetted:
        lines = [
            f"{CONFIG_NAME} carries {len(enabled) + len(unvetted)} key(s) the installed "
            f"peft has no field for and this loader will not drop:"
        ]
        if enabled:
            lines.append(
                f"  * set to a non-default value, so dropping them would load a model "
                f"that differs from the one that was trained: {enabled!r}"
            )
        if unvetted:
            lines.append(
                f"  * written by a peft newer than this table knows about, so whether "
                f"they are inert has never been checked: {unvetted!r}"
            )
        lines += [
            "Fix by one of:",
            "  * evaluate/resume this checkpoint in the stack that wrote it "
            "(tf447 = transformers 4.47 / peft 0.14) — see README.md "
            "§'Checkpoint portability across envs';",
            f"  * once you have confirmed a key is inert at its saved value, add it to "
            f"INERT_FORWARD_KEYS in {__name__} with a note on why.",
        ]
        raise AdapterConfigCompatError("\n".join(lines))

    kwargs = {k: v for k, v in raw.items() if k in accepted}
    return kwargs, unknown


def load_lora_config(path: Path | str):
    """Build a `LoraConfig` from `path/adapter_config.json` under any peft.

    Replaces `LoraConfig.from_pretrained`, which passes the parsed JSON straight
    into the dataclass constructor and so raises `TypeError` on a config written
    by a newer peft.
    """
    config_file = Path(path) / CONFIG_NAME
    if not config_file.is_file():
        raise FileNotFoundError(f"No {CONFIG_NAME} in {path}")

    raw = json.loads(config_file.read_text())

    peft_type = raw.get("peft_type")
    if peft_type not in ("LORA", None):
        # CRONOS only ever writes LoRA adapters; a different adapter type means
        # the checkpoint came from somewhere else and this filter's inert-key
        # table does not describe it.
        raise AdapterConfigCompatError(
            f"{config_file} declares peft_type={peft_type!r}; only 'LORA' is supported."
        )

    # Imported here, not at module scope, so the checkpoint can be inspected (and
    # this module tested) in an env without peft.
    from peft import LoraConfig

    accepted = {f.name for f in dataclasses.fields(LoraConfig)}
    kwargs, dropped = split_forward_keys(raw, accepted)
    if dropped:
        import peft
        print(
            f"[peft-compat] {config_file}: dropped {sorted(dropped)} — written by a "
            f"newer peft, inert at their saved values, unrepresentable in peft "
            f"{peft.__version__}."
        )

    return LoraConfig(**kwargs)


def load_peft_adapter(model, path: Path | str, *, is_trainable: bool = True,
                      adapter_name: str = "default"):
    """`PeftModel.from_pretrained` that survives a cross-stack checkpoint.

    Passing a prebuilt `config` is what makes this work: `from_pretrained` reads
    `adapter_config.json` itself only when `config is None`, and the resulting
    `PeftModel` registers the adapter under `adapter_name`, so the subsequent
    `load_adapter` call does not re-read the file either. The adapter *weights*
    still go through peft's normal loader.
    """
    import inspect

    from peft import PeftModel

    config = load_lora_config(path)

    # The `config` parameter is the entire mechanism, so check it is there rather
    # than letting an unrelated peft raise a TypeError that looks like the bug
    # this module exists to fix. Present in every peft CRONOS pins (0.11.1, 0.14.0).
    if "config" not in inspect.signature(PeftModel.from_pretrained).parameters:
        import peft
        raise AdapterConfigCompatError(
            f"peft {peft.__version__}'s PeftModel.from_pretrained takes no `config` "
            f"argument, so a cross-stack adapter config cannot be injected. Install a "
            f"peft that CRONOS pins (0.11.1 for tf440, 0.14.0 for tf447) — see setup.sh."
        )

    return PeftModel.from_pretrained(
        model, str(path), adapter_name=adapter_name,
        is_trainable=is_trainable, config=config,
    )
