"""Cross-stack adapter-config compatibility. No GPU, no torch, no peft.

`simpler_env.policies.peft_compat` imports peft lazily (only inside the two
functions that construct peft objects), so the pure key-splitting logic is
testable here. The module is loaded by path rather than by import so the suite
keeps running in a checkout where `simpler_env` is not pip-installed.

Run from the CRONOS directory:
    python -m pytest tests/ -q
"""

import contextlib
import dataclasses
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMPAT_PY = ROOT.parent / "SimplerEnv" / "simpler_env" / "policies" / "peft_compat.py"

_spec = importlib.util.spec_from_file_location("cronos_peft_compat", COMPAT_PY)
peft_compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(peft_compat)

split_forward_keys = peft_compat.split_forward_keys
AdapterConfigCompatError = peft_compat.AdapterConfigCompatError

# The fields peft 0.11.1's LoraConfig actually has — i.e. what the tf440 stack
# can accept. Anything outside this set in a serialized config was written by a
# newer peft.
PEFT_011_LORA_FIELDS = {
    "peft_type", "auto_mapping", "base_model_name_or_path", "revision", "task_type",
    "inference_mode", "r", "target_modules", "lora_alpha", "lora_dropout", "fan_in_fan_out",
    "bias", "use_rslora", "modules_to_save", "init_lora_weights", "layers_to_transform",
    "layers_pattern", "rank_pattern", "alpha_pattern", "megatron_config", "megatron_core",
    "loftq_config", "use_dora", "layer_replication",
}

# A CRONOS OpenVLA adapter as peft 0.14.0 (tf447) serializes it: the LoRA setup
# from openvla_train.py plus the three fields 0.11.1 has no field for.
TF447_CONFIG = {
    "alpha_pattern": {},
    "auto_mapping": None,
    "base_model_name_or_path": "openvla/openvla-7b",
    "bias": "none",
    "eva_config": None,
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": "gaussian",
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": 16,
    "lora_bias": False,
    "lora_dropout": 0.0,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": 32,
    "rank_pattern": {},
    "revision": None,
    "target_modules": ["proj", "qkv", "fc1", "fc2", "q", "kv", "fc3", "q_proj", "k_proj",
                       "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": None,
    "use_dora": False,
    "use_rslora": False,
}

FORWARD_KEYS = {"eva_config", "exclude_modules", "lora_bias"}


class TestSplitForwardKeys(unittest.TestCase):
    def test_tf447_config_loads_under_peft_011(self):
        """The real failure: 3 peft-0.14 keys drop out, everything else survives."""
        kwargs, dropped = split_forward_keys(TF447_CONFIG, PEFT_011_LORA_FIELDS)

        self.assertEqual(set(dropped), FORWARD_KEYS)
        self.assertEqual(set(kwargs), set(TF447_CONFIG) - FORWARD_KEYS)

    def test_surviving_kwargs_are_bit_identical(self):
        """Filtering must not rewrite a single retained value."""
        kwargs, _ = split_forward_keys(TF447_CONFIG, PEFT_011_LORA_FIELDS)

        for key, value in kwargs.items():
            self.assertEqual(value, TF447_CONFIG[key], key)
        # The adapter geometry specifically: r, alpha and the full target set.
        self.assertEqual(kwargs["r"], 32)
        self.assertEqual(kwargs["lora_alpha"], 16)
        self.assertEqual(len(kwargs["target_modules"]), 14)
        self.assertEqual(kwargs["init_lora_weights"], "gaussian")

    def test_tf440_config_is_untouched_under_peft_014(self):
        """Reverse direction (old writer, new reader) is the identity transform."""
        tf440 = {k: v for k, v in TF447_CONFIG.items() if k not in FORWARD_KEYS}
        peft_014_fields = PEFT_011_LORA_FIELDS | FORWARD_KEYS

        kwargs, dropped = split_forward_keys(tf440, peft_014_fields)

        self.assertEqual(dropped, {})
        self.assertEqual(kwargs, tf440)

    def test_same_stack_roundtrip_drops_nothing(self):
        kwargs, dropped = split_forward_keys(
            TF447_CONFIG, PEFT_011_LORA_FIELDS | FORWARD_KEYS
        )
        self.assertEqual(dropped, {})
        self.assertEqual(kwargs, TF447_CONFIG)


class TestNonDefaultForwardKeysRaise(unittest.TestCase):
    """The guard: only keys that are provably inert may be dropped."""

    def test_lora_bias_enabled_raises(self):
        cfg = dict(TF447_CONFIG, lora_bias=True)
        with self.assertRaises(AdapterConfigCompatError) as cm:
            split_forward_keys(cfg, PEFT_011_LORA_FIELDS)
        self.assertIn("lora_bias", str(cm.exception))

    def test_exclude_modules_set_raises(self):
        cfg = dict(TF447_CONFIG, exclude_modules=["lm_head"])
        with self.assertRaises(AdapterConfigCompatError) as cm:
            split_forward_keys(cfg, PEFT_011_LORA_FIELDS)
        self.assertIn("exclude_modules", str(cm.exception))

    def test_eva_config_set_raises(self):
        cfg = dict(TF447_CONFIG, eva_config={"rho": 2.0})
        with self.assertRaises(AdapterConfigCompatError):
            split_forward_keys(cfg, PEFT_011_LORA_FIELDS)

    def test_unvetted_future_key_raises(self):
        """A key from some future peft is not silently dropped just because it is None."""
        cfg = dict(TF447_CONFIG, corda_config=None)
        with self.assertRaises(AdapterConfigCompatError) as cm:
            split_forward_keys(cfg, PEFT_011_LORA_FIELDS)
        self.assertIn("corda_config", str(cm.exception))

    def test_error_names_every_offending_key(self):
        cfg = dict(TF447_CONFIG, lora_bias=True, exclude_modules=["lm_head"])
        with self.assertRaises(AdapterConfigCompatError) as cm:
            split_forward_keys(cfg, PEFT_011_LORA_FIELDS)
        msg = str(cm.exception)
        self.assertIn("lora_bias", msg)
        self.assertIn("exclude_modules", msg)
        # and points at the two ways out
        self.assertIn("tf447", msg)
        self.assertIn("INERT_FORWARD_KEYS", msg)


class TestInertTable(unittest.TestCase):
    def test_table_covers_exactly_the_peft_014_additions(self):
        self.assertEqual(set(peft_compat.INERT_FORWARD_KEYS), FORWARD_KEYS)

    def test_inert_values_match_peft_014_defaults(self):
        self.assertEqual(
            peft_compat.INERT_FORWARD_KEYS,
            {"eva_config": None, "exclude_modules": None, "lora_bias": False},
        )


def stub_peft(field_names, version="0.11.1"):
    """A fake `peft` exposing a LoraConfig dataclass with exactly `field_names`.

    `load_lora_config` introspects `dataclasses.fields(LoraConfig)` and imports
    peft lazily, so injecting this into sys.modules exercises the real function
    against a peft 0.11.1-shaped config class without installing peft.
    """
    module = types.ModuleType("peft")
    module.__version__ = version
    module.LoraConfig = dataclasses.make_dataclass(
        "LoraConfig",
        [(name, Any, dataclasses.field(default=None)) for name in sorted(field_names)],
    )
    return module


@contextlib.contextmanager
def peft_installed(field_names, version="0.11.1"):
    module = stub_peft(field_names, version)
    saved = sys.modules.get("peft")
    sys.modules["peft"] = module
    try:
        yield module
    finally:
        if saved is None:
            del sys.modules["peft"]
        else:
            sys.modules["peft"] = saved


class TestLoadLoraConfigEndToEnd(unittest.TestCase):
    """The whole path: tf447 JSON on disk -> a LoraConfig peft 0.11.1 accepts."""

    def write_ckpt(self, directory, cfg):
        path = Path(directory) / "episode_0000"
        path.mkdir()
        (path / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
        return path

    def test_tf447_checkpoint_builds_a_peft_011_config(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = self.write_ckpt(d, TF447_CONFIG)
            with peft_installed(PEFT_011_LORA_FIELDS):
                config = peft_compat.load_lora_config(ckpt)

        # Reconstructed identically, minus the three fields 0.11.1 cannot hold.
        self.assertEqual(config.r, 32)
        self.assertEqual(config.lora_alpha, 16)
        self.assertEqual(config.lora_dropout, 0.0)
        self.assertEqual(config.init_lora_weights, "gaussian")
        self.assertEqual(config.bias, "none")
        self.assertEqual(config.peft_type, "LORA")
        self.assertEqual(len(config.target_modules), 14)
        self.assertEqual(config.target_modules, TF447_CONFIG["target_modules"])
        self.assertFalse(hasattr(config, "eva_config"))

    def test_every_field_the_config_class_has_is_carried_over(self):
        """No retained key is silently lost between JSON and dataclass."""
        with tempfile.TemporaryDirectory() as d:
            ckpt = self.write_ckpt(d, TF447_CONFIG)
            with peft_installed(PEFT_011_LORA_FIELDS):
                config = peft_compat.load_lora_config(ckpt)

        for key, value in TF447_CONFIG.items():
            if key in FORWARD_KEYS:
                continue
            self.assertEqual(getattr(config, key), value, key)

    def test_newer_peft_keeps_the_forward_fields(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = self.write_ckpt(d, TF447_CONFIG)
            with peft_installed(PEFT_011_LORA_FIELDS | FORWARD_KEYS, version="0.14.0"):
                config = peft_compat.load_lora_config(ckpt)

        self.assertIsNone(config.eva_config)
        self.assertIsNone(config.exclude_modules)
        self.assertIs(config.lora_bias, False)

    def test_enabled_forward_key_blocks_the_load(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = self.write_ckpt(d, dict(TF447_CONFIG, lora_bias=True))
            with peft_installed(PEFT_011_LORA_FIELDS):
                with self.assertRaises(AdapterConfigCompatError):
                    peft_compat.load_lora_config(ckpt)


class TestLoadPeftAdapter(unittest.TestCase):
    """The contract with peft: a prebuilt config goes in, so peft never re-reads
    the file that it cannot parse."""

    def make_peft(self, field_names, calls):
        module = stub_peft(field_names)

        class PeftModel:
            @classmethod
            def from_pretrained(cls, model, model_id, adapter_name="default",
                                is_trainable=False, config=None, **kwargs):
                calls.append(dict(model=model, model_id=model_id,
                                  adapter_name=adapter_name,
                                  is_trainable=is_trainable, config=config))
                return ("peft-model", model, config)

        module.PeftModel = PeftModel
        return module

    def test_prebuilt_config_is_handed_to_peft(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "episode_0000"
            ckpt.mkdir()
            (ckpt / "adapter_config.json").write_text(json.dumps(TF447_CONFIG))

            module = self.make_peft(PEFT_011_LORA_FIELDS, calls)
            saved = sys.modules.get("peft")
            sys.modules["peft"] = module
            try:
                out = peft_compat.load_peft_adapter("base", ckpt, is_trainable=True)
            finally:
                if saved is None:
                    del sys.modules["peft"]
                else:
                    sys.modules["peft"] = saved

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIsNotNone(call["config"], "config must be prebuilt, not left to peft")
        self.assertEqual(call["config"].r, 32)
        self.assertFalse(hasattr(call["config"], "eva_config"))
        self.assertTrue(call["is_trainable"])
        self.assertEqual(call["adapter_name"], "default")
        # peft's own loader joins paths with os.path.join; hand it a str, not a Path.
        self.assertIsInstance(call["model_id"], str)
        self.assertEqual(call["model_id"], str(ckpt))
        self.assertEqual(out[0], "peft-model")

    def test_missing_config_parameter_is_reported_clearly(self):
        """A peft without the `config` hook fails with our message, not a TypeError."""
        module = stub_peft(PEFT_011_LORA_FIELDS)

        class PeftModel:
            @classmethod
            def from_pretrained(cls, model, model_id, adapter_name="default",
                                is_trainable=False):
                raise AssertionError("should not be reached")

        module.PeftModel = PeftModel

        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "episode_0000"
            ckpt.mkdir()
            (ckpt / "adapter_config.json").write_text(json.dumps(TF447_CONFIG))

            saved = sys.modules.get("peft")
            sys.modules["peft"] = module
            try:
                with self.assertRaises(AdapterConfigCompatError) as cm:
                    peft_compat.load_peft_adapter("base", ckpt)
            finally:
                if saved is None:
                    del sys.modules["peft"]
                else:
                    sys.modules["peft"] = saved

        self.assertIn("config", str(cm.exception))


class TestLoadLoraConfigOnDisk(unittest.TestCase):
    """The file-reading half, as far as it goes without peft installed."""

    def test_missing_config_file_raises_filenotfound(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                peft_compat.load_lora_config(d)

    def test_non_lora_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "adapter_config.json").write_text(
                json.dumps({"peft_type": "IA3", "r": 8})
            )
            try:
                peft_compat.load_lora_config(d)
            except AdapterConfigCompatError as e:
                self.assertIn("IA3", str(e))
            except ImportError:
                self.skipTest("peft not installed")
            else:
                self.fail("expected AdapterConfigCompatError")


if __name__ == "__main__":
    unittest.main()
