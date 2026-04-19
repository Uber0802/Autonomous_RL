"""CRONOS V0.2 M4 — Scene registry.

A *scene* is a named visual variation applied to the Bridge table stage.
Within a single ManiSkill vectorized env, all envs share the same stage
geometry (flat_table or sink), so scene variation is limited to:

- RGB overlay image (the inpainting background)
- Lighting parameters (ambient + directional colours/intensities)

Different stage geometries (flat_table vs sink) require separate env instances
and are NOT mixed within a single run.  Cross-stage mixing is deferred to V0.3.

Usage::

    from envs.scenes import SCENES, SceneSpec
    spec = SCENES["flat_table_default"]
    # spec.overlay_path, spec.ambient, spec.directional_lights
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mani_skill import ASSET_DIR

BRIDGE_INPAINT = ASSET_DIR / "tasks/bridge_v2_real2sim_dataset/real_inpainting"


@dataclass(frozen=True)
class DirectionalLight:
    direction: Tuple[float, float, float]
    color: Tuple[float, float, float]
    shadow: bool = False


@dataclass(frozen=True)
class SceneSpec:
    """Describes one named scene variation."""
    name: str
    scene_setting: str = "flat_table"  # flat_table | sink
    overlay_path: Optional[str] = None  # None → use stage default
    ambient: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    directional_lights: Tuple[DirectionalLight, ...] = field(default_factory=lambda: (
        DirectionalLight(direction=(0, 0, -1), color=(2.2, 2.2, 2.2)),
        DirectionalLight(direction=(-1, -0.5, -1), color=(0.7, 0.7, 0.7)),
        DirectionalLight(direction=(1, 1, -1), color=(0.7, 0.7, 0.7)),
    ))


# ---------------------------------------------------------------------------
# Built-in scenes
# ---------------------------------------------------------------------------

SCENES: Dict[str, SceneSpec] = {}


def register_scene(spec: SceneSpec) -> SceneSpec:
    SCENES[spec.name] = spec
    return spec


# 1. Default flat table — matches V0.1 / AutoRL exactly
register_scene(SceneSpec(
    name="flat_table_default",
    scene_setting="flat_table",
    overlay_path=str(BRIDGE_INPAINT / "bridge_real_eval_1.png"),
))

# 2. Dimmed lighting — same overlay, reduced light intensity
register_scene(SceneSpec(
    name="flat_table_dim",
    scene_setting="flat_table",
    overlay_path=str(BRIDGE_INPAINT / "bridge_real_eval_1.png"),
    ambient=(0.15, 0.15, 0.15),
    directional_lights=(
        DirectionalLight(direction=(0, 0, -1), color=(1.2, 1.2, 1.2)),
        DirectionalLight(direction=(-1, -0.5, -1), color=(0.4, 0.4, 0.4)),
        DirectionalLight(direction=(1, 1, -1), color=(0.4, 0.4, 0.4)),
    ),
))

# 3. Warm lighting — warm directional tint
register_scene(SceneSpec(
    name="flat_table_warm",
    scene_setting="flat_table",
    overlay_path=str(BRIDGE_INPAINT / "bridge_real_eval_1.png"),
    ambient=(0.35, 0.28, 0.20),
    directional_lights=(
        DirectionalLight(direction=(0, 0, -1), color=(2.5, 2.0, 1.5)),
        DirectionalLight(direction=(-1, -0.5, -1), color=(0.8, 0.6, 0.4)),
        DirectionalLight(direction=(1, 1, -1), color=(0.8, 0.6, 0.4)),
    ),
))

# 4. Alternative overlay (eval_2)
register_scene(SceneSpec(
    name="flat_table_eval2",
    scene_setting="flat_table",
    overlay_path=str(BRIDGE_INPAINT / "bridge_real_eval_2.png"),
))

# 5. Sink stage — different geometry + overlay
register_scene(SceneSpec(
    name="sink_default",
    scene_setting="sink",
    overlay_path=str(BRIDGE_INPAINT / "bridge_sink.png"),
))


def get_scene(name: str) -> SceneSpec:
    """Look up a scene by name. Raises KeyError if unknown."""
    if name not in SCENES:
        raise KeyError(f"Unknown scene '{name}'. Available: {sorted(SCENES.keys())}")
    return SCENES[name]
