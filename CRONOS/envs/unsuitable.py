"""CRONOS V0.2 M2 Phase A — pluggable unsuitable-env detector.

A "detector" decides which envs in a vectorized run are in an unsuitable
state (object fallen, stuck out of reach, etc.) and should be respawned via
``reset_unsuitable_envs``. V0.1 hard-coded a single rule (``obj_z < 0.7 |
recep_z < 0.7``); V0.2 makes that rule pluggable so:

- The grid-size dependence introduced by ``PickPlaceNxM-v1`` (Phase B) can
  swap to a more appropriate detector without touching ``reset.py``.
- Contributors can drop a single file into ``envs/unsuitable/`` (or this
  module) with ``@register("my_name")`` and have CRONOS pick it up via the
  ``--unsuitable_detector`` CLI flag.

Phase A wires the registry through ``ResetStrategy.get_unsuitable_envs``.
That method is currently a *preview* (the runtime respawn path inlines its
own threshold inside ``bridge_multi.py::BasePickPlace.reset_unsuitable_envs``).
Phase B refactors that env-side method to consult ``ResetStrategy``, at which
point the registered detector becomes the single source of truth. Until
Phase B lands, the registry exists for configuration and unit-testability;
behavior is unchanged from V0.1.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Protocol, Union

import torch


class UnsuitableDetector(Protocol):
    """A detector takes the *unwrapped* env and returns a per-env bool mask.

    Implementations should be **pure** w.r.t. observable env state — no
    hidden side effects, no caching across calls — so they can be swapped
    in and out without invalidating training runs mid-flight.
    """

    name: str

    def __call__(self, env) -> torch.Tensor:  # noqa: D401  (Protocol)
        """Return a bool tensor of shape ``(num_envs,)``; ``True`` = unsuitable."""
        ...


# Registry: name -> detector instance.
DETECTORS: Dict[str, UnsuitableDetector] = {}


def register(name: str) -> Callable[[type], type]:
    """Class decorator that instantiates the detector and registers it."""

    def deco(cls: type) -> type:
        instance = cls()  # type: ignore[call-arg]
        # Make sure the instance carries its registered name even if the
        # class didn't set one explicitly.
        if not getattr(instance, "name", None):
            instance.name = name  # type: ignore[attr-defined]
        if name in DETECTORS:
            raise ValueError(f"Unsuitable detector already registered: {name}")
        DETECTORS[name] = instance
        return cls

    return deco


def get_detector(name_or_detector: Union[str, UnsuitableDetector]) -> UnsuitableDetector:
    """Resolve a name to a registered detector instance, or pass through if
    given an already-constructed detector."""
    if isinstance(name_or_detector, str):
        if name_or_detector not in DETECTORS:
            raise KeyError(
                f"Unknown unsuitable_detector '{name_or_detector}'. "
                f"Available: {sorted(DETECTORS.keys())}"
            )
        return DETECTORS[name_or_detector]
    return name_or_detector


def list_detectors() -> List[str]:
    return sorted(DETECTORS.keys())


# ----- built-in detectors --------------------------------------------------


@register("low_z")
class LowZDetector:
    """V0.1 default: an env is unsuitable iff any object or receptacle has
    fallen below ``z_threshold``. Reads ``env.get_obj_pos()`` and
    ``env.get_recep_pos()``, both of which are torch tensors of shape
    ``(num_envs, 3)`` exposed by ``BasePickPlace``.

    Threshold defaults to ``0.7`` to preserve V0.1 behavior bit-for-bit.

    Exposes ``per_actor_class(env) -> {"obj": mask, "recep": mask, "any": mask,
    "reasons": list[dict[env_idx, str]]}``
    so the runtime respawn path in ``BasePickPlace.reset_unsuitable_envs``
    can respawn carrot-only or plate-only when only one of them fell, matching
    V0.1's asymmetric behavior.
    """

    name: str = "low_z"
    z_threshold: float = 0.7

    def per_actor_class(self, env) -> dict:
        obj_z = env.get_obj_pos()[:, 2]
        recep_z = env.get_recep_pos()[:, 2]
        obj_low = obj_z < self.z_threshold
        recep_low = recep_z < self.z_threshold

        # V0.4 M3c: per-env reason strings naming the violated axis + value.
        reasons: List[Dict[int, str]] = []
        for i in range(obj_z.shape[0]):
            entry: Dict[int, str] = {}
            if obj_low[i]:
                entry[i] = f"obj_z={obj_z[i].item():.3f} < {self.z_threshold}"
            elif recep_low[i]:
                entry[i] = f"recep_z={recep_z[i].item():.3f} < {self.z_threshold}"
            reasons.append(entry)

        return {"obj": obj_low, "recep": recep_low, "any": obj_low | recep_low,
                "reasons": reasons}

    def __call__(self, env) -> torch.Tensor:
        return self.per_actor_class(env)["any"]


class WorkspaceAABBDetector:
    """V0.4 M3: workspace-AABB detector. An env is unsuitable iff any of its
    actors (carrot, plate) lies outside the axis-aligned bounding box that
    describes the robot's reachable workspace.

    Catches both kinds of physical mis-behavior:
    - Objects that fall (low z) — same as ``LowZDetector``.
    - Objects knocked sideways out of the robot's reach but still at table
      height (low/high x or y, in-range z) — silently ignored by ``LowZDetector``.

    Workspace bounds are constructor arguments, plumbed from YAML by the
    wrapper layer. Each axis is ``[min, max]``. Optional ``obj_z``/``recep_z``
    overrides allow tighter z floors for carrots vs receptacles.

    Not registered via ``@register`` because instances are parametric — the
    wrapper builds one with config-specific bounds and passes it to
    ``ResetStrategy`` directly.
    """

    name: str = "workspace_aabb"

    def __init__(
        self,
        x: tuple = (-float("inf"), float("inf")),
        y: tuple = (-float("inf"), float("inf")),
        z: tuple = (-float("inf"), float("inf")),
        obj_z: Optional[tuple] = None,
        recep_z: Optional[tuple] = None,
    ):
        self.x_min, self.x_max = float(x[0]), float(x[1])
        self.y_min, self.y_max = float(y[0]), float(y[1])
        self.z_min, self.z_max = float(z[0]), float(z[1])
        # Per-actor z overrides — fall back to the shared z range when None.
        if obj_z is None:
            self.obj_z_min, self.obj_z_max = self.z_min, self.z_max
        else:
            self.obj_z_min, self.obj_z_max = float(obj_z[0]), float(obj_z[1])
        if recep_z is None:
            self.recep_z_min, self.recep_z_max = self.z_min, self.z_max
        else:
            self.recep_z_min, self.recep_z_max = float(recep_z[0]), float(recep_z[1])

    @staticmethod
    def _axis_violation(value: float, lo: float, hi: float, axis: str) -> Optional[str]:
        """Return a reason string if value is outside [lo, hi], else None."""
        if value < lo:
            return f"{axis}={value:.3f} < {axis}_min={lo:.3f}"
        if value > hi:
            return f"{axis}={value:.3f} > {axis}_max={hi:.3f}"
        return None

    def per_actor_class(self, env) -> dict:
        obj_xyz = env.get_obj_pos()    # (num_envs, 3)
        recep_xyz = env.get_recep_pos()  # (num_envs, 3)
        n = obj_xyz.shape[0]
        device = obj_xyz.device

        # Vectorized AABB check per axis. mask = True if any axis is out of range.
        def _aabb(xyz: torch.Tensor, z_lo: float, z_hi: float) -> torch.Tensor:
            x_bad = (xyz[:, 0] < self.x_min) | (xyz[:, 0] > self.x_max)
            y_bad = (xyz[:, 1] < self.y_min) | (xyz[:, 1] > self.y_max)
            z_bad = (xyz[:, 2] < z_lo) | (xyz[:, 2] > z_hi)
            return x_bad | y_bad | z_bad

        obj_bad = _aabb(obj_xyz, self.obj_z_min, self.obj_z_max)
        recep_bad = _aabb(recep_xyz, self.recep_z_min, self.recep_z_max)

        # Per-env reason: first violated axis (obj checked before recep,
        # x before y before z). Empty dict when env is fine. Cheap because
        # we only inspect flagged envs.
        reasons: List[Dict[int, str]] = []
        any_bad = obj_bad | recep_bad
        for i in range(n):
            entry: Dict[int, str] = {}
            if bool(any_bad[i].item()):
                if bool(obj_bad[i].item()):
                    o = obj_xyz[i]
                    for axis_name, val, lo, hi in (
                        ("obj_x", float(o[0].item()), self.x_min, self.x_max),
                        ("obj_y", float(o[1].item()), self.y_min, self.y_max),
                        ("obj_z", float(o[2].item()), self.obj_z_min, self.obj_z_max),
                    ):
                        r = self._axis_violation(val, lo, hi, axis_name)
                        if r is not None:
                            entry[i] = r
                            break
                if i not in entry and bool(recep_bad[i].item()):
                    rp = recep_xyz[i]
                    for axis_name, val, lo, hi in (
                        ("recep_x", float(rp[0].item()), self.x_min, self.x_max),
                        ("recep_y", float(rp[1].item()), self.y_min, self.y_max),
                        ("recep_z", float(rp[2].item()), self.recep_z_min, self.recep_z_max),
                    ):
                        r = self._axis_violation(val, lo, hi, axis_name)
                        if r is not None:
                            entry[i] = r
                            break
            reasons.append(entry)

        return {"obj": obj_bad, "recep": recep_bad, "any": any_bad, "reasons": reasons}

    def __call__(self, env) -> torch.Tensor:
        return self.per_actor_class(env)["any"]


def build_workspace_aabb(cfg: dict) -> "WorkspaceAABBDetector":
    """Construct a ``WorkspaceAABBDetector`` from a YAML ``unsuitable_detector``
    block. Returns an instance ready to pass to ``ResetStrategy``.

    Schema::

        unsuitable_detector:
          name: workspace_aabb
          workspace:
            x: [xmin, xmax]
            y: [ymin, ymax]
            z: [zmin, zmax]
          obj_workspace: { z: [zmin, zmax] }     # optional, overrides shared z
          recep_workspace: { z: [zmin, zmax] }   # optional

    Raises ``ValueError`` if required keys are missing or malformed.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"unsuitable_detector must be a mapping, got {type(cfg).__name__}")

    ws = cfg.get("workspace")
    if not isinstance(ws, dict):
        raise ValueError("unsuitable_detector.workspace: missing or not a mapping")

    def _pair(key: str) -> tuple:
        v = ws.get(key)
        if v is None or len(v) != 2:
            raise ValueError(
                f"unsuitable_detector.workspace.{key}: must be a 2-element [min, max] list"
            )
        return (float(v[0]), float(v[1]))

    x = _pair("x")
    y = _pair("y")
    z = _pair("z")

    def _opt_z(key: str) -> Optional[tuple]:
        sub = cfg.get(key)
        if sub is None:
            return None
        if not isinstance(sub, dict) or "z" not in sub or len(sub["z"]) != 2:
            raise ValueError(f"unsuitable_detector.{key}.z: must be a 2-element [min, max] list")
        return (float(sub["z"][0]), float(sub["z"][1]))

    obj_z = _opt_z("obj_workspace")
    recep_z = _opt_z("recep_workspace")

    return WorkspaceAABBDetector(x=x, y=y, z=z, obj_z=obj_z, recep_z=recep_z)


def compose_or(detectors: Iterable[UnsuitableDetector]) -> UnsuitableDetector:
    """Internal helper: build an ad-hoc detector that ORs several others.
    Not registered — used for tests and future composite detectors."""

    detectors = list(detectors)

    class _Composite:
        name = "or(" + ",".join(getattr(d, "name", "?") for d in detectors) + ")"

        def __call__(self, env) -> torch.Tensor:
            mask = detectors[0](env)
            for d in detectors[1:]:
                mask = mask | d(env)
            return mask

    return _Composite()
