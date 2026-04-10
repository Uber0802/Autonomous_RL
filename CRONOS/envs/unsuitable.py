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

from typing import Callable, Dict, Iterable, List, Protocol, Union

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

    Exposes ``per_actor_class(env) -> {"obj": mask, "recep": mask, "any": mask}``
    so the runtime respawn path in ``BasePickPlace.reset_unsuitable_envs``
    can respawn carrot-only or plate-only when only one of them fell, matching
    V0.1's asymmetric behavior.
    """

    name: str = "low_z"
    z_threshold: float = 0.7

    def per_actor_class(self, env) -> dict:
        obj_low = env.get_obj_pos()[:, 2] < self.z_threshold
        recep_low = env.get_recep_pos()[:, 2] < self.z_threshold
        return {"obj": obj_low, "recep": recep_low, "any": obj_low | recep_low}

    def __call__(self, env) -> torch.Tensor:
        return self.per_actor_class(env)["any"]


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
