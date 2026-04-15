import itertools
import numpy as np
import torch
from pathlib import Path
from mani_skill.utils import io_utils
from transforms3d.euler import euler2quat

# Constants from AutoRL/RL4VLA
CARROT_DATASET_DIR = Path(__file__).resolve().parents[2] / "ManiSkill" / "mani_skill" / "assets" / "carrot"

# Standard 6×6 grid used across all environments (range [-1, 1])
GRID_POS_UNIT = np.array([
    [0.0, 0.0], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8], [0.0, 1.0],
    [0.2, 0.0], [0.2, 0.2], [0.2, 0.4], [0.2, 0.6], [0.2, 0.8], [0.2, 1.0],
    [0.4, 0.0], [0.4, 0.2], [0.4, 0.4], [0.4, 0.6], [0.4, 0.8], [0.4, 1.0],
    [0.6, 0.0], [0.6, 0.2], [0.6, 0.4], [0.6, 0.6], [0.6, 0.8], [0.6, 1.0],
    [0.8, 0.0], [0.8, 0.2], [0.8, 0.4], [0.8, 0.6], [0.8, 0.8], [0.8, 1.0],
    [1.0, 0.0], [1.0, 0.2], [1.0, 0.4], [1.0, 0.6], [1.0, 0.8], [1.0, 1.0],
]) * 2 - 1  # [36, 2]

# 4×4 grid for environments with more slots (e.g., 3×3)
GRID_POS_4x4 = np.array([
    [0.0, 0.0], [0.0, 0.33], [0.0, 0.66], [0.0, 1.0],
    [0.33, 0.0], [0.33, 0.33], [0.33, 0.66], [0.33, 1.0],
    [0.66, 0.0], [0.66, 0.33], [0.66, 0.66], [0.66, 1.0],
    [1.0, 0.0], [1.0, 0.33], [1.0, 0.66], [1.0, 1.0],
]) * 2 - 1  # [16, 2]

# Standard quaternion configs (4 rotations)
QUAT_CONFIGS = np.stack([
    np.array([euler2quat(0, 0, 0.0), [1, 0, 0, 0]]),
    np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
    np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
    np.array([euler2quat(0, 0, np.pi * 3 / 4), [1, 0, 0, 0]]),
])


def generate_pose_configs(slot_heights, half_edge_length, spacing_matrix,
                          xy_center=(-0.16, 0.00), grid=None):
    """Generates xyz position configs for N slots (objects + receptacles) on a grid.

    Args:
        slot_heights: list of z-values for each slot, e.g. [1.0, 1.0, 0.95, 0.95]
        half_edge_length: (2,) workspace half-size [x, y]
        spacing_matrix: (N, N) minimum pairwise distances.
            spacing_matrix[i][j] is the min distance between slot i and slot j.
            Use 0.0 for no constraint. Only upper triangle is checked.
        xy_center: (2,) workspace center [x, y]
        grid: optional custom grid in unit coords [-1, 1]. Default: GRID_POS_UNIT (6×6).

    Returns:
        xyz_configs: np.ndarray of shape (N_configs, N_slots, 3)
    """
    center = np.array(xy_center).reshape(1, 2)
    half = np.array(half_edge_length).reshape(1, 2)
    base_grid = grid if grid is not None else GRID_POS_UNIT
    grid_pos = base_grid * half + center

    num_slots = len(slot_heights)
    spacing = np.array(spacing_matrix)

    xyz_configs = []
    for combo in itertools.product(grid_pos, repeat=num_slots):
        valid = True
        for i in range(num_slots):
            for j in range(i + 1, num_slots):
                if spacing[i][j] > 0 and np.linalg.norm(combo[i] - combo[j]) <= spacing[i][j]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            config = np.array([np.append(combo[k], slot_heights[k]) for k in range(num_slots)])
            xyz_configs.append(config)

    return np.stack(xyz_configs)


def _uniform_spacing(n, d):
    """Creates an NxN spacing matrix where all pairs have distance d."""
    return [[d if i != j else 0.0 for j in range(n)] for i in range(n)]


# Pre-defined pose configurations matching AutoRL's environments exactly.
# Verified: each produces identical xyz_configs as the original _generate_init_pose().
POSE_PRESETS = {
    # 3 slots: carrot(z=0.95), plate(z=0.92), extra_carrot(z=1.0)
    # Non-uniform: carrot-plate 0.07, plate-extra 0.07, carrot-extra 0.15
    "TwoObjectOneReceptacle": dict(
        slot_heights=[0.95, 0.92, 1.0],
        half_edge_length=(0.075, 0.075),
        spacing_matrix=[
            [0.0, 0.07, 0.15],
            [0.07, 0.0, 0.07],
            [0.15, 0.07, 0.0],
        ],
    ),

    # 3 slots: carrot(z=1.0), plate1(z=0.92), extra_plate(z=0.96)
    # Non-uniform: carrot-plate1 0.07, plate1-extra 0.15, carrot-extra 0.07
    "OneObjectTwoReceptacle": dict(
        slot_heights=[1.0, 0.92, 0.96],
        half_edge_length=(0.075, 0.075),
        spacing_matrix=[
            [0.0, 0.07, 0.07],
            [0.07, 0.0, 0.15],
            [0.07, 0.15, 0.0],
        ],
    ),

    # 4 slots: carrot1(z=1.0), carrot2(z=1.0), plate1(z=0.95), plate2(z=0.95)
    # Uniform: all pairs > 0.12
    "TwoObjectTwoReceptacle": dict(
        slot_heights=[1.0, 1.0, 0.95, 0.95],
        half_edge_length=(0.075, 0.075),
        spacing_matrix=_uniform_spacing(4, 0.12),
    ),
    "TwoObjectTwoReceptacle_OOD": dict(
        slot_heights=[1.0, 1.0, 0.95, 0.95],
        half_edge_length=(0.11, 0.15),
        spacing_matrix=_uniform_spacing(4, 0.12),
    ),

    # 6 slots: 3 carrots(z=1.0), 3 plates(z=0.95)
    # Uniform: all pairs > 0.1. Uses 4×4 grid (16 points) instead of 6×6.
    "ThreeObjectThreeReceptacle": dict(
        slot_heights=[1.0, 1.0, 1.0, 0.95, 0.95, 0.95],
        half_edge_length=(0.13, 0.12),
        spacing_matrix=_uniform_spacing(6, 0.1),
        grid=GRID_POS_4x4,
    ),
    "ThreeObjectThreeReceptacle_OOD": dict(
        slot_heights=[1.0, 1.0, 1.0, 0.95, 0.95, 0.95],
        half_edge_length=(0.11, 0.13),
        spacing_matrix=_uniform_spacing(6, 0.1),
        grid=GRID_POS_4x4,
    ),

    # --- New environment configurations ---

    # 4 slots: 1 carrot(z=1.0), 3 plates(z=0.95)
    # Uniform: all pairs > 0.12
    "OneObjectThreeReceptacle": dict(
        slot_heights=[1.0, 0.95, 0.95, 0.95],
        half_edge_length=(0.075, 0.075),
        spacing_matrix=_uniform_spacing(4, 0.12),
    ),

    # 4 slots: 3 carrots(z=1.0), 1 plate(z=0.95)
    # Uniform: all pairs > 0.12
    "ThreeObjectOneReceptacle": dict(
        slot_heights=[1.0, 1.0, 1.0, 0.95],
        half_edge_length=(0.075, 0.075),
        spacing_matrix=_uniform_spacing(4, 0.12),
    ),

    # 5 slots: 2 carrots(z=1.0), 3 plates(z=0.95)
    # Uniform: all pairs > 0.1. Uses 4×4 grid.
    "TwoObjectThreeReceptacle": dict(
        slot_heights=[1.0, 1.0, 0.95, 0.95, 0.95],
        half_edge_length=(0.11, 0.11),
        spacing_matrix=_uniform_spacing(5, 0.1),
        grid=GRID_POS_4x4,
    ),

    # 5 slots: 3 carrots(z=1.0), 2 plates(z=0.95)
    # Uniform: all pairs > 0.1. Uses 4×4 grid.
    "ThreeObjectTwoReceptacle": dict(
        slot_heights=[1.0, 1.0, 1.0, 0.95, 0.95],
        half_edge_length=(0.11, 0.11),
        spacing_matrix=_uniform_spacing(5, 0.1),
        grid=GRID_POS_4x4,
    ),
}


class TaskSuite:
    """Manages object/receptacle databases and initial pose configurations."""

    def __init__(self, device="cuda"):
        self.device = device
        self.model_db_carrot = io_utils.load_json(CARROT_DATASET_DIR / "more_carrot" / "model_db.json")
        self.model_db_plate = io_utils.load_json(CARROT_DATASET_DIR / "more_plate" / "model_db.json")
        self.model_db_table = io_utils.load_json(CARROT_DATASET_DIR / "more_table" / "model_db.json")

        # Filter plate database as in AutoRL
        only_plate_name = list(self.model_db_plate.keys())[0]
        self.model_db_plate = {k: v for k, v in self.model_db_plate.items() if k == only_plate_name}

        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

    def get_pose_configs(self, preset_name):
        """Returns (xyz_configs, quat_configs) for a named preset."""
        if preset_name not in POSE_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(POSE_PRESETS.keys())}")
        params = POSE_PRESETS[preset_name]
        xyz = generate_pose_configs(**params)
        print(f"xyz_configs: {xyz.shape}")
        print(f"quat_configs: {QUAT_CONFIGS.shape}")
        return xyz, QUAT_CONFIGS.copy()
