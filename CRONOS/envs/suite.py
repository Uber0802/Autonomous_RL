import torch
import numpy as np
from pathlib import Path
from mani_skill.utils import io_utils
from transforms3d.euler import euler2quat

# Constants from AutoRL/RL4VLA
CARROT_DATASET_DIR = Path(__file__).resolve().parents[2] / "ManiSkill" / "mani_skill" / "assets" / "carrot"

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
        
        self.xyz_configs = None
        self.quat_configs = None
        self._generate_default_poses()

    def _generate_default_poses(self):
        """Generates standard initial pose configurations from AutoRL."""
        xy_center = np.array([-0.16, 0.00]).reshape(1, 2)
        half_edge_length = np.array([0.075, 0.075]).reshape(1, 2)

        grid_pos = np.array([
            [0.0, 0.0], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8], [0.0, 1.0],
            [0.2, 0.0], [0.2, 0.2], [0.2, 0.4], [0.2, 0.6], [0.2, 0.8], [0.2, 1.0],
            [0.4, 0.0], [0.4, 0.2], [0.4, 0.4], [0.4, 0.6], [0.4, 0.8], [0.4, 1.0],
            [0.6, 0.0], [0.6, 0.2], [0.6, 0.4], [0.6, 0.6], [0.6, 0.8], [0.6, 1.0],
            [0.8, 0.0], [0.8, 0.2], [0.8, 0.4], [0.8, 0.6], [0.8, 0.8], [0.8, 1.0],
            [1.0, 0.0], [1.0, 0.2], [1.0, 0.4], [1.0, 0.6], [1.0, 0.8], [1.0, 1.0],
        ]) * 2 - 1
        grid_pos = grid_pos * half_edge_length + xy_center

        xyz_configs = []
        for p1 in grid_pos:
            for p2 in grid_pos:
                for p3 in grid_pos:
                    for p4 in grid_pos:
                        # Ensure sufficient spacing for 2 objects and 2 receptacles (simplifying 3x3 case)
                        if (np.linalg.norm(p1 - p2) > 0.12 and np.linalg.norm(p1 - p3) > 0.12 and 
                            np.linalg.norm(p1 - p4) > 0.12 and np.linalg.norm(p2 - p3) > 0.12):
                            xyz_configs.append(np.array([
                                np.append(p1, 1.0),  # Carrot 1
                                np.append(p2, 1.0),  # Carrot 2
                                np.append(p3, 0.95), # Plate 1
                                np.append(p4, 0.95), # Plate 2
                            ]))
        
        self.xyz_configs = torch.tensor(np.stack(xyz_configs), device=self.device)
        self.quat_configs = torch.tensor(np.stack([
            np.array([euler2quat(0, 0, 0.0), [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, np.pi * 3 / 4), [1, 0, 0, 0]]),
        ]), device=self.device)

    def get_task_params(self, env_id, obj_set="rand"):
        """Returns parameters specific to a task environment."""
        # This will contain the logic from _initialize_episode_pre for different classes
        pass
