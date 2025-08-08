from typing import Dict, Any
from copy import deepcopy

import tensorflow as tf

from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
from prismatic.vla.datasets.rlds.utils.data_utils import binarize_gripper_actions

def simpler_env_replay_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    The trajectory transformation function for the simpler_env_replay_dataset.

    We just need the same binarization of the gripper action as bridge_orig_dataset_transform.
    """

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )

    return trajectory


NEW_OXE_STANDARDIZATION_TRANSFORMS = deepcopy(OXE_STANDARDIZATION_TRANSFORMS)
NEW_OXE_STANDARDIZATION_TRANSFORMS["simpler_env_replay_dataset"] = (
    simpler_env_replay_dataset_transform
)
NEW_OXE_STANDARDIZATION_TRANSFORMS["sd_bridge"] = (
    simpler_env_replay_dataset_transform
)
