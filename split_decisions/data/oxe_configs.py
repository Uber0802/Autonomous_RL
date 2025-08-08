from copy import deepcopy

from prismatic.vla.datasets.rlds.oxe.configs import (
    OXE_DATASET_CONFIGS,
    ActionEncoding,
    StateEncoding
)


NEW_OXE_DATASET_CONFIGS = deepcopy(OXE_DATASET_CONFIGS)
NEW_OXE_DATASET_CONFIGS["simpler_env_replay_dataset"] = {
    # Original version of Bridge V2 from project website
    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["EEF_state", None, "gripper_state"],
    "state_encoding": StateEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
}
NEW_OXE_DATASET_CONFIGS["sd_bridge"] = {
    # Original version of Bridge V2 from project website
    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["EEF_state", None, "gripper_state"],
    "state_encoding": StateEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
}