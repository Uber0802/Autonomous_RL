from typing import Any, Dict, Tuple, Type, Optional
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from split_decisions.prismatic.vla.datasets.datasets import EpisodicRLDSDataset, RLDSBatchTransform
from split_decisions.prismatic.vla.datasets.rlds import make_single_dataset
from split_decisions.prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES
from split_decisions.prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from split_decisions.prismatic.overwatch import initialize_overwatch
# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

from .oxe_materialize import tree_map, get_oxe_dataset_kwargs_and_weights


class CustomizedEpisodicRLDSDataset(EpisodicRLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        dataset_statistics: Optional[Dict[str, Any]] = None,
        split: str = "success"
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
            dataset_statistics=dataset_statistics,
            split=split
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

        # Update dataset length if dataset_statistics are provided
        if dataset_statistics is not None:
            cardinality = self.dataset.cardinality().numpy()
            if cardinality == tf.data.INFINITE_CARDINALITY:
                raise ValueError("Cannot compute dataset statistics for infinite datasets.")

            overwatch.info("Computing dataset statistics. This may take a bit, but should only need to happen once.")
            num_transitions, num_trajectories = 0, 0
            for traj in tqdm(self.dataset.iterator(), total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None):
                num_transitions += traj["action"].shape[0]
                num_trajectories += 1

            self.dataset_statistics['num_transitions'] = np.array(num_transitions)
            self.dataset_statistics['num_trajectories'] = np.array(num_trajectories)

            self.dataset_length = num_trajectories

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out