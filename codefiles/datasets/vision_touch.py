import h5py
import os
import pickle
import pandas as pd
import numpy as np
import torch
import torchvision

from tqdm import tqdm
from torch.utils.data import Dataset

from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask


class VisionTouch(Dataset):

    """ 
    Code from https://github.com/stanford-iprl-lab/multimodal_representation/blob/master/multimodal/trainers/selfsupervised.py
    """

    def __init__(
        self,
        transform=None,
        episode_length=50,
        training_type="selfsupervised",
        n_time_steps=1,
        action_dim=4,
        pairing_tolerance=0.06,
        split: str = "train",
        split_nr: int = 1, 
        variant: str = "unimodal_1",
        zero_fill_rates: list = [0.0, 0.0],
        seed: int = 42
    ):
        """
        Args:
            hdf5_file (handle): h5py handle of the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.num_modalities = 3
        self.variant = variant
        
        self.transform = transform
        self.episode_length = episode_length
        self.training_type = training_type
        self.n_time_steps = n_time_steps
        self.dataset = {}
        self.action_dim = action_dim
        self.pairing_tolerance = pairing_tolerance

        split_csv = pd.read_csv("/sc-projects/sc-proj-ukb-cvd/projects/data/vision_and_touch/multimodal/dataset/triangle_real_data/splits.csv")
        self.dataset_path = split_csv[split_csv[f"split_{split_nr}"] == split]["filename"].tolist()
        self.dataset_path = [path for path in self.dataset_path if path.endswith('.h5')]
        
        #self._config_checks()
        #self._init_paired_filenames()
        # --- Caching Logic ---
        cache_path = os.path.join(os.path.dirname(self.dataset_path[0]), f"paired_filenames_{split}_{split_nr}.pkl")
        if os.path.exists(cache_path):
            print(f"Loading paired filenames from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.paired_filenames = pickle.load(f)
        else:
            print("Cache not found. Pre-calculating paired filenames (this will take a while)...")
            self._config_checks()
            self._init_paired_filenames()
            
            print(f"Saving paired filenames to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.paired_filenames, f)
        # --- End Caching Logic ---

        # missing data
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=self.__len__(),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
        )

    def __len__(self):
        return len(self.dataset_path) * (self.episode_length - self.n_time_steps)

    def __getitem__(self, idx):

        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[list_index][:-8]

        file_number, filename = self._parse_filename(filename)

        unpaired_filename, unpaired_idx = self.paired_filenames[(list_index, dataset_index)]

        if dataset_index >= self.episode_length - self.n_time_steps - 1:
            dataset_index = np.random.randint(
                self.episode_length - self.n_time_steps - 1
            )

        sample = self._get_single(
            self.dataset_path[list_index],
            list_index,
            unpaired_filename,
            dataset_index,
            unpaired_idx,
            idx
        )
        return sample

    def _get_single(
        self, dataset_name, list_index, unpaired_filename, dataset_index, unpaired_idx, idx
    ):

        dataset = h5py.File(dataset_name, "r", swmr=True, libver="latest")
        unpaired_dataset = h5py.File(unpaired_filename, "r", swmr=True, libver="latest")

        if self.training_type == "selfsupervised":

            image = dataset["image"][dataset_index]
            depth = dataset["depth_data"][dataset_index]
            proprio = dataset["proprio"][dataset_index][:8]
            force = dataset["ee_forces_continuous"][dataset_index]

            if image.shape[0] == 3:
                image = np.transpose(image, (2, 1, 0))

            if depth.ndim == 2:
                depth = depth.reshape((128, 128, 1))

            flow = np.array(dataset["optical_flow"][dataset_index])
            flow_mask = np.expand_dims(
                np.where(
                    flow.sum(axis=2) == 0,
                    np.zeros_like(flow.sum(axis=2)),
                    np.ones_like(flow.sum(axis=2)),
                ),
                2,
            )

            unpaired_image = image
            unpaired_depth = depth
            unpaired_proprio = unpaired_dataset["proprio"][unpaired_idx][:8]
            unpaired_force = unpaired_dataset["ee_forces_continuous"][unpaired_idx]

            # Missing Data
            if not "unimodal" in self.variant:
                image = apply_missing_mask(torch.from_numpy(image.astype(np.float32)), self.zero_fill_masks[idx, 0])
                proprio = apply_missing_mask(torch.from_numpy(proprio.astype(np.float32)), self.zero_fill_masks[idx, 1])
                force = apply_missing_mask(torch.from_numpy(force.astype(np.float32)), self.zero_fill_masks[idx, 2])

            image = image.permute(2, 0, 1)
            image = torchvision.transforms.Resize((224, 224))(image)
            
            sample = {
                "image": image,
                "depth": depth,
                "flow": flow,
                "flow_mask": flow_mask,
                "action": torch.from_numpy(dataset["action"][dataset_index + 1]),  # 
                "force": force,
                "proprio": proprio,
                "ee_yaw_next": dataset["proprio"][dataset_index + 1][:self.action_dim],  # 
                "contact_next": np.array(
                    [dataset["contact"][dataset_index + 1].sum() > 0]
                ).astype(float),
                "unpaired_image": unpaired_image,
                "unpaired_force": unpaired_force,
                "unpaired_proprio": unpaired_proprio,
                "unpaired_depth": unpaired_depth,
            }

        dataset.close()
        unpaired_dataset.close()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _init_paired_filenames(self):
        """
        Precalculates the paired filenames.
        Imposes a distance tolerance between paired images
        """
        tolerance = self.pairing_tolerance

        all_combos = set()

        self.paired_filenames = {}
        for list_index in tqdm(range(len(self.dataset_path)), desc="pairing_files"):
            filename = self.dataset_path[list_index]
            file_number, _ = self._parse_filename(filename[:-8])

            dataset = h5py.File(filename, "r", swmr=True, libver="latest")

            for idx in range(self.episode_length - self.n_time_steps):

                proprio_dist = None
                while proprio_dist is None or proprio_dist < tolerance:
                    # Get a random idx, file that is not the same as current
                    unpaired_dataset_idx = np.random.randint(self.__len__())
                    unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(unpaired_dataset_idx)

                    while unpaired_filename == filename:
                        unpaired_dataset_idx = np.random.randint(self.__len__())
                        unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(unpaired_dataset_idx)

                    with h5py.File(unpaired_filename, "r", swmr=True, libver="latest") as unpaired_dataset:
                        proprio_dist = np.linalg.norm(dataset['proprio'][idx][:3] - unpaired_dataset['proprio'][unpaired_idx][:3])

                self.paired_filenames[(list_index, idx)] = (unpaired_filename, unpaired_idx)
                all_combos.add((unpaired_filename, unpaired_idx))

            dataset.close()

    def _idx_to_filename_idx(self, idx):
        """
        Utility function for finding info about a dataset index

        Args:
            idx (int): Dataset index

        Returns:
            filename (string): Filename associated with dataset index
            dataset_index (int): Index of data within that file
            list_index (int): Index of data in filename list
        """
        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[list_index]
        return filename, dataset_index, list_index

    def _parse_filename(self, filename):
        """ Parses the filename to get the file number and filename"""
        if filename[-2] == "_":
            file_number = int(filename[-1])
            filename = filename[:-1]
        else:
            file_number = int(filename[-2:])
            filename = filename[:-2]

        return file_number, filename

    def _config_checks(self):
        if self.training_type != "selfsupervised":
            raise ValueError(
                "Training type not supported: {}".format(self.training_type)
            )