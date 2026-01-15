import torch
import numpy as np
from torch.utils.data import Dataset

from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask

class MysteryMML(Dataset):
    def __init__(
            self, 
            split: str = "train",
            split_nr: int = 1, 
            variant: str = "unimodal_1",
            zero_fill_rates: list = [0.0],
            seed: int = 42
    ) -> None: 
        super().__init__()

        self.num_modalities = 2
        self.variant = variant

        self.datadir = f"/path/to/data/mystery_mml_dataset/multimodal_dataset.npz"
        self.dataset = np.load(self.datadir)
        self.dataset = {k: torch.tensor(v) for k, v in self.dataset.items()}

        # first 60% train, 20% valid, 20% test
        len_dataset = self.dataset["X_m1"].shape[0]
        train_end = int(0.6 * len_dataset)
        valid_start = train_end
        valid_end = int(0.8 * len_dataset)
        test_start = valid_end
        if split == "train":
            self.dataset = {key: self.dataset[key][:train_end, ...] for key in self.dataset.keys()}
        elif split == "valid":
            self.dataset = {key: self.dataset[key][valid_start:valid_end, ...] for key in self.dataset.keys()}
        elif split == "test":
            self.dataset = {key: self.dataset[key][test_start:, ...] for key in self.dataset.keys()}
        len_dataset = self.dataset["X_m1"].shape[0]

        # missing data 
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len_dataset,
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
        )

    def __len__(self):
        return self.dataset["X_m1"].shape[0]

    def __getitem__(self, idx) -> list:
        # label
        label = self.dataset["y"][idx].unsqueeze(0)

        # X1
        x1 = self.dataset["X_m1"][idx]

        # X2
        x2 = self.dataset["X_m2"][idx]

        datareturn = [x1, x2]

        # Missing Data
        if not "unimodal" in self.variant:
            datareturn[0] = apply_missing_mask(datareturn[0], self.zero_fill_masks[idx, 0])
            datareturn[1] = apply_missing_mask(datareturn[1], self.zero_fill_masks[idx, 1])

        return {
            "x_m1": datareturn[0],
            "x_m2": datareturn[1],
            "label": label.float()
        }