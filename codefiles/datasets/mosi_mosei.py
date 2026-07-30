import torch
import pickle 
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset

from codefiles.datasets.utils import create_missing_data_masks, apply_missing_mask

class MOSI_MOSEI(Dataset):
    def __init__(
            self, 
            dataset: str = "mosi",
            split: str = "train",
            split_nr: int = 1, 
            variant: str = "unimodal_1",
            zero_fill_rates: list = [0.0, 0.0],
            seed: int = 42
    ) -> None: 
        super().__init__()

        self.num_modalities = 3
        self.variant = variant

        if dataset == "mosi":
            #self.datadir = f"/sc-projects/sc-proj-ukb-cvd/projects/data/MOSI/aligned_50_testsplits_split_{split_nr}.pkl"
            self.datadir = f"/sc-projects/sc-proj-ukb-cvd/projects/rhti10/simple_mml_baseline_tr/archiv/file.pkl"
        elif dataset == "mosei":
            self.datadir = f"/sc-projects/sc-proj-ukb-cvd/projects/data/MOSEI/aligned_50_testsplits_split_{split_nr}.pkl"

        with open(self.datadir, 'rb') as f:
            self.dataset = pickle.load(f)

        if dataset == "mosi":
            self.dataset = self.dataset[split]
        elif dataset == "mosei":
            self.dataset = self.dataset[split]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # missing data 
        self.zero_fill_masks, self.missing_stats = create_missing_data_masks(
            total_samples=len(self.dataset["id"]),
            num_modalities=self.num_modalities,
            missing_rates=zero_fill_rates,
            random_seed=seed
        )

    def __len__(self):
        return len(self.dataset["id"])

    def __getitem__(self, idx) -> list:
        # label
        label = self.dataset["regression_labels"][idx]  # classification_labels, annotations
        label = torch.tensor(label).unsqueeze(0)
        label = (torch.round(label) + 3)

        # full multimodal datareturn
        datareturn = [
            self.transform(self.dataset["text_bert"][idx]).to(torch.float32).squeeze(),
            self.transform(self.dataset["vision"][idx]).to(torch.float32).squeeze(),
            self.transform(self.dataset["audio"][idx]).to(torch.float32).squeeze()
        ]

        # Missing Data
        if not "unimodal" in self.variant:
            datareturn[0] = apply_missing_mask(datareturn[0], self.zero_fill_masks[idx, 0])
            datareturn[1] = apply_missing_mask(datareturn[1], self.zero_fill_masks[idx, 1])
            datareturn[2] = apply_missing_mask(datareturn[2], self.zero_fill_masks[idx, 2])

        return {
            "language": datareturn[0],
            "vision": datareturn[1],
            "audio": datareturn[2],
            "label": label.long()
        }

