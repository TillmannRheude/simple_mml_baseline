import torch
import pickle 

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
            zero_fill_rates: list = [0.3, 0.3, 0.0],
            seed: int = 42
    ) -> None: 
        super().__init__()

        if dataset == "mosi":
            self.datadir = "/sc-projects/sc-proj-ukb-cvd/projects/mml_tr/IMDer/dataset/MOSI/aligned_50.pkl"
        elif dataset == "mosei":
            self.datadir = "/sc-projects/sc-proj-ukb-cvd/projects/mml_tr/IMDer/dataset/MOSEI/aligned_50.pkl"

        self.split = split
        self.variant = variant
        self.num_modalities = 3

        with open(self.datadir, 'rb') as f:
            self.dataset = pickle.load(f)
        self.dataset = self.dataset[self.split]

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
        label = torch.tensor(label) 

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

        return [
            datareturn, 
            label, 
            datareturn
        ]

