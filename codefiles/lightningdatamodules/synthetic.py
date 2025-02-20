import torch 
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

from codefiles.datasets.synthetic import Halved_Fashion_or_Vanilla_MNIST

class Halved_Fashion_or_Vanilla_MNIST_Datamodule(pl.LightningDataModule):

    def __init__(
            self, 
            data_dir: str = "./data", 
            batch_size: int = 64, 
            num_workers: int = 4,
            num_modalities: int = 2,
            dataset: str = "fmnist",
            missing: dict = {"missing_train": [], "missing_valid": [], "missing_test": []},
            seed: int = 42,
            variant: str = "unimodal_1",
    ) -> None: 
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_modalities = num_modalities
        self.dataset = dataset
        self.missing = missing
        self.seed = seed
        self.variant = variant

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # sum of zero_fill_rates must be less than or equal to 1, assert
        assert sum(self.missing["missing_train"]) <= 1, "sum of zero_fill_rates must be less than or equal to 1"
        assert sum(self.missing["missing_valid"]) <= 1, "sum of zero_fill_rates must be less than or equal to 1"

        self.train_dataset = Halved_Fashion_or_Vanilla_MNIST(
            root='data', train=True, transform=transform, 
            download=True, variant=self.variant,
            zero_fill_rates=self.missing["missing_train"], 
            num_modalities=self.num_modalities, dataset=self.dataset, seed=self.seed
        )
        self.val_dataset = Halved_Fashion_or_Vanilla_MNIST(
            root='data', train=False, transform=transform, 
            download=True, variant=self.variant,
            zero_fill_rates=self.missing["missing_valid"], 
            num_modalities=self.num_modalities, dataset=self.dataset, seed=self.seed
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True)
