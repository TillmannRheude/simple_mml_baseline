import torch 
import pytorch_lightning as pl 
import numpy as np

from torch.utils.data import DataLoader

from codefiles.datasets.kinetics import Kinetics

class Kinetics_Datamodule(pl.LightningDataModule):
    
    def __init__(self, 
        batch_size: int = 64,
        split_nr: int = 1,
        num_workers: int = 4,
        variant: str = "unimodal_1",
        missing: dict = {
                "missing_train": [], "missing_valid": [], "missing_test": []
        },
        seed: int = 420,
        num_classes: int = 400
    ) -> None: 
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_nr = split_nr
        self.variant = variant
        self.missing = missing
        self.seed = seed
        self.num_classes = num_classes

    def setup(self, stage=None):
        self.train_dataset = Kinetics(split="train", zero_fill_rates=self.missing["missing_train"], variant=self.variant, split_nr=self.split_nr, seed=self.seed, num_classes=self.num_classes)
        self.val_dataset = Kinetics(split="val", zero_fill_rates=self.missing["missing_valid"], variant=self.variant, split_nr=self.split_nr, seed=self.seed, num_classes=self.num_classes)
        self.test_dataset = Kinetics(split="test", zero_fill_rates=self.missing["missing_test"], variant=self.variant, split_nr=self.split_nr, seed=self.seed, num_classes=self.num_classes)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=self.num_workers, shuffle=True,
                          worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id), generator=torch.Generator().manual_seed(self.seed))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=self.num_workers, shuffle=False, 
                          worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id), generator=torch.Generator().manual_seed(self.seed))
    
    def test_dataloader(self,):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=self.num_workers, shuffle=False,
                          worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id), generator=torch.Generator().manual_seed(self.seed)) 