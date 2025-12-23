import torch 
import pytorch_lightning as pl 
import numpy as np

from torch.utils.data import DataLoader

from codefiles.datasets.inspect import INSPECT

class INSPECT_Datamodule(pl.LightningDataModule):

    def __init__(
        self, 
        batch_size: int = 64,
        split_nr: int = 1,
        num_workers: int = 4,
        variant: str = "unimodal_1",
        missing: dict = {"missing_train": [], "missing_valid": [], "missing_test": []},
        seed: int = 42
    ) -> None: 
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_nr = split_nr
        self.variant = variant
        self.seed = seed 
        self.missing = missing

    def setup(self, stage=None):
        self.train_dataset = INSPECT(split="train", zero_fill_rates=self.missing["missing_train"], split_nr=self.split_nr, variant=self.variant, seed=self.seed)
        self.val_dataset = INSPECT(split="valid", zero_fill_rates=self.missing["missing_valid"], split_nr=self.split_nr, variant=self.variant, seed=self.seed)
        self.test_dataset = INSPECT(split="test", zero_fill_rates=self.missing["missing_test"], split_nr=self.split_nr, variant=self.variant, seed=self.seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=self.num_workers, shuffle=True)  

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self,):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=self.num_workers, shuffle=False) 

