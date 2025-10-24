import torch
import copy 
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from codefiles.datasets.ch_sims import CH_Sims, collate_fn
from codefiles.datasets.ch_sims_v2 import CH_Sims_v2, collate_fn_v2


class CH_Sims_Datamodule(pl.LightningDataModule):

    def __init__(
        self, 
        batch_size: int = 64,
        split_nr: int = 1,
        num_workers: int = 4,
        variant: str = "unimodal_1",
        missing: dict = {"missing_train": [], "missing_valid": [], "missing_test": []},
        seed: int = 42,
        v2: bool = False
    ) -> None: 
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_nr = split_nr
        self.variant = variant
        self.seed = seed 
        self.missing = missing
        self.v2 = v2

        self.collate_fn = collate_fn_v2 if v2 else collate_fn

    def setup(self, stage=None):
        if self.v2:
            self.train_dataset = CH_Sims_v2(split="train", zero_fill_rates=self.missing["missing_train"], split_nr=self.split_nr, variant=self.variant, seed=self.seed)
            self.val_dataset = CH_Sims_v2(split="valid", zero_fill_rates=self.missing["missing_valid"], split_nr=self.split_nr, variant=self.variant, seed=self.seed)
            self.test_dataset = CH_Sims_v2(split="test", zero_fill_rates=self.missing["missing_test"], split_nr=self.split_nr, variant=self.variant, seed=self.seed)
        else:
            self.train_dataset = CH_Sims(split="train", zero_fill_rates=self.missing["missing_train"], split_nr=self.split_nr, variant=self.variant, seed=self.seed)
            self.val_dataset = CH_Sims(split="valid", zero_fill_rates=self.missing["missing_valid"], split_nr=self.split_nr, variant=self.variant, seed=self.seed)
            self.test_dataset = CH_Sims(split="test", zero_fill_rates=self.missing["missing_test"], split_nr=self.split_nr, variant=self.variant, seed=self.seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)
    
    def test_dataloader(self,):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)



