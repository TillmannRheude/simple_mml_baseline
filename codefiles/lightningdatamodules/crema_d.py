import torch 
import pytorch_lightning as pl 
import numpy as np

from torch.utils.data import DataLoader

from codefiles.datasets.crema_d import CREMAD, CREMAD_Embeddings

class CREMAD_Datamodule(pl.LightningDataModule):
    
    def __init__(self, 
        batch_size: int = 64,
        split_nr: int = 1,
        num_workers: int = 4,
        variant: str = "unimodal_1",
        missing: dict = {
                "missing_train": [], "missing_valid": [], "missing_test": []
        },
        seed: int = 420,
        corrupted_data_protocol: bool = False,
        use_embeddings: bool = False,
    ) -> None: 
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_nr = split_nr
        self.variant = variant
        self.missing = missing
        self.seed = seed
        self.corrupted_data_protocol = corrupted_data_protocol
        self.use_embeddings = use_embeddings

    def setup(self, stage=None):
        if self.use_embeddings:
            self.train_dataset = CREMAD_Embeddings(split="train", zero_fill_rates=self.missing["missing_train"], variant=self.variant, split_nr=self.split_nr, seed=self.seed, csv_splits="/path/to/data/crema-d-mirror/splits.csv", preproc_root_embeddings="/path/to/data/crema-d-mirror/embeddings")
            self.val_dataset = CREMAD_Embeddings(split="val", zero_fill_rates=self.missing["missing_valid"], variant=self.variant, split_nr=self.split_nr, seed=self.seed, csv_splits="/path/to/data/crema-d-mirror/splits.csv", preproc_root_embeddings="/path/to/data/crema-d-mirror/embeddings")
            self.test_dataset = CREMAD_Embeddings(split="test", zero_fill_rates=self.missing["missing_test"], variant=self.variant, split_nr=self.split_nr, seed=self.seed, csv_splits="/path/to/data/crema-d-mirror/splits.csv", preproc_root_embeddings="/path/to/data/crema-d-mirror/embeddings")
        else:
            if self.corrupted_data_protocol:
                # Leakage between train/test and no validation set 
                self.train_dataset = CREMAD(split="train", zero_fill_rates=self.missing["missing_test"], variant=self.variant, split_nr=self.split_nr, seed=self.seed, csv_splits="/path/to/data/crema-d-mirror/splits_aug.csv")
                self.val_dataset = CREMAD(split="test", zero_fill_rates=self.missing["missing_valid"], variant=self.variant, split_nr=self.split_nr, seed=self.seed, csv_splits="/path/to/data/crema-d-mirror/splits_aug.csv")
                self.test_dataset = CREMAD(split="test", zero_fill_rates=self.missing["missing_test"], variant=self.variant, split_nr=self.split_nr, seed=self.seed, csv_splits="/path/to/data/crema-d-mirror/splits_aug.csv")
            else:
                # No leakage between and existing train/val/test
                self.train_dataset = CREMAD(split="train", zero_fill_rates=self.missing["missing_train"], variant=self.variant, split_nr=self.split_nr, seed=self.seed, csv_splits="/path/to/data/crema-d-mirror/splits.csv")
                self.val_dataset = CREMAD(split="val", zero_fill_rates=self.missing["missing_valid"], variant=self.variant, split_nr=self.split_nr, seed=self.seed, csv_splits="/path/to/data/crema-d-mirror/splits.csv")
                self.test_dataset = CREMAD(split="test", zero_fill_rates=self.missing["missing_test"], variant=self.variant, split_nr=self.split_nr, seed=self.seed, csv_splits="/path/to/data/crema-d-mirror/splits.csv")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=self.num_workers, shuffle=True, worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id), generator=torch.Generator().manual_seed(self.seed))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=self.num_workers, shuffle=False, worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id), generator=torch.Generator().manual_seed(self.seed))
    
    def test_dataloader(self,):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True, pin_memory=True, persistent_workers=True, num_workers=self.num_workers, shuffle=False, worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id), generator=torch.Generator().manual_seed(self.seed)) 