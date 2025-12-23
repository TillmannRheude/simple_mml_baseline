import pytorch_lightning as pl
from torch.utils.data import DataLoader

from codefiles.datasets.mysterymml import MysteryMML

class MysteryMML_Datamodule(pl.LightningDataModule):

    def __init__(
            self, 
            batch_size: int = 32, 
            num_workers: int = 4,
            variant: str = "unimodal_1",
            split_nr: int = 1, 
            missing: dict = {"missing_train": [], "missing_valid": []},
            seed: int = 42
    ) -> None: 
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.variant = variant
        self.split_nr = split_nr
        self.missing = missing
        self.seed = seed

    def prepare_data(self):
        self.train_dataset = MysteryMML(split="train", split_nr=self.split_nr, variant=self.variant, zero_fill_rates=self.missing["missing_train"], seed=self.seed)
        self.val_dataset = MysteryMML(split="valid", split_nr=self.split_nr, variant=self.variant, zero_fill_rates=self.missing["missing_valid"], seed=self.seed)
        self.test_dataset = MysteryMML(split="test", split_nr=self.split_nr, variant=self.variant, zero_fill_rates=self.missing["missing_valid"], seed=self.seed)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)

