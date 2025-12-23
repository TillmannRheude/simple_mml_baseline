import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy, BinaryF1Score, MulticlassCalibrationError, BinaryAUROC

from codefiles.lightningmodules.utils import (
    get_input,
    get_target,
    LightningModuleParent
)


class MysteryMML_Lightning_Module(LightningModuleParent):

    def __init__(
        self,
        model,
        params_optimizer: dict = {
            "lr": 3e-5,
            "weight_decay": 0.000000001,
            "eps": 1e-7,
            "betas": (0.95, 0.99),
            "warmup_steps": 1
        },
        manual_opt: bool = True,
        dataset: str = "mystery_mml",
        params_ogm: dict = {
                "use_ge": True,
                "ge_noise_level": 0.1,
        },
        params_arl: dict = {},
        params_dgl: dict = {},
        params_mmpareto: dict = {},
        params_bmml: dict = {},
        params_gblend: dict = {},
        params_pdf: dict = {},
        params_omib: dict = {},
        params_aug: dict = {},
    ) -> None: 
        super().__init__(
            manual_opt=manual_opt, 
            params_ogm=params_ogm, 
            params_arl=params_arl, 
            params_dgl=params_dgl, 
            params_mmpareto=params_mmpareto, 
            params_bmml=params_bmml, 
            params_gblend=params_gblend, 
            params_pdf=params_pdf,
            params_omib=params_omib,
            params_aug=params_aug,
        )
        
        self.model = model
        self.params_optimizer = params_optimizer
        self.dataset = dataset

        self.loss = nn.BCEWithLogitsLoss()

        self.auroc_train = BinaryAUROC()
        self.auroc_val = BinaryAUROC()
        self.auroc_test = BinaryAUROC()

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="train")

        logits = shared_dict["logits"]
        y = shared_dict["y"]

        self.auroc_train.update(logits, y)

        return shared_dict["loss"]

    def validation_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="val")

        logits = shared_dict["logits"]
        y = shared_dict["y"]

        self.auroc_val.update(logits, y)

        return shared_dict["loss"]

    def test_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="test")

        logits = shared_dict["logits"]
        y = shared_dict["y"]

        self.auroc_test.update(logits, y)

        return shared_dict["loss"]

    def on_train_epoch_end(self):
        self.log("train/auroc", self.auroc_train.compute())
        self.auroc_train.reset()
    
    def on_validation_epoch_end(self):
        val_auroc = self.auroc_val.compute()
        self.log("val/auroc", val_auroc, sync_dist=True)
        self.auroc_val.reset()

    def on_test_epoch_end(self):
        test_auroc = self.auroc_test.compute()
        self.log("test/auroc", test_auroc, sync_dist=True)
        self.auroc_test.reset()

