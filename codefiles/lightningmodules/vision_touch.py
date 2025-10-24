import torch
import torch.nn as nn
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy

from codefiles.lightningmodules.utils import (
    NaNMultilabelAUROC,
    LightningModuleParent
)
from codefiles.losses.nanbce import WeightedNaNBCEWithLogitsLoss

class VisionTouch_Lightning_Module(LightningModuleParent):
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
        dataset: str = "vision_touch",
        manual_opt: bool = True,
        params_ogm: dict = {
            "use_ge": True,
            "ge_noise_level": 0.1,
        },
        params_arl: dict = {},
        params_dgl: dict = {},
        params_mcr: dict = {},
        params_mmpareto: dict = {},
        params_bmml: dict = {},
        params_gblend: dict = {},
        params_pdf: dict = {},
        params_pmr: dict = {},
        params_omib: dict = {},
        params_smil: dict = {},
        params_avmc: dict = {},
        params_ebr: dict = {},
        params_simmlm: dict = {},
    ) -> None: 
        super().__init__(
            manual_opt=manual_opt, 
            params_ogm=params_ogm, 
            params_arl=params_arl, 
            params_dgl=params_dgl, 
            params_mcr=params_mcr, 
            params_mmpareto=params_mmpareto, 
            params_bmml=params_bmml, 
            params_gblend=params_gblend, 
            params_pdf=params_pdf,
            params_pmr=params_pmr,
            params_omib=params_omib,
            params_smil=params_smil,
            params_avmc=params_avmc,
            params_ebr=params_ebr,
            params_simmlm=params_simmlm,
        )
        
        self.model = model
        self.params_optimizer = params_optimizer
        self.dataset = dataset
        self.loss = WeightedNaNBCEWithLogitsLoss()

        # Only one label (binary prediction)
        self.binary_acc_train = BinaryAccuracy()
        self.binary_acc_val = BinaryAccuracy()
        self.binary_acc_test = BinaryAccuracy()
        self.all_val_accs = []
        self.all_test_accs = []

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="train")
        logits = shared_dict["logits"]
        y = shared_dict["y"].float().view(-1, 1)

        sigmoid_logits = torch.sigmoid(logits)
        self.binary_acc_train.update(sigmoid_logits, y)
        return shared_dict["loss"]

    def validation_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="val")
        logits = shared_dict["logits"]
        y = shared_dict["y"].float().view(-1, 1)

        sigmoid_logits = torch.sigmoid(logits)
        self.binary_acc_val.update(sigmoid_logits, y)
        return shared_dict["loss"]

    def test_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="test")
        logits = shared_dict["logits"]
        y = shared_dict["y"].float().view(-1, 1)

        sigmoid_logits = torch.sigmoid(logits)
        self.binary_acc_test.update(sigmoid_logits, y)
        return shared_dict["loss"]

    def on_train_epoch_end(self):
        self.log("train/acc", self.binary_acc_train.compute(), sync_dist=False)
        self.binary_acc_train.reset()

    def on_validation_epoch_end(self):
        val_acc = self.binary_acc_val.compute()
        self.all_val_accs.append(val_acc)
        max_val_acc = max(self.all_val_accs)
        self.log("val/acc_epoch", val_acc, sync_dist=False)
        self.log("val/acc_max", max_val_acc, sync_dist=False)
        self.binary_acc_val.reset()

    def on_test_epoch_end(self):
        test_acc = self.binary_acc_test.compute()
        self.all_test_accs.append(test_acc)
        max_test_acc = max(self.all_test_accs)
        self.log("test/acc_epoch", test_acc, sync_dist=False)
        self.log("test/acc_max", max_test_acc, sync_dist=False)
        self.binary_acc_test.reset()
