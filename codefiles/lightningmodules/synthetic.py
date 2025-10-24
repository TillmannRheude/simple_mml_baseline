
import torch
import torch.nn as nn 

from torchmetrics.classification import MulticlassAccuracy, MulticlassCalibrationError

from codefiles.lightningmodules.utils import (
    get_input,
    get_target,
    LightningModuleParent
)


class Synthetic_Lightning_Module(LightningModuleParent):

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
        dataset: str = "fmnist",
        manual_opt: bool = True,
        params_ogm: dict = {
                "use_ge": True,
                "ge_noise_level": 0.1,
        },
        params_arl: dict = {},
        params_dgl: dict = {},
        params_mcr: dict = {},
        params_mmpareto: dict = {},
    ) -> None: 
        super().__init__(manual_opt=manual_opt, params_ogm=params_ogm, params_arl=params_arl, params_dgl=params_dgl, params_mcr=params_mcr, params_mmpareto=params_mmpareto)
        self.model = model
        self.params_optimizer = params_optimizer
        self.loss = nn.CrossEntropyLoss()
        self.dataset = dataset

        self.acc_train = MulticlassAccuracy(num_classes=10)
        self.acc_val = MulticlassAccuracy(num_classes=10)
        self.acc_test = MulticlassAccuracy(num_classes=10)
        self.ece = MulticlassCalibrationError(num_classes=10, norm="l1")
        self.mce = MulticlassCalibrationError(num_classes=10, norm="max")
        self.rmsce = MulticlassCalibrationError(num_classes=10, norm="l2")
        self.all_val_accs = []

        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch)

        self.acc_train.update(torch.argmax(shared_dict["logits"], dim=1).view(*shared_dict["y"].shape), shared_dict["y"])
    
        return shared_dict["loss"]
    
    def validation_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch)

        self.acc_val.update(torch.argmax(shared_dict["logits"], dim=1).view(*shared_dict["y"].shape), shared_dict["y"])
        self.ece.update(shared_dict["logits"], shared_dict["y"])
        self.mce.update(shared_dict["logits"], shared_dict["y"])
        self.rmsce.update(shared_dict["logits"], shared_dict["y"])

        return shared_dict["loss"]
    
    def test_step(self, batch, batch_idx):
        x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)

        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log("test/loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.acc_test.update(torch.argmax(logits, dim=1).view(*y.shape), y)

        return logits

    def on_train_epoch_end(self):
        self.log("train/acc_epoch", self.acc_train.compute())
        self.acc_train.reset()

    def on_validation_epoch_end(self):
        val_acc = self.acc_val.compute()
        self.all_val_accs.append(val_acc)
        max_val_acc = max(self.all_val_accs)

        self.log("val/acc_epoch", val_acc, sync_dist=True)
        self.log("val/acc_max", max_val_acc, sync_dist=True)
        self.log("val/ece_epoch", self.ece.compute())
        self.log("val/mce_epoch", self.mce.compute())
        self.log("val/rmsce_epoch", self.rmsce.compute())

        self.acc_val.reset()

    def on_test_epoch_end(self):
        self.log("test/acc_epoch", self.acc_test.compute())
        self.acc_test.reset()

