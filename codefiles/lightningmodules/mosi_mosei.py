import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy, BinaryF1Score, MulticlassCalibrationError

from codefiles.lightningmodules.utils import (
    get_input,
    get_target,
    LightningModuleParent
)


class MOSI_Lightning_Module(LightningModuleParent):

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
        dataset: str = "mimic_symile",
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
        self.loss = nn.CrossEntropyLoss()  #nn.L1Loss()
        self.dataset = dataset

        self.acc_2_train = BinaryAccuracy()
        self.acc_2_val = BinaryAccuracy()
        self.acc_2_test = BinaryAccuracy()

        self.acc_7_train = MulticlassAccuracy(num_classes=7, average="micro")
        self.acc_7_val = MulticlassAccuracy(num_classes=7, average="micro")
        self.acc_7_test = MulticlassAccuracy(num_classes=7, average="micro")

        self.f1_train = BinaryF1Score()
        self.f1_val = BinaryF1Score()
        self.f1_test = BinaryF1Score()

        self.all_val_2_accs = []
        self.all_val_7_accs = []
        self.all_val_f1s = []

        self.ece = MulticlassCalibrationError(num_classes=7, norm="l1")
        self.mce = MulticlassCalibrationError(num_classes=7, norm="max")
        self.rmsce = MulticlassCalibrationError(num_classes=7, norm="l2")

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="train", convert_logits="mosi")

        logits = shared_dict["logits"]
        y = shared_dict["y"]

        # Filter non-zero entries
        #non_zeros = torch.tensor([i for i, e in enumerate(y) if e != 0])
        #binary_targets = (y[non_zeros] > 0).squeeze()
        #binary_predictions = (logits[non_zeros] > 0)

        predicted_classes = torch.argmax(logits, dim=1)
        non_neutral_mask = (y != 3)

        # Binary classification transformations
        binary_targets = (y[non_neutral_mask] > 3).long()
        binary_predictions = (predicted_classes[non_neutral_mask.squeeze()] > 3).long()
        
        # Update metrics
        if binary_predictions.numel() > 0:
            self.acc_2_train.update(binary_predictions.view(-1), binary_targets.view(-1))
            self.f1_train.update(binary_predictions.view(-1), binary_targets.view(-1))
        #self.acc_7_train.update(
        #    torch.round(torch.clamp(logits, -3, 3)),
        #    torch.round(torch.clamp(y.squeeze(), -3, 3))
        #)
        self.acc_7_train.update(predicted_classes.view(-1), y.view(-1))

        return shared_dict["loss"]

    def validation_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="val", convert_logits="mosi")

        logits = shared_dict["logits"]
        y = shared_dict["y"]

        # Filter non-zero entries
        #non_zeros = torch.tensor([i for i, e in enumerate(y) if e != 0])
        #binary_targets = (y[non_zeros] > 0).squeeze()
        #binary_predictions = (logits[non_zeros] > 0)

        predicted_classes = torch.argmax(logits, dim=1)
        non_neutral_mask = (y != 3)

        # Binary classification transformations
        binary_targets = (y[non_neutral_mask] > 3).long()
        binary_predictions = (predicted_classes[non_neutral_mask.squeeze()] > 3).long()
        
        # Update metrics
        if binary_predictions.numel() > 0:
            self.acc_2_val.update(binary_predictions.view(-1), binary_targets.view(-1))
            self.f1_val.update(binary_predictions.view(-1), binary_targets.view(-1))
        #self.acc_7_val.update(
        #    torch.round(torch.clamp(logits, -3, 3)),
        #    torch.round(torch.clamp(y.squeeze(), -3, 3))
        #)
        self.acc_7_val.update(predicted_classes.view(-1), y.view(-1))

        return shared_dict["loss"]

    def test_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="test", convert_logits="mosi")

        logits = shared_dict["logits"]
        y = shared_dict["y"]

        # Filter non-zero entries
        #non_zeros = torch.tensor([i for i, e in enumerate(y) if e != 0])
        #binary_targets = (y[non_zeros] > 0).squeeze()
        #binary_logits = (logits.squeeze()[non_zeros] > 0)

        predicted_classes = torch.argmax(logits, dim=1)
        non_neutral_mask = (y != 3)

        # Binary classification transformations
        binary_targets = (y[non_neutral_mask] > 3).long()
        binary_predictions = (predicted_classes[non_neutral_mask.squeeze()] > 3).long()
        
        # Update metrics 
        self.acc_2_test.update(binary_predictions.view(-1), binary_targets.view(-1))
        self.f1_test.update(binary_predictions.view(-1), binary_targets.view(-1))  
        #self.acc_7_test.update(
        #    torch.round(torch.clamp(logits, -3, 3)),
        #    torch.round(torch.clamp(y.squeeze(), -3, 3))
        #)
        self.acc_7_test.update(predicted_classes.view(-1), y.view(-1))

        return shared_dict["loss"]

    def on_train_epoch_end(self):
        self.log("train/acc_2_epoch", self.acc_2_train.compute())
        self.log("train/acc_7_epoch", self.acc_7_train.compute())
        self.log("train/f1_epoch", self.f1_train.compute())
        self.acc_2_train.reset()
        self.acc_7_train.reset()
        self.f1_train.reset()
    
    def on_validation_epoch_end(self):
        val_2_acc = self.acc_2_val.compute()
        self.all_val_2_accs.append(val_2_acc)
        max_val_2_acc = max(self.all_val_2_accs)
        self.log("val/acc_2_epoch", val_2_acc, sync_dist=True)
        self.log("val/acc_2_max", max_val_2_acc, sync_dist=True)

        val_7_acc = self.acc_7_val.compute()
        self.all_val_7_accs.append(val_7_acc)
        max_val_7_acc = max(self.all_val_7_accs)
        self.log("val/acc_7_epoch", val_7_acc, sync_dist=True)
        self.log("val/acc_7_max", max_val_7_acc, sync_dist=True)

        val_f1 = self.f1_val.compute()
        self.all_val_f1s.append(val_f1)
        max_val_f1 = max(self.all_val_f1s)
        self.log("val/f1_epoch", val_f1, sync_dist=True)
        self.log("val/f1_max", max_val_f1, sync_dist=True)

        #self.log("val/ece_epoch", self.ece.compute())
        #self.log("val/mce_epoch", self.mce.compute())
        #self.log("val/rmsce_epoch", self.rmsce.compute())

        self.acc_2_val.reset()
        self.acc_7_val.reset()
        self.f1_val.reset()

    def on_test_epoch_end(self):
        test_2_acc = self.acc_2_test.compute()
        self.log("test/acc_2_epoch", test_2_acc, sync_dist=True)
        self.acc_2_test.reset()

        test_7_acc = self.acc_7_test.compute()
        self.log("test/acc_7_epoch", test_7_acc, sync_dist=True)
        self.acc_7_test.reset()

        test_f1 = self.f1_test.compute()
        self.log("test/f1_epoch", test_f1, sync_dist=True)
        self.f1_test.reset()
