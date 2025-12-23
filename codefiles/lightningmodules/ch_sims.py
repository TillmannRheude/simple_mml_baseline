import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy, BinaryF1Score, MulticlassCalibrationError, MulticlassF1Score

from codefiles.lightningmodules.utils import (
    LightningModuleParent
)


class CH_Sims_Lightning_Module(LightningModuleParent):

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
        self.loss = nn.CrossEntropyLoss()
        self.dataset = dataset

        self.acc_2_train = BinaryAccuracy()
        self.acc_2_val = BinaryAccuracy()
        self.acc_2_test = BinaryAccuracy()

        self.acc_5_train = MulticlassAccuracy(num_classes=5)
        self.acc_5_val = MulticlassAccuracy(num_classes=5)
        self.acc_5_test = MulticlassAccuracy(num_classes=5)

        self.f1_train = MulticlassF1Score(num_classes=5)
        self.f1_val = MulticlassF1Score(num_classes=5)
        self.f1_test = MulticlassF1Score(num_classes=5)

        self.all_val_2_accs = []
        self.all_val_f1s = []
        self.all_val_5_accs = []

        self.five_classmapping = {
            -1.0: 0, -0.8: 0,
            -0.6: 1, -0.4: 1, -0.2: 1,
            0.0: 2,
            0.2: 3, 0.4: 3, 0.6: 3,
            0.8: 4, 1.0: 4
        }  # 0=negative, 1=weakly negative, 2=neutral, 3=weakly positive, 4=positive

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="train", convert_logits="ch_sims")

        logits = shared_dict["logits"]
        y = shared_dict["y"]

        predicted_classes = torch.argmax(logits, dim=1)
        non_neutral_mask = (y != 2)

        # Binary classification transformations
        binary_targets = (y[non_neutral_mask] > 2).long()
        binary_predictions = (predicted_classes[non_neutral_mask.squeeze()] > 2).long()
        
        # Update metrics
        if binary_predictions.numel() > 0:
            self.acc_2_train.update(binary_predictions.view(-1), binary_targets.view(-1))
            self.f1_train.update(binary_predictions.view(-1), binary_targets.view(-1))
        self.acc_5_train.update(predicted_classes.view(-1), y.view(-1))

        return shared_dict["loss"]

    def validation_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="val", convert_logits="ch_sims")

        logits = shared_dict["logits"]
        y = shared_dict["y"]

        predicted_classes = torch.argmax(logits, dim=1)
        non_neutral_mask = (y != 2)

        # Binary classification
        binary_targets = (y[non_neutral_mask] > 2).long()
        binary_predictions = (predicted_classes[non_neutral_mask.squeeze()] > 2).long()

        # Update metrics
        if binary_predictions.numel() > 0:
            self.acc_2_val.update(binary_predictions.view(-1), binary_targets.view(-1))
            self.f1_val.update(binary_predictions.view(-1), binary_targets.view(-1))
        self.acc_5_val.update(predicted_classes.view(-1), y.view(-1))

        return shared_dict["loss"]

    def test_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="test", convert_logits="ch_sims")

        logits = shared_dict["logits"]
        y = shared_dict["y"]

        predicted_classes = torch.argmax(logits, dim=1)
        non_neutral_mask = (y != 2)

        # Binary classification
        binary_targets = (y[non_neutral_mask] > 2).long()
        binary_predictions = (predicted_classes[non_neutral_mask.squeeze()] > 2).long()

        # Update metrics
        if binary_predictions.numel() > 0:
            self.acc_2_test.update(binary_predictions.view(-1), binary_targets.view(-1))
            self.f1_test.update(binary_predictions.view(-1), binary_targets.view(-1))  
        self.acc_5_test.update(predicted_classes.view(-1), y.view(-1))

        return shared_dict["loss"]

    def on_train_epoch_end(self):
        self.log("train/acc_2_epoch", self.acc_2_train.compute())
        self.log("train/f1_epoch", self.f1_train.compute())
        self.log("train/acc_5_epoch", self.acc_5_train.compute())
        self.acc_2_train.reset()
        self.f1_train.reset()
        self.acc_5_train.reset()

    def on_validation_epoch_end(self):
        val_2_acc = self.acc_2_val.compute()
        self.all_val_2_accs.append(val_2_acc)
        max_val_2_acc = max(self.all_val_2_accs)
        self.log("val/acc_2_epoch", val_2_acc, sync_dist=True)
        self.log("val/acc_2_max", max_val_2_acc, sync_dist=True)

        val_f1 = self.f1_val.compute()
        self.all_val_f1s.append(val_f1)
        max_val_f1 = max(self.all_val_f1s)
        self.log("val/f1_epoch", val_f1, sync_dist=True)
        self.log("val/f1_max", max_val_f1, sync_dist=True)

        val_5_acc = self.acc_5_val.compute()
        self.all_val_5_accs.append(val_5_acc)
        max_val_5_acc = max(self.all_val_5_accs)
        self.log("val/acc_5_epoch", val_5_acc, sync_dist=True)
        self.log("val/acc_5_max", max_val_5_acc, sync_dist=True)

        self.acc_2_val.reset()
        self.f1_val.reset()
        self.acc_5_val.reset()

    def on_test_epoch_end(self):
        test_2_acc = self.acc_2_test.compute()
        self.log("test/acc_2_epoch", test_2_acc, sync_dist=True)
        self.acc_2_test.reset()

        test_f1 = self.f1_test.compute()
        self.log("test/f1_epoch", test_f1, sync_dist=True)
        self.f1_test.reset()

        test_5_acc = self.acc_5_test.compute()
        self.log("test/acc_5_epoch", test_5_acc, sync_dist=True)
        self.acc_5_test.reset()
