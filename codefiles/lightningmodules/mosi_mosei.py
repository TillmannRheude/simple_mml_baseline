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
        dataset: str = "mosi"
    ):
        super().__init__()
        self.model = model
        self.params_optimizer = params_optimizer
        self.loss = nn.L1Loss()
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
        x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)
        logits = self.forward(x).to(y.dtype)
        
        # Convert logits to predicted values (-3 to 3 range)
        values = torch.tensor([-3, -2, -1, 0, 1, 2, 3], device=logits.device)
        predictions = (F.softmax(logits, dim=1) * values.view(1, -1)).sum(dim=1)
        
        # Calculate loss
        loss = self.loss(predictions, y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Filter non-zero entries
        non_zeros = torch.tensor([i for i, e in enumerate(y) if e != 0])
        binary_targets = (y[non_zeros] > 0)
        binary_predictions = (predictions[non_zeros] > 0)
        
        # Update metrics
        self.acc_2_train.update(binary_predictions, binary_targets)
        self.acc_7_train.update(
            torch.round(torch.clamp(predictions, -3, 3)),
            torch.round(torch.clamp(y, -3, 3))
        )
        self.f1_train.update(binary_predictions, binary_targets)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)
        logits = self.forward(x).to(y.dtype)
        
        # Convert logits to predicted values (-3 to 3 range)
        values = torch.tensor([-3, -2, -1, 0, 1, 2, 3], device=logits.device)
        predictions = (F.softmax(logits, dim=1) * values.view(1, -1)).sum(dim=1)
        
        # Calculate loss
        loss = self.loss(predictions, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Filter non-zero entries
        non_zeros = torch.tensor([i for i, e in enumerate(y) if e != 0])
        binary_targets = (y[non_zeros] > 0)
        binary_predictions = (predictions[non_zeros] > 0)
        
        # Update metrics
        self.acc_2_val.update(binary_predictions, binary_targets)
        self.acc_7_val.update(
            torch.round(torch.clamp(predictions, -3, 3)),
            torch.round(torch.clamp(y, -3, 3))
        )
        self.f1_val.update(binary_predictions, binary_targets)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)

        logits = self.forward(x)
        loss = self.loss(logits.squeeze(), y.squeeze())
        self.log("test/loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        non_zeros = torch.tensor([i for i, e in enumerate(y) if e != 0])
        binary_targets = (y[non_zeros] > 0)
        binary_logits = (logits.squeeze()[non_zeros] > 0)
        
        self.acc_2_test.update(binary_logits, binary_targets)
        self.acc_7_test.update(torch.round(torch.clamp(logits, -3, 3)).squeeze(), torch.round(torch.clamp(y, -3, 3)))
        self.f1_test.update(binary_logits, binary_targets)  

        return logits

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
