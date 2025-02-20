import torch 

import torch.nn as nn

from torchmetrics.classification import MulticlassAccuracy, BinaryCalibrationError

from codefiles.lightningmodules.utils import (
    WeightedNaNBCEWithLogitsLoss, 
    NaNMultilabelAUROC,
    get_input,
    get_target,
    LightningModuleParent
)

class MIMIC_Lightning_Module(LightningModuleParent):

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
        dataset: str = "mimic_symile"
    ) -> None: 
        super().__init__()
        self.model = model

        self.params_optimizer = params_optimizer

        self.dataset = dataset
        
        self.loss = WeightedNaNBCEWithLogitsLoss()

        self.acc_train = MulticlassAccuracy(num_classes=10)

        self.ignore_index = -100
        self.binary_eces = [BinaryCalibrationError(norm="l1", ignore_index=self.ignore_index) for _ in range(10)]
        self.binary_mces = [BinaryCalibrationError(norm="max", ignore_index=self.ignore_index) for _ in range(10)]
        self.binary_rmsces = [BinaryCalibrationError(norm="l2", ignore_index=self.ignore_index) for _ in range(10)]
        self.binary_eces_test = [BinaryCalibrationError(norm="l1", ignore_index=self.ignore_index) for _ in range(10)]
        self.binary_mces_test = [BinaryCalibrationError(norm="max", ignore_index=self.ignore_index) for _ in range(10)]
        self.binary_rmsces_test = [BinaryCalibrationError(norm="l2", ignore_index=self.ignore_index) for _ in range(10)]
        # get AUROC for all labels together
        self.metric_train_macro = NaNMultilabelAUROC(num_labels=10, average='macro')
        self.metric_val_macro = NaNMultilabelAUROC(num_labels=10, average='macro')
        self.metric_test_macro = NaNMultilabelAUROC(num_labels=10, average='macro')
        self.metric_train_micro = NaNMultilabelAUROC(num_labels=10, average='micro')
        self.metric_val_micro = NaNMultilabelAUROC(num_labels=10, average='micro')
        self.metric_test_micro = NaNMultilabelAUROC(num_labels=10, average='micro')
        # get AUROC of every label
        self.metrics_detailed = [NaNMultilabelAUROC(num_labels=10, average=None) for i in range(3)]
        self.all_val_aurocs_macro, self.all_val_aurocs_micro = [], []
        self.all_test_aurocs_macro, self.all_test_aurocs_micro = [], []

        self.clf_labels = ["Fracture", "Enlarged Cardiomediastinum", "Consolidation", "Atelectasis", 
                            "Edema", "Cardiomegaly", "Lung Lesion", "Lung Opacity", 
                            "Pneumonia", "Pneumothorax"]

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)

        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log("train/loss", loss.detach().item(), on_epoch=True, prog_bar=True, logger=True)
        
        self.metric_train_macro.update(logits.view(-1, 10), y.view(-1, 10)) 
        self.metric_train_micro.update(logits.view(-1, 10), y.view(-1, 10)) 
        self.metrics_detailed[0].update(logits.view(-1, 10), y.view(-1, 10))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)

        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)

        self.metric_val_macro.update(logits.view(-1, 10), y.view(-1, 10))
        self.metric_val_micro.update(logits.view(-1, 10), y.view(-1, 10))
        self.metrics_detailed[1].update(logits.view(-1, 10), y.view(-1, 10))
        for i in range(10):
            y[torch.isnan(y)] = self.ignore_index
            logits = logits.to(torch.float32)
            y = y.to(torch.float32)
            self.binary_eces[i].update(nn.Sigmoid()(logits[:, i]), y[:, i])
            self.binary_mces[i].update(nn.Sigmoid()(logits[:, i]), y[:, i])
            self.binary_rmsces[i].update(nn.Sigmoid()(logits[:, i]), y[:, i])

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)

        logits = self.forward(x)
        loss = self.loss(logits, y)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True, logger=True)

        self.metric_test_macro.update(logits.view(-1, 10), y.view(-1, 10))
        self.metric_test_micro.update(logits.view(-1, 10), y.view(-1, 10))
        self.metrics_detailed[2].update(logits.view(-1, 10), y.view(-1, 10))
        for i in range(10):
            y[torch.isnan(y)] = self.ignore_index
            self.binary_eces_test[i].update(nn.Sigmoid()(logits[:, i]), y[:, i])
            self.binary_mces_test[i].update(nn.Sigmoid()(logits[:, i]), y[:, i])
            self.binary_rmsces_test[i].update(nn.Sigmoid()(logits[:, i]), y[:, i])

        return loss

    def on_train_epoch_end(self):
        self.log("train/auroc_macro", self.metric_train_macro.compute(), sync_dist=True)
        self.log("train/auroc_micro", self.metric_train_micro.compute(), sync_dist=True)
        auroc_detailed = self.metrics_detailed[0].compute()

        self.logits_train = []
        self.targets_train = []
        self.masks_train = []
        self.metric_train_macro.reset()
        self.metric_train_micro.reset()
        self.metrics_detailed[0].reset()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            val_acc = self.metric_val_macro.compute()
            self.all_val_aurocs_macro.append(val_acc)
            max_val_acc = max(self.all_val_aurocs_macro)
            self.log("val/auroc_macro_epoch", val_acc, sync_dist=True)
            self.log("val/auroc_macro_max", max_val_acc, sync_dist=True)

            val_acc = self.metric_val_micro.compute()
            self.all_val_aurocs_micro.append(val_acc)
            max_val_acc = max(self.all_val_aurocs_micro)
            self.log("val/auroc_micro_epoch", val_acc, sync_dist=True)
            self.log("val/auroc_micro_max", max_val_acc, sync_dist=True)

            binary_eces = [self.binary_eces[i].compute() for i in range(10)]
            binary_mces = [self.binary_mces[i].compute() for i in range(10)]
            binary_rmsces = [self.binary_rmsces[i].compute() for i in range(10)]
            for i in range(10):
                self.binary_eces[i].reset()
                self.binary_mces[i].reset()
                self.binary_rmsces[i].reset()
            mean_ece = torch.mean(torch.tensor(binary_eces))
            mean_mce = torch.mean(torch.tensor(binary_mces))
            mean_rmsce = torch.mean(torch.tensor(binary_rmsces))
            self.log("val/ece_epoch", mean_ece, sync_dist=True)
            self.log("val/mce_epoch", mean_mce, sync_dist=True)
            self.log("val/rmsce_epoch", mean_rmsce, sync_dist=True)

        self.metric_val_macro.reset()
        self.metric_val_micro.reset()
        self.metrics_detailed[1].reset()

    def on_test_epoch_end(self):
        test_acc = self.metric_test_macro.compute()
        self.all_test_aurocs_macro.append(test_acc)
        max_test_acc = max(self.all_test_aurocs_macro)
        self.log("test/auroc_macro_epoch", test_acc, sync_dist=True)
        self.log("test/auroc_macro_max", max_test_acc, sync_dist=True)

        test_acc = self.metric_test_micro.compute()
        self.all_test_aurocs_micro.append(test_acc)
        max_test_acc = max(self.all_test_aurocs_micro)
        self.log("test/auroc_micro_epoch", test_acc, sync_dist=True)
        self.log("test/auroc_micro_max", max_test_acc, sync_dist=True)

        auroc_detailed = self.metrics_detailed[2].compute()

        clf_labels = ["test/" + clf for clf in self.clf_labels]
        label_auroc_dict = {lab: auroc for lab, auroc in zip(clf_labels, auroc_detailed)}
        self.log_dict(label_auroc_dict, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        binary_eces = [self.binary_eces_test[i].compute() for i in range(10)]
        binary_mces = [self.binary_mces_test[i].compute() for i in range(10)]
        binary_rmsces = [self.binary_rmsces_test[i].compute() for i in range(10)]
        for i in range(10):
            self.binary_eces_test[i].reset()
            self.binary_mces_test[i].reset()
            self.binary_rmsces_test[i].reset()
        mean_ece = torch.mean(torch.tensor(binary_eces))
        mean_mce = torch.mean(torch.tensor(binary_mces))
        mean_rmsce = torch.mean(torch.tensor(binary_rmsces))
        self.log("test/ece_epoch", mean_ece, sync_dist=True)
        self.log("test/mce_epoch", mean_mce, sync_dist=True)
        self.log("test/rmsce_epoch", mean_rmsce, sync_dist=True)

        self.metric_test_macro.reset()
        self.metric_test_micro.reset()
        self.metrics_detailed[2].reset()


