import torch 

import torch.nn as nn

from torchmetrics.classification import BinaryCalibrationError, BinaryAUROC

from codefiles.lightningmodules.utils import ( 
    LightningModuleParent
)

class UKB_Lightning_Module(LightningModuleParent):

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
        dataset: str = "ukb",
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
        self.dataset = dataset
        
        self.loss = nn.BCEWithLogitsLoss()

        self.ignore_index = -100
        self.binary_eces = BinaryCalibrationError(norm="l1", ignore_index=self.ignore_index)
        self.binary_mces = BinaryCalibrationError(norm="max", ignore_index=self.ignore_index)
        self.binary_rmsces = BinaryCalibrationError(norm="l2", ignore_index=self.ignore_index)
        self.binary_eces_test = BinaryCalibrationError(norm="l1", ignore_index=self.ignore_index)
        self.binary_mces_test = BinaryCalibrationError(norm="max", ignore_index=self.ignore_index)
        self.binary_rmsces_test = BinaryCalibrationError(norm="l2", ignore_index=self.ignore_index)

        self.metric_train_macro = BinaryAUROC()
        self.metric_val_macro = BinaryAUROC()
        self.metric_test_macro = BinaryAUROC()
        self.metric_train_micro = BinaryAUROC()
        self.metric_val_micro = BinaryAUROC()
        self.metric_test_micro = BinaryAUROC()

        self.all_val_aurocs_macro, self.all_val_aurocs_micro = [], []
        self.all_test_aurocs_macro, self.all_test_aurocs_micro = [], []

        self.clf_labels = ["10y_mortality"]  # "5y_mortality", 

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="train")

        self.metric_train_macro.update(shared_dict["logits"].view(-1, 1), shared_dict["y"].view(-1, 1)) 
        self.metric_train_micro.update(shared_dict["logits"].view(-1, 1), shared_dict["y"].view(-1, 1)) 
    
        return shared_dict["loss"]

    def validation_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="val")

        self.metric_val_macro.update(shared_dict["logits"].view(-1, 1), shared_dict["y"].view(-1, 1))
        self.metric_val_micro.update(shared_dict["logits"].view(-1, 1), shared_dict["y"].view(-1, 1))
            
        shared_dict["y"][torch.isnan(shared_dict["y"])] = self.ignore_index
        logits = shared_dict["logits"].to(torch.float32)
        y = shared_dict["y"].to(torch.float32)

        return shared_dict["loss"]
    
    def test_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="test")

        self.metric_test_macro.update(shared_dict["logits"].view(-1, 1), shared_dict["y"].view(-1, 1))
        self.metric_test_micro.update(shared_dict["logits"].view(-1, 1), shared_dict["y"].view(-1, 1))
        shared_dict["y"][torch.isnan(shared_dict["y"])] = self.ignore_index
        logits = shared_dict["logits"].to(torch.float32)
        y = shared_dict["y"].to(torch.float32)

        return shared_dict["loss"]

    def on_train_epoch_end(self):
        if len(self.params_gblend) > 0:
            super().on_train_epoch_end()

        self.log("train/auroc_macro", self.metric_train_macro.compute(), sync_dist=False)
        self.log("train/auroc_micro", self.metric_train_micro.compute(), sync_dist=False)

        self.logits_train = []
        self.targets_train = []
        self.masks_train = []
        self.metric_train_macro.reset()
        self.metric_train_micro.reset()

    def on_validation_epoch_end(self):
        if len(self.params_gblend) > 0:
            super().on_validation_epoch_end()

        if not self.trainer.sanity_checking:
            val_acc = self.metric_val_macro.compute()
            self.all_val_aurocs_macro.append(val_acc)
            max_val_acc = max(self.all_val_aurocs_macro)
            self.log("val/auroc_macro_epoch", val_acc, sync_dist=False)
            self.log("val/auroc_macro_max", max_val_acc, sync_dist=False)

            val_acc = self.metric_val_micro.compute()
            self.all_val_aurocs_micro.append(val_acc)
            max_val_acc = max(self.all_val_aurocs_micro)
            self.log("val/auroc_micro_epoch", val_acc, sync_dist=False)
            self.log("val/auroc_micro_max", max_val_acc, sync_dist=False)

            self.metric_val_macro.reset()
            self.metric_val_micro.reset()

    def on_test_epoch_end(self):
        test_acc = self.metric_test_macro.compute()
        self.all_test_aurocs_macro.append(test_acc)
        max_test_acc = max(self.all_test_aurocs_macro)
        self.log("test/auroc_macro_epoch", test_acc, sync_dist=False)
        self.log("test/auroc_macro_max", max_test_acc, sync_dist=False)

        test_acc = self.metric_test_micro.compute()
        self.all_test_aurocs_micro.append(test_acc)
        max_test_acc = max(self.all_test_aurocs_micro)
        self.log("test/auroc_micro_epoch", test_acc, sync_dist=False)
        self.log("test/auroc_micro_max", max_test_acc, sync_dist=False)

        self.metric_test_macro.reset()
        self.metric_test_micro.reset()


