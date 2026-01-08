import torch 
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from codefiles.lightningmodules.utils import LightningModuleParent

from codefiles.transformer import Multimodal_Transformer
from codefiles.methods.aug.aug import AUG_Transformer
from codefiles.lightningmodules.utils import get_input, get_target

class CREMAD_Lightning_Module(LightningModuleParent):

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
        dataset: str = "crema_d",
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
        self.loss = nn.CrossEntropyLoss()

        num_classes = 6  # 24 
        self.acc_train = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.acc_val = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.acc_test = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self.unimodal_accs = [MulticlassAccuracy(num_classes=num_classes, average="micro").to(self.device) for _ in range(2)]
        self.unimodal_accs_test = [MulticlassAccuracy(num_classes=num_classes, average="micro").to(self.device) for _ in range(2)]

        self.all_val_accs, self.all_test_accs = [], []

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="train")
        y = shared_dict["y"].long()
        logits = shared_dict["logits"]
        
        self.acc_train.update(logits, y) 
    
        return shared_dict["loss"]

    def validation_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="val")
        y = shared_dict["y"].long()
        logits = shared_dict["logits"]

        self.acc_val.update(logits, y)

        if isinstance(self.model.transformer, Multimodal_Transformer):
            x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)

            unimodal_accs = self.model._unimodal_probing(x, y)
            for i, unimodal_acc in enumerate(self.unimodal_accs):
                self.unimodal_accs[i].to(self.device)
                self.unimodal_accs[i].update(unimodal_accs[:, i, :], y)
        

        if isinstance(self.model.transformer, AUG_Transformer):
            x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)
            unimodal_logits = self.model(x, y)["unimodal_logits"]
            for i in range(2):
                self.unimodal_accs[i].to(self.device)
                self.unimodal_accs[i].update(unimodal_logits[i], y)

        return shared_dict["loss"]
    
    def test_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="test")
        y = shared_dict["y"].long()
        logits = shared_dict["logits"]
        
        self.acc_test.update(logits, y)

        if isinstance(self.model.transformer, Multimodal_Transformer):
            x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)

            unimodal_accs = self.model._unimodal_probing(x, y)
            for i, unimodal_acc in enumerate(self.unimodal_accs_test):
                self.unimodal_accs_test[i].to(self.device)
                self.unimodal_accs_test[i].update(unimodal_accs[:, i, :], y)

        if isinstance(self.model.transformer, AUG_Transformer):
            x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)
            unimodal_logits = self.model(x, y)["unimodal_logits"]
            for i in range(2):
                self.unimodal_accs_test[i].to(self.device)
                self.unimodal_accs_test[i].update(unimodal_logits[i], y)

        return shared_dict["loss"]

    def on_train_epoch_end(self):
        self.log("train/acc", self.acc_train.compute(), sync_dist=False)

        self.acc_train.reset()

    def on_validation_epoch_end(self):
        self.log("val/acc", self.acc_val.compute(), sync_dist=False)
        self.all_val_accs.append(self.acc_val.compute())
        max_val_acc = max(self.all_val_accs)
        self.log("val/acc_max", max_val_acc, sync_dist=False)
        self.acc_val.reset()

        if isinstance(self.model.transformer, Multimodal_Transformer):
            for i, unimodal_acc in enumerate(self.unimodal_accs):
                self.log(f"val/acc_unimodal_{i}", unimodal_acc.compute(), sync_dist=False)
                self.unimodal_accs[i].reset()

        if isinstance(self.model.transformer, AUG_Transformer):
            for i in range(2):
                self.log(f"val/acc_unimodal_{i}", self.unimodal_accs[i].compute(), sync_dist=True)
                self.unimodal_accs[i].reset()

    def on_test_epoch_end(self):
        acc_test = self.acc_test.compute()
        self.log("test/acc", acc_test, sync_dist=True)
        self.acc_test.reset()

        if isinstance(self.model.transformer, Multimodal_Transformer):
            for i, unimodal_acc in enumerate(self.unimodal_accs_test):
                self.log(f"test/acc_unimodal_{i}", unimodal_acc.compute(), sync_dist=True)
                self.unimodal_accs_test[i].reset()

        if isinstance(self.model.transformer, AUG_Transformer):
            for i in range(2):
                self.log(f"test/acc_unimodal_{i}", self.unimodal_accs_test[i].compute(), sync_dist=True)
                self.unimodal_accs_test[i].reset()