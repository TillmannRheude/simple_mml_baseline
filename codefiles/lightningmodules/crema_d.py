import torch 
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from codefiles.lightningmodules.utils import LightningModuleParent

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
        self.loss = nn.CrossEntropyLoss()

        num_classes = 6  # 24 
        self.acc_train = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.acc_val = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.acc_test = MulticlassAccuracy(num_classes=num_classes, average="micro")

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

        return shared_dict["loss"]
    
    def test_step(self, batch, batch_idx):
        shared_dict = self.shared_step(batch, set="test")
        y = shared_dict["y"].long()
        logits = shared_dict["logits"]
        
        self.acc_test.update(logits, y)

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

    def on_test_epoch_end(self):
        self.log("test/acc", self.acc_test.compute(), sync_dist=False)

        self.acc_test.reset()