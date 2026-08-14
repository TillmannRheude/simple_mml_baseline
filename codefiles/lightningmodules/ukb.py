"""PyTorch Lightning training module for SimBaMM on UK Biobank."""

from __future__ import annotations

from collections.abc import Mapping

import pytorch_lightning as pl
import schedulefree
import torch
from torch import nn
from torchmetrics.classification import BinaryAUROC

from codefiles.architecture import SimBaMM


class UKBLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: SimBaMM,
        modality_names: list[str],
        optimizer_config: Mapping,
    ) -> None:
        super().__init__()
        self.model = model
        self.modality_names = modality_names
        self.optimizer_config = dict(optimizer_config)
        self.loss = nn.BCEWithLogitsLoss()
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, inputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(inputs)

    def _unpack_batch(
        self, batch: Mapping[str, object]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        inputs = [batch[name]["tabular_data"] for name in self.modality_names]
        target = batch["labels"]["tabular_data"].float()
        return inputs, target

    def _shared_step(self, batch: Mapping[str, object], stage: str) -> torch.Tensor:
        inputs, target = self._unpack_batch(batch)
        logits = self(inputs)["logits"]
        target = target.reshape_as(logits)
        loss = self.loss(logits, target)

        metric = getattr(self, f"{stage}_auroc")
        metric.update(logits.reshape(-1), target.long().reshape(-1))
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=target.shape[0],
        )
        return loss

    def training_step(self, batch: Mapping[str, object], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Mapping[str, object], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Mapping[str, object], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def on_train_epoch_end(self) -> None:
        self.log("train/auroc", self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        self.val_auroc.reset()

    def on_test_epoch_end(self) -> None:
        self.log("test/auroc", self.test_auroc.compute(), prog_bar=True)
        self.test_auroc.reset()

    def on_train_epoch_start(self) -> None:
        self._set_schedulefree_mode(training=True)

    def on_validation_start(self) -> None:
        self._set_schedulefree_mode(training=False)

    def on_test_start(self) -> None:
        self._set_schedulefree_mode(training=False)

    def _set_schedulefree_mode(self, training: bool) -> None:
        optimizer = self.optimizers()
        if self.optimizer_config["name"] == "schedulefree_adamw":
            optimizer.train() if training else optimizer.eval()

    def configure_optimizers(self):
        if self.optimizer_config["name"] != "schedulefree_adamw":
            raise ValueError("Only schedulefree_adamw is supported in this SimBaMM branch.")
        return schedulefree.AdamWScheduleFree(
            self.parameters(),
            lr=self.optimizer_config["lr"],
            weight_decay=self.optimizer_config["weight_decay"],
            eps=self.optimizer_config["eps"],
            warmup_steps=self.optimizer_config["warmup_steps"],
            betas=tuple(self.optimizer_config["betas"]),
        )
