import imp
import os
from icecream import ic

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision import transforms as T
import torchmetrics

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer

from hateful_memes.utils import get_project_logger
from hateful_memes.data.hateful_memes import MaeMaeDataModule


class BaseMaeMaeModel(LightningModule):

    def __init__(self) -> None:
        super().__init__()

        # TODO log for each metric through macro
        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1(average="micro")
        self.train_auroc = torchmetrics.AUROC(average="micro")
        self.val_acc = torchmetrics.Accuracy()
        self.val_f1 = torchmetrics.F1(average="micro")
        self.val_auroc = torchmetrics.AUROC(average="micro")

    def forward(self, batch):
        raise NotImplemented
    
    def training_step(self, batch, batch_idx):

        y = batch['label']
        y_hat = self(batch)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))

        self.train_acc(y_hat, y)
        self.train_f1(y_hat, y)
        self.train_auroc(y_hat, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/accuracy", self.train_acc, on_step=False, on_epoch=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['label']
        y_hat = self(batch)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))

        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)
        self.val_auroc(y_hat, y)

        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log("val/accuracy", self.val_acc, on_step=False, on_epoch=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer= torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": ReduceLROnPlateau(optimizer, patience=3, verbose=True),
            "monitor": "train/loss",
        }


def base_train(
        *, 
        model, 
        project, 
        epochs, 
        batch_size, 
        model_dir='/tmp', 
        log_dir=None, 
        grad_clip=1.0, 
        fast_dev_run=False,
        monitor_metric="val/loss",
        monitor_metric_mode="min",
        stopping_patience=10,
    ):
    logger = get_project_logger(project=project, save_dir=log_dir, offline=fast_dev_run)

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode=monitor_metric_mode,
        dirpath=model_dir, 
        # filename="{epoch}-{step}-{loss:.4f}",
        verbose=True,
        save_top_k=1)

    early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=stopping_patience, 
            mode=monitor_metric_mode,
            verbose=True)

    trainer = Trainer(
        devices=1, 
        accelerator='auto',
        logger=logger,
        max_epochs=epochs,
        gradient_clip_val=grad_clip,
        track_grad_norm=2, 
        fast_dev_run=fast_dev_run, 
        callbacks=[checkpoint_callback, early_stopping])

    # TODO should I move module inside lightning module?
    trainer.fit(
        model, 
        datamodule=MaeMaeDataModule(
            batch_size=batch_size, 
        )
    )