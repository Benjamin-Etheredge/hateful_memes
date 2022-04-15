import os
from icecream import ic

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision import transforms as T

from pytorch_lightning import LightningModule


class BaseMaeMaeModel(LightningModule):

    def forward(self, batch):
        raise NotImplemented

    def _shared_step(self, batch):

        y = batch['label']
        y_hat = self(batch)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))

        acc = torch.sum(torch.round(torch.sigmoid(y_hat)) == y.data) / (y.shape[0] * 1.0)

        return loss, acc

    def training_step(self, batch, batch_idx):
        # x_img, x_txt = self.preprocess(x_img, x_txt)
        loss, acc = self._shared_step(batch)
        self.log("train/loss", loss, batch_size=len(batch['label']), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/acc", acc, batch_size=len(batch['label']), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("val/loss", loss, batch_size=len(batch['label']), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc", acc, batch_size=len(batch['label']), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer= torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": ReduceLROnPlateau(optimizer, patience=3, verbose=True),
            "monitor": "train/loss",
        }
