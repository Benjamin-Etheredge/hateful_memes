import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision import transforms as T
from icecream import ic
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class BaseMaeMaeModel(LightningModule):

    def forward(self, batch):
        raise NotImplemented

    def _shared_step(self, batch):

        y = batch['label']
        y_hat = self(batch)
        loss = F.binary_cross_entropy(y_hat, y.to(y_hat.dtype))

        acc = torch.sum(torch.round(y_hat) == y.data) / (y.shape[0] * 1.0)

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
