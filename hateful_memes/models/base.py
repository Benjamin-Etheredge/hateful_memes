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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning import Trainer
import wandb

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

        # ic.enable()
        y = batch['label']
        y_hat = self(batch)
        y_hat = torch.squeeze(y_hat)
        # if batch_idx % 1 == 0:
        # if batch_idx == 0:
        #     ic(batch, y_hat, torch.sigmoid(y_hat), y)
            # ic(y_hat, torch.sigmoid(y_hat), y)
            # input()
        loss = F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))
        # loss = F.binary_cross_entropy(torch.sigmoid(y_hat), y.to(y_hat.dtype))

        self.train_acc(y_hat, y)
        self.train_f1(y_hat, y)
        self.train_auroc(y_hat, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("train/f1", self.train_f1, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, batch_size=len(y))

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['label']
        y_hat = self(batch)
        y_hat = torch.squeeze(y_hat)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))

        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)
        self.val_auroc(y_hat, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("val/accuracy", self.val_acc, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("val/f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(y))
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, batch_size=len(y))

        return loss

    def configure_optimizers(self):
        optimizer= torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": ReduceLROnPlateau(optimizer, patience=8, verbose=True),
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
        stopping_patience=20,
    ):
    logger = get_project_logger(project=project, save_dir=log_dir, offline=fast_dev_run)
    # TODO pull out lr and maybe arg optimizer

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

    stw = StochasticWeightAveraging()

    trainer = Trainer(
        devices=1, 
        accelerator='auto',
        logger=logger,
        max_epochs=epochs,
        gradient_clip_val=grad_clip,
        track_grad_norm=2, 
        fast_dev_run=fast_dev_run, 
        auto_lr_find=True,
        auto_scale_batch_size='power',
        # precision=16,
        # amp_backend='native',
        # detect_anomaly=True,
        callbacks=[checkpoint_callback, early_stopping, stw])

    data = MaeMaeDataModule(batch_size=batch_size)

    if not fast_dev_run:
        # TODO should I move datamodule inside lightning module?
        result = trainer.tune(
            model, 
            scale_batch_size_kwargs=dict(max_trials=6),
            lr_find_kwargs=dict(num_training=100),
            datamodule=data,
        )

        ic.enable()
        ic(result)
        lr_find = result['lr_find']
        plt = lr_find.plot(suggest=True)
        wandb.log({"lr_plot": plt})

        # new_lr = trainer.tuner.lr_find.suggestion()
        # model.hparams.lr = new_lr
        # model.lr = new_lr

    trainer.fit(model, datamodule=data)

    # # Setup data for predictions
    # data = MaeMaeDataModule(batch_size=batch_size)
    # data.setup(None)
    # train_data = data.train_dataloader(shuffle=False, drop_last=False)
    # train_labels = []
    # for batch in train_data:
    #     train_labels += batch['label']
    # val_data = data.val_dataloader()
    # val_labels = []
    # for batch in val_data:
    #     val_labels += batch['label']

    # train_pred, val_pred = trainer.predict(model, dataloaders=[train_data, val_data])
    # ic(train_pred, val_pred)

    # train_cm = wandb.plot.confusion_matrix(
    #     y_true=train_labels,
    #     preds=train_pred,
    #     class_names=['not hateful', 'hateful'],
    # )
    # val_cm = wandb.plot.confusion_matrix(
    #     y_true=val_labels,
    #     preds=val_pred,
    #     class_names=['not hateful', 'hateful'],
    # )
    # wandb.log({"train_cm": train_cm})
    # wandb.log({"val_cm": val_cm})