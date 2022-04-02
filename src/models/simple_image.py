
import click
from dvclive.lightning import DvcLiveLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule, Trainer
import os

from models.baseline import BaseMaeMaeModel
from data.hateful_memes import MaeMaeDataModule
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class SimpleImageMaeMaeModel(BaseMaeMaeModel):
    def __init__(
        self, 
        lr=0.003, 
        dense_dim=128, 
        dropout_rate=0.1,
        batch_norm=False,

    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        # TODO better batch norm usage and remove bias

        self.l1 = nn.Linear(43264, dense_dim)
        self.l2 = nn.Linear(dense_dim, dense_dim)
        self.l3 = nn.Linear(dense_dim, 1)

        self.lr = lr
        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.save_hyperparameters()

    def forward(self, batch):
        x_img = batch['image']
        x = x_img
        x = self.conv1(x)
        # if self.batch_norm:
            # x = F.batch_norm(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        # if self.batch_norm:
            # x = F.batch_norm(x, training=self.training)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        # if self.batch_norm:
            # x = F.batch_norm(x, training=self.training)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.shape[0], -1)

        x = self.l1(x)
        x = F.relu(x)
        x = F.dropout(input=x, p=self.dropout_rate)

        x = self.l2(x)
        x = F.relu(x)
        x = F.dropout(input=x, p=self.dropout_rate)

        x = self.l3(x)

        x = torch.sigmoid(x)
        x = torch.squeeze(x)
        return x


# Model to process text
@click.command()
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--grad_clip', default=1.0, help='Gradient clipping')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--batch_norm', default=False, help='Batch norm')
@click.option('--epochs', default=100, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Model path')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
def main(batch_size, lr, dense_dim, grad_clip, 
         dropout_rate, batch_norm, epochs, model_dir, fast_dev_run):
    logger = DvcLiveLogger() if not fast_dev_run else None

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc", 
        mode="max", 
        dirpath=model_dir, 
        filename="{epoch}-{step}-{val_acc:.4f}",
        verbose=True,
        save_top_k=1)
    early_stopping = EarlyStopping(
            monitor='val/acc', 
            patience=10, 
            mode='max', 
            verbose=True)

    model = SimpleImageMaeMaeModel(
        lr=lr, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate,
        batch_norm=batch_norm)

    trainer = Trainer(
        logger=logger,
        max_epochs=epochs,
        gradient_clip_val=grad_clip,
        gpus=1,
        fast_dev_run=fast_dev_run, 
        callbacks=[checkpoint_callback, early_stopping])
    # TODO should I move module inside lightning module?
    trainer.fit(
        model, 
        datamodule=MaeMaeDataModule(
            batch_size=batch_size, 
            train_num_workers=max(1, min(os.cpu_count()//2, 16)),
            val_num_workers=max(1, min(os.cpu_count()//2, 16))
        )
    )

if __name__ == "__main__":
    pl.seed_everything(42)
    main()
