
import click

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from hateful_memes.models.base import BaseMaeMaeModel, base_train


class SimpleImageMaeMaeModel(BaseMaeMaeModel):
    """ Simple Image Model """
    def __init__(
        self, 
        dense_dim=128, 
        dropout_rate=0.1,
        batch_norm=False,
        include_top=True,
        *base_args, **base_kwargs
    ):
        super().__init__(*base_args, **base_kwargs, plot_name="Simple-image")

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.MaxPool2d(2)
        )

        # TODO better batch norm usage and remove bias

        # self.l1 = nn.Linear(25088, dense_dim)
        self.dense_layers = nn.Sequential(
            nn.Linear(4608, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.fc = nn.Linear(dense_dim, 1)

        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.last_hidden_size = dense_dim
        self.include_top = include_top
        self.last_hidden_size = dense_dim
        self.save_hyperparameters()

    def forward(self, batch):
        x_img = batch['image']
        x = x_img

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.shape[0], -1)
        # x = x.mean(dim=(2, 3))

        x = self.dense_layers(x)

        if self.include_top:
            x = self.fc(x)

        x = torch.squeeze(x)
        return x


# Model to process text
@click.command()
# Model args
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--batch_norm', default=False, help='Batch norm')
# Trainer args
@click.option('--grad_clip', default=1.0, help='Gradient clipping')
@click.option('--batch_size', default=0, help='Batch size')
@click.option('--epochs', default=100, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Model path')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--log_dir', default="data/08_reporting/simple_image", help='Log dir')
@click.option('--project', default="simple-image", help='Project')
def main(lr, dense_dim, dropout_rate, batch_norm,
         **train_kwargs):
    """ Train model """
    model = SimpleImageMaeMaeModel(
        lr=lr, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate,
        batch_norm=batch_norm,
        weight_decay=1e-5,)

    base_train(model=model, **train_kwargs)


if __name__ == "__main__":
    # pl.seed_everything(42)
    main()
