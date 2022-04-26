import click

from hateful_memes.models.base import BaseMaeMaeModel, base_train
from torch import nn
from torch.nn import functional as F
import torch
import pytorch_lightning as pl


class SimpleMLPImageMaeMaeModel(BaseMaeMaeModel):
    """Simple MLP model """
    def __init__(
        self, 
        lr=0.003, 
        dense_dim=128, 
        num_dense_layers=2,
        dropout_rate=0.1,
        include_top=True,

    ):
        super().__init__()
        # TODO better batch norm usage and remove bias

        self.l1 = nn.Linear(224*224*3, dense_dim)
        dense_layers = []

        for _ in range(num_dense_layers):
            dense_layers.append(nn.Linear(dense_dim, dense_dim, bias=False))
            dense_layers.append(nn.BatchNorm1d(dense_dim))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(p=dropout_rate))

        self.dense_layers = nn.Sequential(*dense_layers)
        self.final_fc = nn.Linear(dense_dim, 1)

        self.lr = lr
        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate
        self.include_top = include_top
        self.last_hidden_size = dense_dim
        self.save_hyperparameters()

    def forward(self, batch):
        x_img = batch['image']
        x = x_img
        x = x.view(x.shape[0], -1)

        x = self.l1(x)
        x = F.relu(x)
        x = F.dropout(input=x, p=self.dropout_rate)

        x = self.dense_layers(x)

        if self.include_top:
            x = self.final_fc(x)

        x = torch.squeeze(x)
        return x


# Model to process text
@click.command()
# Model Args
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--num_dense_layers', default=2)
#Trainer args
@click.option('--batch_size', default=0, help='Batch size')
@click.option('--grad_clip', default=1.0, help='Gradient clipping')
@click.option('--epochs', default=100, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Model path')
@click.option('--fast_dev_run', type=bool, default=False, help='Fast dev run')
@click.option('--log_dir', default="data/08_reporting/simple_mlp_image", help='Fast dev run')
@click.option('--project', default="simple-mlp-image", help='Fast dev run')
def main(lr, dense_dim, dropout_rate, num_dense_layers,
         **train_kwargs):
    """ shut up pylint """
    model = SimpleMLPImageMaeMaeModel(
        lr=lr, 
        dense_dim=dense_dim, 
        num_dense_layers=num_dense_layers,
        dropout_rate=dropout_rate)

    base_train(model=model, **train_kwargs)


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
