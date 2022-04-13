import pytorch_lightning as pl
import torch
import click
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
import torchvision.models as models
from hateful_memes.data.hateful_memes import MaeMaeDataModule
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from icecream import ic


class ElectraModule(pl.LightningModule):

    def __init__(
        self, 
        lr=0.003, 
        max_length=512, 
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
    ):
        super().__init__()
        self.tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")
        self.ElectraModel = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator")

        self.fc1 = nn.Linear(2, 1)
        # self.fc2 = nn.Linear(dense_dim, 1)

        self.lr = lr
        self.max_length = max_length
        self.include_top = include_top
        # self.dropout_rate = dropout_rate
        # self.dense_dim = dense_dim

    def _shared_step(self, batch):
        y_hat = self.forward(batch)
        y = batch['label']
        loss = F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))
        acc = torch.sum(torch.round(torch.sigmoid(y_hat)) == y.data) / (y.shape[0] * 1.0)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].size(0))
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].size(0))
        return loss
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].size(0))
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].size(0))
        return loss
    
    def forward(self, batch):
        text = batch['text']
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
        )
        inputs = inputs.to(self.device)
        
        x = self.ElectraModel(**inputs)
        x = x.logits
        x = F.softmax(x, dim=1)
        x = torch.index_select(x, 1, torch.tensor([1]).to(self.device))
        # x = self.fc1(x)
        x.squeeze_()

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    


@click.command()
@click.option('--batch_size',          default=32,     help='Batch size')
@click.option('--lr',                  default=1e-4,   help='Learning rate')
@click.option('--max_length',          default=128,    help='Max length')
@click.option('--dense_dim',           default=256,    help='Dense dim')
@click.option('--dropout_rate',        default=0.1,    help='Dropout rate')
@click.option('--epochs',              default=10,     help='Epochs')
@click.option('--model_dir',           default='/tmp', help='Save dir')
@click.option('--gradient_clip_value', default=1.0,    help='Gradient clip')
@click.option('--fast_dev_run',        default=False,  help='Fast dev run')
def main(batch_size, lr, max_length, dense_dim, dropout_rate, 
         epochs, model_dir, gradient_clip_value, fast_dev_run):
    logger = None if fast_dev_run else WandbLogger(project="electra")

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc", 
        mode="max", 
        dirpath=model_dir, 
        filename="{epoch}-{step}-{val_acc:.4f}",
        verbose=True,
        save_top_k=1
    )

    early_stopping = EarlyStopping(
        monitor='val/acc', 
        patience=10, 
        mode='max', 
        verbose=True
    )
    
    trainer = pl.Trainer(
        devices=1,
        accelerator='auto',
        max_epochs=epochs, 
        logger=logger,
        gradient_clip_val=gradient_clip_value,
        callbacks=[checkpoint_callback, early_stopping],
        fast_dev_run=fast_dev_run,
    )
    
    model = ElectraModule(
        lr=lr, 
        max_length=max_length, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate
    )

    trainer.fit(
        model, 
        datamodule=MaeMaeDataModule(batch_size=batch_size, train_num_workers=6, val_num_workers=6, test_num_workers=6)
    )

if __name__ == "__main__":
    pl.seed_everything(42)
    main()
