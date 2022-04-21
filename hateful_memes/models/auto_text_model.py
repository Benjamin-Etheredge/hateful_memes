from pyexpat import model
import pytorch_lightning as pl
import torch
import click
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from icecream import ic

from hateful_memes.models.baseline import BaseMaeMaeModel
from hateful_memes.data.hateful_memes import MaeMaeDataModule

class AutoTextModule(BaseMaeMaeModel):

    def __init__(
        self, 
        lr=0.003, 
        max_length=512, 
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
        model_name='google/electra-small-discriminator',
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        self.lr = lr
        self.max_length = max_length
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim

        self.fc1 = nn.Linear(self.config.hidden_size * self.max_length, dense_dim)
        self.fc2 = nn.Linear(dense_dim, 1)

    def _shared_step(self, batch):
        y_hat = self.forward(batch)
        y = batch['label']
        loss = F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))
        acc = torch.sum(torch.round(torch.sigmoid(y_hat)) == y.data) / (y.shape[0] * 1.0)
        return loss, acc

    
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
        
        x = self.model(**inputs)
        x = x.last_hidden_state
        x = x.view(x.shape[0], -1)

        if self.include_top:
            x = self.fc1(x)
            x = F.relu(x)
            x = F.dropout(input=x, p=self.dropout_rate)

            x = self.fc2(x)

            x.squeeze_()

        return x
    
@click.command()
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--max_length', default=128, help='Max length')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--gradient_clip_value', default=1.0, help='Gradient clip')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--model_name', default='google/electra-small-discriminator', help='Model name')
@click.option('--model_name_simple', default='Electra', help='Simple model name for wandb')
def main(batch_size, lr, max_length, dense_dim, dropout_rate, 
         epochs, model_dir, gradient_clip_value, fast_dev_run, model_name,
         model_name_simple):
    logger = None if fast_dev_run else WandbLogger(project=model_name_simple)

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
    
    model = AutoTextModule(
        lr=lr, 
        max_length=max_length, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate,
        model_name=model_name,
    )

    trainer.fit(
        model, 
        datamodule=MaeMaeDataModule(batch_size=batch_size)
    )

if __name__ == "__main__":
    pl.seed_everything(42)
    main()
