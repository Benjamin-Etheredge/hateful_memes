from pyexpat import model
import pytorch_lightning as pl
import torch
import click
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import ElectraTokenizer, ElectraModel, ElectraConfig
from torch.nn import functional as F
from torch import nn
from icecream import ic

from hateful_memes.models.base import BaseMaeMaeModel, base_train

class AutoTextModule(BaseMaeMaeModel):

    def __init__(
        self, 
        model_name,
        max_length=512, 
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
        *base_args, **base_kwargs
    ):
        super().__init__(*base_args, **base_kwargs, plot_name="Electra" if 'electra' in model_name else "DistilBERT")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        ic(self.config)
        self.model = AutoModel.from_pretrained(
            model_name, 
            config=self.config,
            # NOTE already has dropout
            )
        ic(self.model)

        self.max_length = max_length
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim

        self.last_hidden_size = self.config.hidden_size
        self.fc = nn.Sequential(
            nn.Linear(self.last_hidden_size, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, 1)
        )
        self.backbone = self.model

        self.save_hyperparameters()
    
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
        
        #https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/distilbert/modeling_distilbert.py#L691
        # Made adjustments based on the above link
        x = x[:, 0]

        if self.include_top:
            x = self.fc(x)
            x = x.squeeze(1)

        return x
    
@click.command()
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--max_length', default=128, help='Max length')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--model_name', default='google/electra-small-discriminator', help='Model name')
# Train kwargs
@click.option('--batch_size', default=0, help='Batch size')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--grad_clip', default=1.0, help='Gradient clip')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--project', default='', help='Simple model name for wandb')
def main(lr, max_length, dense_dim, dropout_rate, model_name,
         **train_kwargs):
    
    model = AutoTextModule(
        lr=lr, 
        max_length=max_length, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate,
        model_name=model_name,
    )

    base_train(model=model, **train_kwargs)


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
