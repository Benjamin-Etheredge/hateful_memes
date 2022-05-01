import torch
from torch import nn
from torch.nn import functional as F


import transformers
import click
from icecream import ic

from hateful_memes.utils import get_project_logger
from hateful_memes.models.base import BaseMaeMaeModel, base_train


class BaseTextMaeMaeModel(BaseMaeMaeModel):
    def __init__(
        self, 
        lr=0.003, 
        dropout_rate=0.1,
        # vocab_size=256, 
        embed_dim=512, 
        dense_dim=128, 
        max_length=128,
        num_layers=2,
        # feature_extractor='bert-base-uncased',
        tokenizer_name='bert-base-uncased',
        include_top=True,
    ):

        super().__init__()
        
        # https://huggingface.co/docs/transformers/v4.18.0/en/model_doc/auto#transformers.AutoTokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        # self.feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(tokenizer_name)

        self.vocab_size = self.tokenizer.vocab_size
        self.embedder = nn.Embedding(self.vocab_size, embed_dim)

        self.lr = lr

        self.lstm = nn.LSTM(
            embed_dim, 
            dense_dim, 
            batch_first=True, 
            # dropout=dropout_rate,
            # proj_size=dense_dim,
            num_layers=num_layers)

        # #
        # self.trans = nn.TransformerEncoderLayer()


        self.l1 = nn.Linear(dense_dim, dense_dim)
        self.l2 = nn.Linear(dense_dim, 1)
        # TODO consider 3 classes for offensive detection

        self.embed_dim = embed_dim
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.include_top = include_top
        self.last_hidden_size = dense_dim

        self.save_hyperparameters()
    
    def forward(self, batch):
        text_features = batch['text']
        input = self.tokenizer(text_features, padding='max_length', truncation=True, max_length=self.max_length)
        # ic(input.keys())
        ids = torch.tensor(input['input_ids']).to(self.device)
        x = self.embedder(ids)
        x = F.dropout(x, self.dropout_rate)
        x, (ht, ct) = self.lstm(x)
        # x = x[:, -1, :]
        x = ht[-1]
        # x = x.view(x.shape[0], -1)
        x = self.l1(x)
        x = F.relu(x)

        if self.include_top:
            x = self.l2(x)
            x = torch.squeeze(x, dim=-1)

        return x


# Model to process text
@click.command()
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--num_layers', default=2, help='Num Layers')
@click.option('--embed_dim', default=64, help='Dense dim')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--batch_size', default=0, help='Batch size')
@click.option('--max_length', default=128, help='Max length')
@click.option('--tokenizer_name', default="bert-base-uncased", help='Tokinizer Name')
@click.option('--grad_clip', default=1.0, help='Gradient clipping')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--epochs', default=100, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Model path')
@click.option('--fast_dev_run', type=bool, default=False, help='Fast dev run')
@click.option('--log_dir', default="data/08_reporting/simple_text", help='Fast dev run')
@click.option('--project', default="simple-text", help='Project name')
def main(lr, num_layers, embed_dim, dense_dim, max_length, tokenizer_name, dropout_rate,
         **train_kwargs):

    """ Train Text model """
    model = BaseTextMaeMaeModel(
        embed_dim=embed_dim,
        tokenizer_name=tokenizer_name,
        lr=lr,
        dense_dim=dense_dim,
        max_length=max_length,
        num_layers=num_layers,
        dropout_rate=dropout_rate)
    
    base_train(model=model, **train_kwargs)


if __name__ == "__main__":
    main()