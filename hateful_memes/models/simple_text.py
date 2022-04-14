import torch
from torch import dropout, nn
from torch.nn import functional as F
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping


import transformers
import click
from icecream import ic
ic.disable()

from hateful_memes.utils import get_project_logger
from hateful_memes.models.baseline import BaseMaeMaeModel
from hateful_memes.data.hateful_memes import MaeMaeDataset
from hateful_memes.data.hateful_memes import MaeMaeDataModule

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
        tokenizer_name='bert-base-uncased'
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
            dropout=dropout_rate,
            # proj_size=dense_dim,
            num_layers=num_layers)
        self.l1 = nn.Linear(dense_dim, dense_dim)
        self.l2 = nn.Linear(dense_dim, 1)
        # TODO consider 3 classes for offensive detection

        self.embed_dim = embed_dim
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.save_hyperparameters()
    
    def forward(self, batch):
        ic()
        text_features = batch['text']
        input = self.tokenizer(text_features, padding='max_length', truncation=True, max_length=self.max_length)
        # ic(input)
        ids = torch.tensor(input['input_ids']).to(self.device)
        ic(ids.shape)
        x = self.embedder(ids)
        x = F.dropout(x, self.dropout_rate)
        ic("post embed: ", x.shape)
        # ic(x.view(x.shape[0], 1, -1).shape)
        x, (ht, ct) = self.lstm(x)
        ic("after lstm:", x.shape)
        # x = x[:, -1, :]
        ic(x[0])
        ic(ht[0])
        x = ht[-1]
        ic("after after lstm:", x.shape)
        # x = x.view(x.shape[0], -1)
        ic(x.shape)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = torch.sigmoid(x)
        x = torch.squeeze(x)
        ic(x.shape)
        return x


# Model to process text
@click.command()
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--num_layers', default=2, help='Num Layers')
@click.option('--embed_dim', default=64, help='Dense dim')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--max_length', default=128, help='Max length')
@click.option('--tokenizer_name', default="bert-base-uncased", help='Tokinizer Name')
@click.option('--grad_clip', default=1.0, help='Gradient clipping')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--epochs', default=100, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Model path')
@click.option('--fast_dev_run', type=bool, default=False, help='Fast dev run')
@click.option('--log_dir', default="data/08_reporting/simple_text", help='Fast dev run')
@click.option('--project', default="simple-text", help='Project name')
@click.option('--monitor_metric', default="val/loss", help='Metric to monitor')
@click.option('--monitor_metric_mode', default="min", help='Min or max')
def main(batch_size, lr, num_layers, embed_dim, dense_dim, max_length, tokenizer_name,
         grad_clip, dropout_rate, epochs, model_dir, fast_dev_run,
         log_dir, project, monitor_metric, monitor_metric_mode):

    """ Train Text model """
    logger = get_project_logger(project=project, save_dir=log_dir, offline=fast_dev_run)

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric, 
        mode=monitor_metric_mode, 
        dirpath=model_dir, #"data/06_models/hateful_memes", 
        save_top_k=1)

    early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=10, 
            mode=monitor_metric_mode,
            verbose=True)

    trainer = Trainer(
        devices=1, 
        accelerator='auto',
        max_epochs=epochs, 
        logger=logger, 
        fast_dev_run=fast_dev_run,
        gradient_clip_val=grad_clip,
        callbacks=[checkpoint_callback, early_stopping])
    
    model = BaseTextMaeMaeModel(
        embed_dim=embed_dim,
        tokenizer_name=tokenizer_name,
        lr=lr,
        dense_dim=dense_dim,
        max_length=max_length,
        num_layers=num_layers,
        dropout_rate=dropout_rate)
    trainer.fit(
        model, 
        datamodule=MaeMaeDataModule(batch_size=batch_size))


if __name__ == "__main__":
    main()