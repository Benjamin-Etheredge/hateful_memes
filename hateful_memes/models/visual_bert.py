import click
from icecream import ic

import torch
from torch.nn import functional as F
from torch import nn
import torchvision.models as models

from transformers import BertTokenizer, VisualBertModel

import pytorch_lightning as pl

from hateful_memes.models.base import BaseMaeMaeModel, base_train


class VisualBertModule(BaseMaeMaeModel):
    """ Visual Bert Model """

    def __init__(
        self,
        lr=0.003,
        max_length=512,
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
        freeze=False,
    ):
        """ Visual Bert Model """
        super().__init__()
        # self.hparams = hparams
        self.visual_bert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        if freeze:
            for param in self.visual_bert.parameters():
                param.requires_grad = False
            self.visual_bert.eval()
        ic(self.visual_bert)
        ic(self.visual_bert.config)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        resnet = models.resnet50(pretrained=True)
        self.num_ftrs_resnet = resnet.fc.in_features
        resnet.fc = nn.Flatten()
        ic(resnet)
        self.resnet = resnet

        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False
            resnet.eval()

        # TODO linear vs embedding for dim changing
        # TODO auto size
        self.fc1 = nn.Linear(768, dense_dim)
        self.fc2 = nn.Linear(dense_dim, 1)
        # TODO config modification

        self.lr = lr
        self.max_length = max_length
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim
        self.to_freeze = freeze
        self.visual_bert_config = self.visual_bert.config
        self.last_hidden_size = dense_dim

        self.save_hyperparameters()
    
    def forward(self, batch):
        """ Shut up """
        text = batch['text']
        image = batch['image']
        image_x = self.resnet(image)
        if self.to_freeze:
            with torch.no_grad():
                image_x = self.resnet(image)
        else:
            image_x = self.resnet(image)

        image_x = image_x.view(image_x.shape[0], -1)

        image_x = image_x.unsqueeze(1)

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length)
        inputs = inputs.to(self.device)

        inputs.update(
            {
                "visual_embeds": image_x,
                "visual_token_type_ids": torch.ones(image_x.shape[:-1], dtype=torch.long).to(self.device),
                "visual_attention_mask": torch.ones(image_x.shape[:-1], dtype=torch.float).to(self.device),
            }
        )

        if self.to_freeze:
            with torch.no_grad():
                x = self.visual_bert(**inputs)
        else:
            x = self.visual_bert(**inputs)

        x = x.pooler_output
        x = x.view(x.shape[0], -1)

        x.squeeze_()
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate)

        if self.include_top:
            x = self.fc2(x)

        x.squeeze_()
        return x


@click.command()
@click.option('--freeze', default=True, help='Freeze models')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--max_length', default=128, help='Max length')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
# Train kwargs
@click.option('--batch_size', default=0, help='Batch size')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--grad_clip', default=1.0, help='Gradient clip')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--log_dir', default="data/08_reporting/visual_bert", help='Log dir')
@click.option('--project', default="visual-bert", help='Project')
def main(freeze, lr, max_length, dense_dim, dropout_rate, 
         **train_kwargs):
    """ train model """

    model = VisualBertModule(
        freeze=freeze,
        lr=lr, 
        max_length=max_length, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate)
    base_train(model=model, **train_kwargs)


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
