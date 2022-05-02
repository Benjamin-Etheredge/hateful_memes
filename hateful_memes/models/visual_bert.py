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
        max_length=512,
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
        *base_args, **base_kwargs
    ):
        """ Visual Bert Model """
        super().__init__(*base_args, **base_kwargs)
        visual_bert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        # ic(self.visual_bert)
        # ic(self.visual_bert.config)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        resnet = models.resnet50(pretrained=True)
        self.num_ftrs_resnet = resnet.fc.in_features
        resnet.fc = nn.Flatten()
        self.resnet_fc = nn.Linear(self.num_ftrs_resnet, self.num_ftrs_resnet)
        # ic(resnet)

        # TODO linear vs embedding for dim changing
        # TODO auto size
        self.fc = nn.Sequential(
            nn.Linear(768, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # nn.Linear(dense_dim, dense_dim),
            # nn.GELU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, 1)
        )
        # TODO config modification

        self.max_length = max_length
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim
        self.visual_bert_config = visual_bert.config
        self.last_hidden_size = 768

        self.backbone = nn.ModuleList([visual_bert, resnet])

        self.save_hyperparameters()
    
    def forward(self, batch):
        """ Shut up """
        text = batch['text']
        image = batch['image']
        visual_bert, resnet = self.backbone

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length)

        inputs = {k: v.to(device=self.device, non_blocking=True) for k, v in inputs.items()}
        # inputs = inputs.to(self.device)

        image_x = resnet(image)
        image_x = self.resnet_fc(image_x)
        image_x = image_x.view(image_x.shape[0], 1, -1)

        inputs.update({
            "visual_embeds": image_x,
            "visual_token_type_ids": torch.ones(image_x.shape[:-1], dtype=torch.long, device=self.device),
            "visual_attention_mask": torch.ones(image_x.shape[:-1], dtype=torch.float, device=self.device),
        })

        x = visual_bert(**inputs)

        x = x.pooler_output
        x = x.view(x.shape[0], -1)

        if self.include_top:
            x = self.fc(x)
            x = torch.squeeze(x, dim=1)

        return x


@click.command()
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
def main(lr, max_length, dense_dim, dropout_rate, 
         **train_kwargs):
    """ train model """

    model = VisualBertModule(
        lr=lr, 
        max_length=max_length, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate)
    base_train(model=model, **train_kwargs)


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
