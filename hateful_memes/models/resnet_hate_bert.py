from matplotlib.pyplot import text
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

import transformers
import click
from icecream import ic

from hateful_memes.utils import get_project_logger
from hateful_memes.models.base import BaseMaeMaeModel, base_train

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import pytorch_lightning as pl


#### TODO will leak as was trained on dev and some of test set
class ResnetHateBert(BaseMaeMaeModel):
    def __init__(
        self, 
        dropout_rate=0.1,
        dense_dim=128, 
        max_length=96,
        include_top=True,
        *base_args, **base_kwargs
    ):
        super().__init__(*base_args, **base_kwargs)
        
        # https://huggingface.co/docs/transformers/v4.18.0/en/model_doc/auto#transformers.AutoTokenizer
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        # self.feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(tokenizer_name)
        # self.bert = AutoModelForSequenceClassification.from_pretrained(
            # "am4nsolanki/autonlp-text-hateful-memes-36789092", 
            # output_hidden_states=True,
            # return_dict=True,)
        self.bert = AutoModel.from_pretrained(
            "am4nsolanki/autonlp-text-hateful-memes-36789092", 
            output_hidden_states=True,
            return_dict=True,)
        for param in self.bert.parameters(recurse=True):
            param.requires_grad = False
        # ic(self.bert)
        self.tokenizer = AutoTokenizer.from_pretrained("am4nsolanki/autonlp-text-hateful-memes-36789092")
        # ic(self.bert.config)

        self.vocab_size = self.tokenizer.vocab_size

        self.resnet = torchvision.models.resnet50(pretrained=True)
        # ic(self.resnet)
        self.num_ftrs_resnet = self.resnet.fc.in_features
        self.resnet.fc = nn.Flatten()
        for param in self.resnet.parameters(recurse=True):
            param.requires_grad = False
        # self.resnet_fc = nn.Linear(self.num_ftrs_resnet, self.num_ftrs_resnet)
        # self.bert_fc = nn.Sequential(
            # nn.Linear(768, 768),
        # )
        # ic(self.resnet)

        resenet_size = 2048
        bert_size = 768
        self.last_hidden_size = resenet_size + bert_size
        self.final_fc = nn.Sequential(
            nn.Linear(self.last_hidden_size, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, 1),
        )

        # TODO consider 3 classes for offensive detection

        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.include_top = include_top
        self.backbone = self.resnet

        self.save_hyperparameters()
    
    def forward(self, batch):
        # Text
        text_features = batch['text']
        input = self.tokenizer(
            text_features, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,
            return_tensors='pt',)

        input = {k: v.to(device=self.device, non_blocking=True) for k, v in input.items()}

        # Image
        images = batch['image']
        image_x = self.resnet(images)
        # image_x = self.resnet_fc(image_x)

        text_x = self.bert(**input)

        # text_x = text_x.last_hidden_state[:, 0]
        text_x = text_x[0]
        text_x = text_x[:, 0]
        # text_x = self.bert_fc(text_x)

        x = torch.cat((text_x, image_x), dim=1)

        if self.include_top:
            x = self.final_fc(x)
            x = torch.squeeze(x, dim=-1)

        return x


# Model to process text
@click.command()
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--batch_size', default=4, help='Batch size')
@click.option('--dense_dim', default=128, help='Dense layer dimension')
@click.option('--max_length', default=96, help='Max length')
@click.option('--grad_clip', default=1.0, help='Gradient clipping')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--epochs', default=1000, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Model path')
@click.option('--fast_dev_run', type=bool, default=False, help='Fast dev run')
@click.option('--log_dir', default="data/08_reporting/simple_image_text", help='Fast dev run')
@click.option('--project', default="resnet-bert", help='Project name')
def main(lr, dense_dim, max_length, dropout_rate,
         **train_kwargs):

    """ Train Text model """
    model = ResnetHateBert(
        lr=lr,
        dense_dim=dense_dim,
        max_length=max_length,
        dropout_rate=dropout_rate)
    
    base_train(model=model, finetune_epochs=5, **train_kwargs)


if __name__ == "__main__":
    # pl.seed_everything(42)
    main()