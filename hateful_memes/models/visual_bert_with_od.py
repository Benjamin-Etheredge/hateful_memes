import click
from icecream import ic

import torch
from torch.nn import functional as F
from torch import nn
import torchvision.transforms 
import torchvision.transforms.functional as TF

from transformers import BertTokenizer, VisualBertModel
from transformers import DetrFeatureExtractor, DetrForObjectDetection, AutoConfig

import numpy as np

import pytorch_lightning as pl

from hateful_memes.models.base import BaseMaeMaeModel, base_train

class VisualBertWithODModule(BaseMaeMaeModel):
    """ Visual Bert Model """

    def __init__(
        self,
        max_length=512,
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
        num_queries=50,
        *base_args, **base_kwargs
    ):
        """ Visual Bert Model """
        super().__init__(*base_args, **base_kwargs)

        self.visual_bert = VisualBertModel.from_pretrained("uclanlp/visualbert-nlvr2-coco-pre").to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        ############################################
        # Obj Detection Start
        ############################################
        self.od_config = AutoConfig.from_pretrained('facebook/detr-resnet-50')
        self.od_feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
        self.od_model = DetrForObjectDetection(self.od_config).to(self.device)
        self.num_queries = num_queries
        # self.od_poolsize = (self.num_queries//5) + 1

        # Original
        self.od_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1024),
            nn.Tanh(),
            # nn.Dropout(dropout_rate),
        )
        ############################################
        # Obj Detection End
        ############################################

        # TODO linear vs embedding for dim changing
        # TODO auto size
        self.last_hidden_size = 768
        self.fc = nn.Sequential(
            nn.Linear(self.last_hidden_size, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, 1)
        )
        # TODO config modification

        self.max_length = max_length
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim
        self.visual_bert_config = self.visual_bert.config
        self.backbone = [self.visual_bert, self.od_model]

        self.save_hyperparameters()
    
    def forward(self, batch):
        """ Shut up """
        image = batch['raw_pil_image']
        text = batch['text']

        ############################################
        # Obj Detection Start
        ############################################
        # images_list = [batch_img for batch_img in image.cpu()]

        od_inputs = self.od_feature_extractor(images=image, return_tensors="pt")
        # for k, v in od_inputs.items():
            # od_inputs[k] = v.to(self.device, non_blocking=True)
        od_inputs = od_inputs.to(device=self.device)

        ############################################
        # Obj Detection End
        ############################################

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length)

        # for k, v in inputs.items():
            # inputs[k] = v.to(self.device, non_blocking=True)
        inputs = inputs.to(self.device)

        od_outputs = self.od_model(**od_inputs)
        image_x = od_outputs.last_hidden_state
        image_x = self.od_fc(image_x)
        image_x = image_x.mean(dim=1, keepdim=True)

        inputs.update( {
            "visual_embeds": image_x,
            "visual_token_type_ids": torch.ones(image_x.shape[:-1], dtype=torch.long, device=self.device),
            "visual_attention_mask": torch.ones(image_x.shape[:-1], dtype=torch.float, device=self.device),
        })

        x = self.visual_bert(**inputs)

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
@click.option('--num_queries', default=70, help='Number of queries')
# Train kwargs
@click.option('--batch_size', default=0, help='Batch size')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--grad_clip', default=1.0, help='Gradient clip')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--project', default="visual-bert-with-od", help='Project')
def main(lr, max_length, dense_dim, dropout_rate, num_queries, **train_kwargs):
    """ train model """

    model = VisualBertWithODModule(
        lr=lr, 
        max_length=max_length, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate,
        num_queries=num_queries)
    base_train(model=model, **train_kwargs)


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
