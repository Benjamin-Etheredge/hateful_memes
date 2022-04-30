import click
from icecream import ic

import torch
from torch.nn import functional as F
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models

from transformers import BertTokenizer, VisualBertModel
from transformers import DetrFeatureExtractor, DetrForObjectDetection, AutoConfig

import numpy as np

import pytorch_lightning as pl

from hateful_memes.models.base import BaseMaeMaeModel, base_train

class VisualBertWithODModule(BaseMaeMaeModel):
    """ Visual Bert Model """

    def __init__(
        self,
        lr=0.003,
        max_length=512,
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
        num_queries=50,
        freeze=False,
    ):
        """ Visual Bert Model """
        super().__init__()
        # self.hparams = hparams
        self.visual_bert = VisualBertModel.from_pretrained("uclanlp/visualbert-nlvr2-coco-pre").to(self.device)
        if freeze:
            for param in self.visual_bert.parameters():
                param.requires_grad = False
            self.visual_bert.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Flatten()
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.eval()
        resnet.to(self.device)
        self.resnet = resnet
        
        self.fc_bridge = nn.Linear(2048, 1024)

        ############################################
        # Obj Detection Start
        ############################################
        # self.od_config = AutoConfig.from_pretrained('facebook/detr-resnet-50')
        # self.od_feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
        # self.od_model = DetrForObjectDetection(self.od_config).to(self.device).eval()
        self.num_queries = num_queries

        od_model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
        for param in od_model.parameters():
            param.requires_grad = False
        od_model.eval()
        od_model.to(self.device)
        self.od_model = od_model
        # self.od_poolsize = (self.num_queries//5) + 1

        # Original
        # self.od_fc = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1024),
        #     # nn.Dropout(dropout_rate),
        # )

        # For use w/o pooling
        # self.od_fc = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 32),
        #     nn.ReLU(),
        #     # nn.Dropout(dropout_rate),
        # )

        # self.od_fc2 = nn.Sequential(
        #     nn.Linear(32 * self.num_queries, 1024),
        #     nn.ReLU()
        # )
        # if freeze:
        #     for param in self.od_model.parameters():
        #         param.requires_grad = False
        #     self.od_model.eval()
        ############################################
        # Obj Detection End
        ############################################

        # TODO linear vs embedding for dim changing
        # TODO auto size
        self.fc1 = nn.Linear(768, dense_dim)
        self.fc2 = nn.Linear(dense_dim, dense_dim)
        self.fc3 = nn.Linear(dense_dim, 1)
        # # TODO config modification

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

        ############################################
        # Obj Detection Start
        ############################################
        self.od_model.eval()
        self.resnet.eval()

        with torch.no_grad():
            images_list = [batch_img.half() for batch_img in image]
            od_outputs = self.od_model(images_list)
            batch_pred_boxes = od_outputs['pred_boxes']

            # crop images
            batch_outputs = []
            for idx, batch_img in enumerate(images_list):
                pred_boxes = batch_pred_boxes[idx]
                w, h = batch_img.shape[2], batch_img.shape[1]
                img_outputs = []
                for i in range(self.num_queries):
                    box = pred_boxes[i]
                    center_x, center_y, norm_w, norm_h = box
                    left = int((center_x - norm_w / 2) * w)
                    upper = int((center_y - norm_h / 2) * h)
                    right = int((center_x + norm_w / 2) * w)
                    lower = int((center_y + norm_h / 2) * h)
                    if right - left > 7 and lower - upper > 7: # 7 is the kernel size for renset. need input to be larger                    
                        obj_img = batch_img[:, upper:lower, left:right]
                        obj_img = torch.unsqueeze(obj_img, 0)
                        try:
                            obj_x = self.resnet(obj_img)
                        except:
                            obj_x = torch.zeros(1, 2048).to(self.device)
                    else:
                        obj_x = torch.zeros(1, 2048).to(self.device)
                    img_outputs.append(obj_x)
                img_outputs = torch.stack(img_outputs)
                img_outputs = torch.squeeze(img_outputs)
                batch_outputs.append(img_outputs)
            image_x = torch.stack(batch_outputs)
            
        image_x = self.fc_bridge(image_x)
        image_x = F.relu(image_x)
        ############################################
        # Obj Detection End
        ############################################

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
                "visual_token_type_ids": torch.ones(image_x.shape[:-1], dtype=torch.long, device=self.device),
                "visual_attention_mask": torch.ones(image_x.shape[:-1], dtype=torch.float, device=self.device),
            }
        )

        if self.to_freeze:
            with torch.no_grad():
                x = self.visual_bert(**inputs)
        else:
            x = self.visual_bert(**inputs)

        x = x.pooler_output
        x = x.view(x.shape[0], -1)

        x = torch.squeeze(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate)

        if self.include_top:
            x = self.fc3(x)

        x = torch.squeeze(x, dim=1)
        return x


@click.command()
@click.option('--freeze', default=True, help='Freeze models')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--max_length', default=128, help='Max length')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--num_queries', default=50, help='Number of queries')
# Train kwargs
@click.option('--batch_size', default=0, help='Batch size')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--grad_clip', default=1.0, help='Gradient clip')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--project', default="visual-bert-with-od", help='Project')
def main(freeze, lr, max_length, dense_dim, dropout_rate, num_queries, **train_kwargs):
    """ train model """

    model = VisualBertWithODModule(
        freeze=freeze,
        lr=lr, 
        max_length=max_length, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate,
        num_queries=num_queries)
    base_train(model=model, **train_kwargs)


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
