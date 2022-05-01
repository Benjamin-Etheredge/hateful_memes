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
        self.visual_bert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre").to(self.device)
        if freeze:
            for param in self.visual_bert.parameters():
                param.requires_grad = False
            self.visual_bert.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Flatten()
        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False
            resnet.eval()
        resnet.to(self.device)
        self.resnet = resnet
        
        self.num_queries = num_queries

        od_model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
        if freeze:
            for param in od_model.parameters():
                param.requires_grad = False
            od_model.eval()
        od_model.to(self.device)
        self.od_model = od_model

        self.pad = nn.ZeroPad2d(3)

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
            logits = od_outputs['pred_logits']
            probas = logits.softmax(-1)
            batch_keep_idxs = np.argsort(probas.max(-1).values.cpu().numpy())[::-1][:, :self.num_queries]
            batch_pred_boxes = od_outputs['pred_boxes']

            batch_keep_boxes = []
            for i in range(batch_keep_idxs.shape[0]):
                img_pred_boxes = batch_pred_boxes[i]
                img_keep_idxs = batch_keep_idxs[i]
                img_keep_boxes = img_pred_boxes[img_keep_idxs]
                batch_keep_boxes.append(img_keep_boxes)
            batch_keep_boxes = torch.stack(batch_keep_boxes)

            # crop images
            batch_outputs = []
            for idx, batch_img in enumerate(images_list):
                w, h = batch_img.shape[2], batch_img.shape[1]
                pred_boxes = batch_keep_boxes[idx]
                obj_imgs = []
                for i in range(self.num_queries):
                    box = pred_boxes[i]
                    center_x, center_y, norm_w, norm_h = box
                    left = int(max((center_x - norm_w / 2), 0) * w)
                    upper = int(max((center_y - norm_h / 2), 0) * h)
                    right = int(min((center_x + norm_w / 2), 1) * w)
                    lower = int(min((center_y + norm_h / 2), 1) * h)
                    try:
                        obj_img = batch_img[:, upper:lower, left:right]
                        obj_img = T.Resize((180, 180))(obj_img)
                    except:
                        obj_img = torch.zeros(3, 180, 180).to(self.device)
                    obj_imgs.append(obj_img)

                obj_imgs = torch.stack(obj_imgs)
                img_outputs = self.resnet(obj_imgs)
                img_outputs = torch.squeeze(img_outputs)
                batch_outputs.append(img_outputs)
            image_x = torch.stack(batch_outputs)

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
