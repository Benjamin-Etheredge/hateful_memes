import click
from icecream import ic

import torch
from torch.nn import functional as F
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models

from transformers import BertTokenizer, VisualBertModel
from transformers import DetrFeatureExtractor, DetrForObjectDetection

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
        # Visual Bert
        pretrained_type = "vqa" # nlvr2 or vqa
        self.visual_bert = VisualBertModel.from_pretrained(f"uclanlp/visualbert-{pretrained_type}-coco-pre")
        for param in self.visual_bert.parameters(recurse=False):
            param.requires_grad = False
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # DETR object detector
        # od_model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
        od_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.od_feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        for param in od_model.parameters(recurse=False):
            param.requires_grad = False
        od_model.eval()

        self.od_model = od_model

        # Resnet
        resnet = models.resnet50(pretrained=True)
        self.num_ftrs_resnet = resnet.fc.in_features
        resnet.fc = nn.Flatten()
        # for param in resnet.parameters():
            # param.requires_grad = False
        # resnet.eval()
        self.resnet = resnet
        for param in self.resnet.parameters(recurse=False):
            param.requires_grad = False

        # FC layer bridging resnet and visualbert
        if pretrained_type == "nlvr2":
            visualbert_input_dim = 1024
        elif pretrained_type == "vqa":
            visualbert_input_dim = 2048
        # self.fc_bridge = nn.Sequential(
        #     nn.Linear(2048, visualbert_input_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout_rate),
        # )

        # FC layers for classification

        self.last_hidden_size = 768
        self.fc = nn.Sequential(
            nn.Linear(768, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, 1)
        )

        self.max_length = max_length
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim
        self.visual_bert_config = self.visual_bert.config
        self.num_queries = num_queries

        self.backbone = [
            # self.od_model,
            self.resnet,
            self.visual_bert
        ]

        self.resizer = T.Resize((224, 224))
        self.normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.save_hyperparameters()

    def detect_objects(self, image):

        image_feats = self.od_feature_extractor(images=image, return_tensors="pt")

        image_feats = image_feats.to(self.device)
        with torch.no_grad():
            od_outputs = self.od_model(**image_feats)
        logits = od_outputs.logits
        probas = logits.softmax(-1)
        batch_keep_idxs = torch.argsort(probas.max(dim=-1)[0], descending=True, dim=-1)[:, :self.num_queries]
        # batch_keep_idxs = torch.argsort(torch.max(probas, dim=-1)[0])[::-1][:, :self.num_queries]
        batch_pred_boxes = od_outputs['pred_boxes']

        batch_keep_boxes = []
        # for idx, img_pred_boxes in batch_keep_idxs, img_pred_boxes:
            # batch_keep_boxes.append(img_keep_boxes)
        batch_keep_boxes = torch.stack([pred_box[idx] for pred_box, idx in zip(batch_pred_boxes, batch_keep_idxs)])

        # crop images
        batch_outputs = []
        batch_inputs = []
        for batch_img, img_pred_boxes in zip(image, batch_keep_boxes):
            batch_img = T.functional.to_tensor(batch_img)
            w, h = batch_img.shape[2], batch_img.shape[1]
            obj_imgs = []
            obj_imgs.append(self.normalizer(self.resizer(batch_img)))
            # obj_imgs.append(self.normalizer(self.resizer(batch_img)).to(self.device, non_blocking=True))

            for i in range(self.num_queries):
                box = img_pred_boxes[i]
                center_x, center_y, norm_w, norm_h = box
                left = int(max((center_x - norm_w / 2), 0) * w)
                upper = int(max((center_y - norm_h / 2), 0) * h)
                right = int(min((center_x + norm_w / 2), 1) * w)
                lower = int(min((center_y + norm_h / 2), 1) * h)

                # yes, i know this is not a good idea, but it allows us to 
                # handle situations where the object is too small (0 pixels in width or height)
                try:
                    obj_img = batch_img[:, upper:lower, left:right]
                    obj_img = self.normalizer(self.resizer(obj_img))
                except:
                    # obj_img = torch.zeros(3, 224, 224, device=self.device)
                    pass
            
            obj_imgs = torch.stack(obj_imgs)
            # obj_img = obj_imgs.to(self.device, non_blocking=True)
            batch_inputs.append(obj_imgs)

        batch_inputs = torch.stack(batch_inputs).to(self.device)
        old_shape = batch_inputs.shape
        # ic(batch_inputs.shape)
        batch_inputs = batch_inputs.reshape(-1, 3, 224, 224)
        # ic(batch_inputs.shape)
        batch_outputs = self.resnet(batch_inputs)
        # ic(batch_outputs.shape)
        batch_outputs = batch_outputs.view(*old_shape[0:2], batch_outputs.shape[-1])

        image_x = batch_outputs
        # image_x = image_x.to(self.device)
        
        # image_x = self.fc_bridge(image_x)

        return image_x

    def forward(self, batch):
        """ Shut up """
        text = batch['text']
        image = batch['raw_pil_image']

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length)

        inputs = {k: v.to(device=self.device, non_blocking=True) for k, v in inputs.items()}
        # inputs = inputs.to(self.device)

        with torch.no_grad():
            image_x = self.detect_objects(image)

        inputs.update(
            {
                "visual_embeds": image_x,
                # "visual_token_type_ids": torch.ones(image_x.shape[:-1], dtype=torch.long, device=self.device),
                "visual_token_type_ids": torch.arange(1, image_x.shape[1]+1).repeat((image_x.shape[0], 1)).to(self.device),
                "visual_attention_mask": torch.ones(image_x.shape[:-1], dtype=torch.float, device=self.device),
            }
        )

        x = self.visual_bert(**inputs)

        x = x.pooler_output
        x = x.view(x.shape[0], -1)

        if self.include_top:
            x = self.fc(x)
            x = torch.squeeze(x, dim=1)

        x = torch.squeeze(x, dim=1) if x.dim() > 1 else x
        return x


@click.command()
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--max_length', default=128, help='Max length')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--num_queries', default=50, help='Number of queries')
@click.option('--weight_decay', default=1e-6, help='Weight decay')
# Train kwargs
@click.option('--batch_size', default=0, help='Batch size')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--grad_clip', default=1.0, help='Gradient clip')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--project', default="visual-bert-with-od", help='Project')
def main(lr, max_length, dense_dim, dropout_rate, num_queries, weight_decay, **train_kwargs):
    """ train model """

    model = VisualBertWithODModule(
        lr=lr, 
        max_length=max_length, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate,
        num_queries=num_queries,
        weight_decay=weight_decay)
    base_train(model=model, finetune_epochs=2, monitor_metric='val/auroc', monitor_metric_mode='max', **train_kwargs)


if __name__ == "__main__":
    # pl.seed_everything(42)
    main()
