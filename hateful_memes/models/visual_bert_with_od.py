import click
from icecream import ic

import torch
from torch.nn import functional as F
from torch import nn
import torchvision.models as models
import torchvision.transforms as T

from transformers import BertTokenizer, VisualBertModel
from transformers import DetrFeatureExtractor, DetrForObjectDetection, AutoConfig

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
        self.visual_bert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        if freeze:
            for param in self.visual_bert.parameters():
                param.requires_grad = False
            self.visual_bert.eval()
        # ic(self.visual_bert)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        ############################################
        # Obj Detection Start
        ############################################
        self.od_config = AutoConfig.from_pretrained('facebook/detr-resnet-50', num_queries=num_queries)
        self.od_feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
        self.od_model = DetrForObjectDetection(self.od_config)

        for param in self.od_model.parameters():
            param.requires_grad = False
        self.od_model.eval()

        # ic(self.od_model)
        self.od_fc = nn.Linear(self.od_config.num_queries * 256, 2048)
        # self.od_fc = nn.Linear(256, 768)

        ############################################
        # Obj Detection End
        ############################################
        if freeze:
            for param in self.od_model.parameters():
                param.requires_grad = False
            self.od_model.eval()

        # TODO linear vs embedding for dim changing
        # TODO auto size
        self.fc1 = nn.Linear(768, dense_dim)
        self.fc2 = nn.Linear(dense_dim, dense_dim)
        self.fc3 = nn.Linear(dense_dim, 1)
        # TODO config modification

        self.lr = lr
        self.max_length = max_length
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim
        self.to_freeze = freeze
        self.visual_bert_config = self.visual_bert.config
        self.last_hidden_size = dense_dim
        self.num_queries = num_queries

        self.save_hyperparameters()
    
    def forward(self, batch):
        """ Shut up """
        text = batch['text']
        image = batch['image']

        ############################################
        # Obj Detection Start
        ############################################
        images_list = []
        for i in range(image.shape[0]):
            batch_image=image[i, :]

            transform = T.ToPILImage()
            pil_image = transform(batch_image)
            images_list.append(pil_image)

        od_inputs = self.od_feature_extractor(images=images_list, return_tensors="pt")
        od_inputs = od_inputs.to(self.device)
        od_outputs = self.od_model(**od_inputs)
        # ic(od_outputs.keys())
        # for key in od_outputs.keys():
        #     ic(key, od_outputs[key].shape)

        # ic(od_outputs.last_hidden_state.shape)
        image_x = od_outputs.last_hidden_state

        image_x = image_x.view(image_x.shape[0], 1, -1)
        image_x = self.od_fc(image_x)
        image_x = torch.tanh(image_x)
        image_x = F.dropout(image_x, p=self.dropout_rate)
        # ic(image_x.shape)
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
def main(freeze, lr, max_length, dense_dim, dropout_rate, num_queries,
         **train_kwargs):
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
