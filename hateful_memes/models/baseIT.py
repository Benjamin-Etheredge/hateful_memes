from matplotlib.pyplot import autoscale
from pyrsistent import freeze
import pytorch_lightning as pl
import torch
import click
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torchvision.models as models
from hateful_memes.models.base import BaseMaeMaeModel, base_train
from torch.nn import functional as F
from torch import nn

from icecream import ic

from torchvision.transforms import ToPILImage  

class BaseITModule(BaseMaeMaeModel):
    """ Pretrained ViT """

    def __init__(
        self, 
        model_name='vit',
        lr=0.003, 
        max_length=512, 
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
        freeze=True,
    ):

        super().__init__()
        
        if model_name == 'vit':
            model_fullname = 'google/vit-base-patch16-224'
        elif model_name == 'beit':
            model_fullname = 'microsoft/beit-base-patch16-224-pt22k-ft22k'

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_fullname)
        self.model = AutoModelForImageClassification.from_pretrained(model_fullname, output_hidden_states=True)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=(3, 5), stride=(1, 1), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3))
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=(3, 5), stride=(1, 1), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3))
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # )
        # self.fc1 = nn.Linear(16128, dense_dim)
        # self.fc2 = nn.Linear(dense_dim, dense_dim)
        # self.fc3 = nn.Linear(dense_dim, 1)

        # TODO pool avg
        self.fc1 = nn.Linear(768, dense_dim)
        self.fc2 = nn.Linear(dense_dim, dense_dim)
        self.fc3 = nn.Linear(dense_dim, 1)

        self.lr = lr
        self.max_length = max_length
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim
        self.last_hidden_size = dense_dim
        self.to_freeze = freeze

        self.save_hyperparameters()
    
    def forward(self, batch):
        image = batch['raw_pil_image']
        # image = [x_ for x_ in image]
        # TODO look into using model config options for classification

        inputs = self.feature_extractor(images=image, return_tensors="pt", device=self.device)
        inputs = inputs.to(self.device)        

        # TODO pooled output?
        if self.to_freeze:
            with torch.no_grad():
                x = self.model(**inputs)
        else:
            x = self.model(**inputs)

        x = x.hidden_states[-1]

        # x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)

        # ic(x.shape)
        # x = x.view(x.shape[0], -1)
        # x = x[:, 0, :]
        x = x.mean(dim=1)
        # ic(x.shape)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(input=x, p=self.dropout_rate)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(input=x, p=self.dropout_rate)

        if self.include_top:
            x = self.fc3(x)

        x = torch.squeeze(x, dim=1)
        return x


@click.command()
@click.option('--model_name', default='vit', help='Model name')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--max_length', default=128, help='Max length')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--freeze', default=True, help='Freeze')
# Train kwargs
@click.option('--batch_size', default=0, help='Batch size')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--grad_clip', default=1.0, help='Gradient clip')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--log_dir', default="data/08_reporting/vit", help='Log dir')
@click.option('--project', default="vit", help='Project')
def main(model_name, lr, max_length, dense_dim, dropout_rate, freeze,
         **train_kwargs):
    """ train model """

    model = BaseITModule(
        model_name=model_name,
        lr=lr, 
        max_length=max_length, 
        dense_dim=dense_dim, 
        freeze=freeze,
        dropout_rate=dropout_rate)
    
    base_train(model=model, **train_kwargs)

if __name__ == "__main__":
    pl.seed_everything(42)
    main()