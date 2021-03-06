from matplotlib.pyplot import autoscale
import pytorch_lightning as pl
import torch
import click
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoModel, AutoConfig
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
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
        *base_args, **base_kwargs
    ):

        super().__init__(*base_args, **base_kwargs, plot_name=model_name.upper())
        
        if model_name == 'vit':
            model_fullname = 'google/vit-base-patch16-224'
        elif model_name == 'beit':
            model_fullname = 'microsoft/beit-base-patch16-224-pt22k-ft22k'

        # self.config = AutoConfig.from_pretrained(model_fullname)
        # ic(self.config)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_fullname)
        # self.model = AutoModelForImageClassification.from_pretrained(model_fullname, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(
            model_fullname, 
            output_hidden_states=True, 
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate)
        ic(self.model)

        # TODO pool avg
        self.last_hidden_size = 768
        self.fc = nn.Sequential(
            nn.Linear(self.last_hidden_size, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, 1)
        )

        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim
        self.backbone = self.model

        self.save_hyperparameters()
    
    def forward(self, batch):
        image = batch['raw_pil_image']
        # image = [x_ for x_ in image]
        # TODO look into using model config options for classification

        inputs = self.feature_extractor(images=image, return_tensors="pt", device=self.device)
        inputs = inputs.to(self.device)        

        # TODO pooled output?
        x = self.model(**inputs)

        x = x['pooler_output']

        if self.include_top:
            x = self.fc(x)
            x = torch.squeeze(x, dim=1)
        return x


@click.command()
@click.option('--model_name', default='vit', help='Model name')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--lr', default=1e-4, help='Learning rate')
# Train kwargs
@click.option('--batch_size', default=0, help='Batch size')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--grad_clip', default=1.0, help='Gradient clip')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--log_dir', default="data/08_reporting/vit", help='Log dir')
@click.option('--project', default="vit", help='Project')
def main(model_name, lr, dense_dim, dropout_rate,
         **train_kwargs):
    """ train model """

    model = BaseITModule(
        model_name=model_name,
        lr=lr, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate)
    
    base_train(model=model, **train_kwargs)

if __name__ == "__main__":
    # pl.seed_everything(42)
    main()