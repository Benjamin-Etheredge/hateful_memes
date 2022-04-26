import pytorch_lightning as pl
import torch
import click
import torchvision.models as models

from torch.nn import functional as F
from torch import nn

from icecream import ic

from hateful_memes.models import *
from hateful_memes.models.base import BaseMaeMaeModel, base_train
from hateful_memes.utils import get_project_logger
from hateful_memes.utils import get_checkpoint_path


class SuperModel(BaseMaeMaeModel):
    """ Visual Bert Model """

    def __init__(
        self,
        lr=0.003,
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
        num_dense_layers=2,
        freeze=True,
        visual_bert_ckpt=None,
        resnet_ckpt=None,
        simple_image_ckpt=False,
        simple_mlp_image_ckpt=None,
        simple_text_ckpt=None,
        vit_ckpt=None,
        beit_ckpt=None,
        electra_ckpt=None,
        distilbert_ckpt=None,
        visual_bert_with_od_ckpt=None,
    ):
        """ Super Model """
        super().__init__()
        # self.hparams = hparams
        self.models = []
        if visual_bert_ckpt:
            visual_bert_ckpt = get_checkpoint_path(visual_bert_ckpt)
            self.models.append(VisualBertModule.load_from_checkpoint(visual_bert_ckpt))

        if resnet_ckpt:
            # self.models.append(ResNetModule.load_from_checkpoint(resnet_ckpt))
            resnet = models.resnet50(pretrained=True)
            self.num_ftrs_resnet = resnet.fc.in_features
            resnet.fc = nn.Flatten()
            self.resnet = resnet
            self.models.append(resnet)

        if simple_image_ckpt:
            simple_image_ckpt = get_checkpoint_path(simple_image_ckpt)
            self.models.append(SimpleImageMaeMaeModel.load_from_checkpoint(simple_image_ckpt))

        if simple_mlp_image_ckpt:
            simple_mlp_image_ckpt = get_checkpoint_path(simple_mlp_image_ckpt)
            self.models.append(SimpleMLPImageMaeMaeModel.load_from_checkpoint(simple_mlp_image_ckpt))

        if simple_text_ckpt:
            simple_text_ckpt = get_checkpoint_path(simple_text_ckpt)
            self.models.append(BaseTextMaeMaeModel.load_from_checkpoint(simple_text_ckpt))
        
        if vit_ckpt:
            vit_ckpt = get_checkpoint_path(vit_ckpt)
            self.models.append(BaseITModule.load_from_checkpoint(vit_ckpt))
        
        if beit_ckpt:
            beit_ckpt = get_checkpoint_path(beit_ckpt)
            self.models.append(BaseITModule.load_from_checkpoint(beit_ckpt))
        
        if electra_ckpt:
            electra_ckpt = get_checkpoint_path(electra_ckpt)
            self.models.append(AutoTextModule.load_from_checkpoint(electra_ckpt))

        if distilbert_ckpt:
            distilbert_ckpt = get_checkpoint_path(distilbert_ckpt)
            self.models.append(AutoTextModule.load_from_checkpoint(distilbert_ckpt))

        if visual_bert_with_od_ckpt:
            visual_bert_with_od_ckpt = get_checkpoint_path(visual_bert_with_od_ckpt)
            self.models.append(VisualBertWithODModule.load_from_checkpoint(visual_bert_with_od_ckpt))

        assert len(self.models) > 1, "Not enough models loaded"
        
        for model in self.models:
            model.eval()
            model.include_top = False


        if freeze:
            for model in self.models:
                model.freeze()
        
        self.models = nn.ModuleList(self.models)

        self.latent_dim = sum([model.last_hidden_size for model in self.models])
        # TODO get in config
        # TODO accumulate grads over n bacthes

        # TODO linear vs embedding for dim changing
        # TODO auto size
        dense_layers = [
            nn.Linear(self.latent_dim, dense_dim),
        ]
        for _ in range(num_dense_layers):
            dense_layers.append(nn.Linear(dense_dim, dense_dim))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout_rate))
        
        self.dense_model = nn.Sequential(*dense_layers)

        self.final_layer = nn.Linear(dense_dim, 1)
        # TODO config modification

        self.lr = lr
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim
        self.to_freeze = freeze

        self.hparams['latent_dim'] = self.latent_dim
        self.save_hyperparameters()
    
    def forward(self, batch):
        """ Shut up """
        with torch.no_grad():
            x = torch.cat([model(batch) for model in self.models], dim=1)

        x = self.dense_model(x)
        if self.include_top:
            x = self.final_layer(x)

        x = torch.squeeze(x, dim=1)
        return x


@click.command()
@click.option('--freeze', default=True, help='Freeze models')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--num_dense_layers', default=2, help='Dense dim')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--visual_bert_ckpt')
@click.option('--simple_image_ckpt')
@click.option('--simple_mlp_image_ckpt')
@click.option('--simple_text_ckpt')
@click.option('--vit_ckpt')
@click.option('--beit_ckpt')
@click.option('--electra_ckpt')
@click.option('--distilbert_ckpt')
@click.option('--batch_size', default=0, help='Batch size')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--project', default="super-model", help='Project')
def main(freeze, lr, num_dense_layers, dense_dim, dropout_rate,
         visual_bert_ckpt, simple_image_ckpt, simple_mlp_image_ckpt, simple_text_ckpt,
         vit_ckpt, beit_ckpt, electra_ckpt, distilbert_ckpt,
         **train_kwargs):
    """ train model """

    model = SuperModel(
        freeze=freeze,
        lr=lr, 
        num_dense_layers=num_dense_layers,
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate,
        visual_bert_ckpt=visual_bert_ckpt,
        simple_image_ckpt=simple_image_ckpt,
        simple_mlp_image_ckpt=simple_mlp_image_ckpt,
        simple_text_ckpt=simple_text_ckpt,
        vit_ckpt=vit_ckpt,
        beit_ckpt=beit_ckpt,
        electra_ckpt=electra_ckpt,
        distilbert_ckpt=distilbert_ckpt,
        )
    base_train(model=model, **train_kwargs)
    

if __name__ == "__main__":
    pl.seed_everything(42)
    main()
