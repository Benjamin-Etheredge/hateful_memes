import pytorch_lightning as pl
import torch
import click
import torchvision.models as models

from torch.nn import functional as F
from torch import nn

from icecream import ic
ic.disable()

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
        visual_bert_chkpt=None,
        resnet_chkpt=None,
        simple_image_chkpt=False,
        simple_mlp_image_chkpt=None,
        simple_text_chkpt=None,
        vit_chkpt=None,
        beit_chkpt=None,
    ):
        """ Super Model """
        super().__init__()
        # self.hparams = hparams
        self.models = []
        if visual_bert_chkpt:
            visual_bert_chkpt = get_checkpoint_path(visual_bert_chkpt)
            self.models.append(VisualBertModule.load_from_checkpoint(visual_bert_chkpt))

        if resnet_chkpt:
            # self.models.append(ResNetModule.load_from_checkpoint(resnet_chkpt))
            resnet = models.resnet50(pretrained=True)
            self.num_ftrs_resnet = resnet.fc.in_features
            resnet.fc = nn.Flatten()
            self.resnet = resnet
            self.models.append(resnet)

        if simple_image_chkpt:
            simple_image_chkpt = get_checkpoint_path(simple_image_chkpt)
            self.models.append(SimpleImageMaeMaeModel.load_from_checkpoint(simple_image_chkpt))

        if simple_mlp_image_chkpt:
            simple_mlp_image_chkpt = get_checkpoint_path(simple_mlp_image_chkpt)
            self.models.append(SimpleMLPImageMaeMaeModel.load_from_checkpoint(simple_mlp_image_chkpt))

        if simple_text_chkpt:
            simple_text_chkpt = get_checkpoint_path(simple_text_chkpt)
            self.models.append(BaseTextMaeMaeModel.load_from_checkpoint(simple_text_chkpt))
        
        if vit_chkpt:
            vit_chkpt = get_checkpoint_path(vit_chkpt)
            self.models.append(BaseITModule.load_from_checkpoint(vit_chkpt))
        
        if beit_chkpt:
            beit_chkpt = get_checkpoint_path(beit_chkpt)
            self.models.append(BaseITModule.load_from_checkpoint(beit_chkpt))

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
        ic()
        with torch.no_grad():
            ic()
            for model in self.models:
                ic(model(batch).shape)
            x = torch.cat([model(batch) for model in self.models], dim=1)

        ic(x.shape)
        x = self.dense_model(x)
        ic(x.shape)
        if self.include_top:
            x = self.final_layer(x)

        ic(x.shape)
        x.squeeze_()
        ic(x.shape)
        return x


@click.command()
@click.option('--freeze', default=True, help='Freeze models')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--num_dense_layers', default=2, help='Dense dim')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--visual_bert_chkpt')
@click.option('--simple_image_chkpt')
@click.option('--simple_mlp_image_chkpt')
@click.option('--simple_text_chkpt')
@click.option('--vit_chkpt')
@click.option('--beit_chkpt')
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--project', default="super-model", help='Project')
def main(freeze, lr, num_dense_layers, dense_dim, dropout_rate,
         visual_bert_chkpt, simple_image_chkpt, simple_mlp_image_chkpt, simple_text_chkpt,
         vit_chkpt, beit_chkpt, **train_kwargs):
    """ train model """

    model = SuperModel(
        freeze=freeze,
        lr=lr, 
        num_dense_layers=num_dense_layers,
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate,
        visual_bert_chkpt=visual_bert_chkpt,
        simple_image_chkpt=simple_image_chkpt,
        simple_mlp_image_chkpt=simple_mlp_image_chkpt,
        simple_text_chkpt=simple_text_chkpt,
        vit_chkpt=vit_chkpt,
        beit_chkpt=beit_chkpt,
        )
    base_train(model=model, **train_kwargs)
    

if __name__ == "__main__":
    pl.seed_everything(42)
    main()
