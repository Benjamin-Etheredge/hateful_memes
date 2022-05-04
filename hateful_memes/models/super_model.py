import pytorch_lightning as pl
import torch
import click
import torchvision.models as models

from torch.nn import functional as F
from torch import nn

from icecream import ic

from hateful_memes.models.baseIT import *
from hateful_memes.models.visual_bert import *
from hateful_memes.models.visual_bert_with_od import *
from hateful_memes.models.simple_image import *
from hateful_memes.models.simple_text import *
from hateful_memes.models.auto_text_model import *
from hateful_memes.models.simple_mlp_image import *

from hateful_memes.models.base import BaseMaeMaeModel, base_train
from hateful_memes.utils import get_checkpoint_path
import sys


class SuperModel(BaseMaeMaeModel):
    """ Visual Bert Model """

    def __init__(
        self,
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
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
        *base_args, **base_kwargs
    ):
        """ Super Model """
        super().__init__(*base_args, **base_kwargs, plot_name="Super-model")
        ic.disable()
        self.models = []
        if resnet_ckpt:
            # self.models.append(ResNetModule.load_from_checkpoint(resnet_ckpt))
            resnet = models.resnet50(pretrained=True)
            self.num_ftrs_resnet = resnet.fc.in_features
            resnet.fc = nn.Flatten()
            self.resnet = resnet
            self.models.append(resnet)

        if visual_bert_ckpt and visual_bert_ckpt != "None":
            visual_bert_ckpt = get_checkpoint_path(visual_bert_ckpt)
            self.models.append(VisualBertModule.load_from_checkpoint(visual_bert_ckpt))

        if simple_image_ckpt and simple_image_ckpt != "None":
            simple_image_ckpt = get_checkpoint_path(simple_image_ckpt)
            self.models.append(SimpleImageMaeMaeModel.load_from_checkpoint(simple_image_ckpt))

        if simple_mlp_image_ckpt and simple_mlp_image_ckpt != "None":
            simple_mlp_image_ckpt = get_checkpoint_path(simple_mlp_image_ckpt)
            self.models.append(SimpleMLPImageMaeMaeModel.load_from_checkpoint(simple_mlp_image_ckpt))

        if simple_text_ckpt and simple_text_ckpt != "None":
            simple_text_ckpt = get_checkpoint_path(simple_text_ckpt)
            self.models.append(BaseTextMaeMaeModel.load_from_checkpoint(simple_text_ckpt))
        
        if vit_ckpt and vit_ckpt != "None":
            vit_ckpt = get_checkpoint_path(vit_ckpt)
            self.models.append(BaseITModule.load_from_checkpoint(vit_ckpt))
        
        if beit_ckpt and beit_ckpt != "None":
            beit_ckpt = get_checkpoint_path(beit_ckpt)
            self.models.append(BaseITModule.load_from_checkpoint(beit_ckpt))
        
        if electra_ckpt and electra_ckpt != "None":
            electra_ckpt = get_checkpoint_path(electra_ckpt)
            self.models.append(AutoTextModule.load_from_checkpoint(electra_ckpt))

        if distilbert_ckpt and distilbert_ckpt != "None":
            distilbert_ckpt = get_checkpoint_path(distilbert_ckpt)
            self.models.append(AutoTextModule.load_from_checkpoint(distilbert_ckpt))

        if visual_bert_with_od_ckpt and visual_bert_with_od_ckpt != "None":
            visual_bert_with_od_ckpt = get_checkpoint_path(visual_bert_with_od_ckpt)
            self.models.append(VisualBertWithODModule.load_from_checkpoint(visual_bert_with_od_ckpt))

        ic.enable()
        assert len(self.models) > 1, "Not enough models loaded"
        
        for model in self.models:
            # model.eval()
            model.include_top = False


        # if freeze:
        #     for model in self.models:
        #         model.freeze()
        
        self.models = nn.ModuleList(self.models)

        self.latent_dim = sum([model.last_hidden_size for model in self.models])
        # TODO get in config
        # TODO accumulate grads over n bacthes

        # TODO linear vs embedding for dim changing
        # TODO auto size
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_dim, 1),
        )

        # TODO config modification

        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim
        self.last_hidden_size = dense_dim

        #self.hparams['latent_dim'] = self.latent_dim  # Breaks checkpoints
        self.backbone = self.models
        self.save_hyperparameters()
    
    def forward(self, batch):
        """ Shut up """
        with torch.no_grad():
            mod_out = []
            for model in self.models:
                out_i = model(batch)
                if out_i.dim() == 1:
                    out_i = torch.unsqueeze(out_i, 0)  # For single-sample batches in interpretability runs
                mod_out.append(out_i)
            x = torch.cat(mod_out, dim=1) 
            #x = torch.cat([model(batch) for model in self.models], dim=1)

        if self.include_top:
            x = self.fc(x)

        x = torch.squeeze(x, dim=1) if x.dim() > 1 else x
        return x


@click.command()
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--visual_bert_ckpt')
@click.option('--visual_bert_with_od_ckpt')
@click.option('--simple_image_ckpt')
@click.option('--simple_mlp_image_ckpt')
@click.option('--simple_text_ckpt')
@click.option('--vit_ckpt')
@click.option('--beit_ckpt')
@click.option('--electra_ckpt')
@click.option('--distilbert_ckpt')
@click.option('--grad_clip', default=0.0, help='Grad clip')
@click.option('--batch_size', default=0, help='Batch size')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--project', default="super-model", help='Project')
def main(lr, dense_dim, dropout_rate,
         visual_bert_ckpt, visual_bert_with_od_ckpt, 
         simple_image_ckpt, simple_mlp_image_ckpt, simple_text_ckpt,
         vit_ckpt, beit_ckpt, electra_ckpt, distilbert_ckpt,
         **train_kwargs):
    """ train model """

    model = SuperModel(
        lr=lr, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate,
        visual_bert_ckpt=visual_bert_ckpt,
        visual_bert_with_od_ckpt=visual_bert_with_od_ckpt,
        simple_image_ckpt=simple_image_ckpt,
        simple_mlp_image_ckpt=simple_mlp_image_ckpt,
        simple_text_ckpt=simple_text_ckpt,
        vit_ckpt=vit_ckpt,
        beit_ckpt=beit_ckpt,
        electra_ckpt=electra_ckpt,
        distilbert_ckpt=distilbert_ckpt,
        weight_decay=0.01)
    base_train(model=model, finetune_epochs=100, **train_kwargs)
    

if __name__ == "__main__":
    # pl.seed_everything(42)
    main()
