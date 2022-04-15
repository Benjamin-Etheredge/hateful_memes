import pytorch_lightning as pl
import torch
import click
from transformers import BertTokenizer, VisualBertModel
import torchvision.models as models

from torch.nn import functional as F
from torch import nn

from torch.optim.lr_scheduler import ReduceLROnPlateau
from dvclive.lightning import DvcLiveLogger

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from icecream import ic
ic.disable()

from hateful_memes.models import *
from hateful_memes.data.hateful_memes import MaeMaeDataModule
from hateful_memes.models.baseline import BaseMaeMaeModel
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
        freeze=False,
        visual_bert_chkpt=None,
        resnet_chkpt=None,
        simple_image_chkpt=False,
        simple_mlp_image_chkpt=None,
        simple_text_chkpt=None,
    ):
        """ Visual Bert Model """
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
        
        assert len(self.models) > 1, "Not enough models loaded"
        
        for model in self.models:
            model.eval()
            model.include_top = False


        if freeze:
            for model in self.models:
                model.freeze()
        
        self.models = nn.ModuleList(self.models)

        self.latent_dim = sum([model.last_hidden_size for model in self.models])

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
        x = self.final_layer(x)
        ic(x.shape)
        x.squeeze_()
        ic(x.shape)
        return x


@click.command()
@click.option('--freeze', default=True, help='Freeze models')
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--num_dense_layers', default=2, help='Dense dim')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--project', default="super-model", help='Project')
@click.option('--visual_bert_chkpt')
@click.option('--simple_image_chkpt')
@click.option('--simple_mlp_image_chkpt')
@click.option('--simple_text_chkpt')
def main(freeze, batch_size, lr, num_dense_layers,dense_dim, dropout_rate, epochs, 
         model_dir, fast_dev_run, project, 
         visual_bert_chkpt, simple_image_chkpt, simple_mlp_image_chkpt, simple_text_chkpt):
    """ train model """

    logger = get_project_logger(project=project, save_dir=None, offline=fast_dev_run)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss", 
        mode="min", 
        dirpath=model_dir, 
        # filename="{epoch}-{step}-{val/loss:.2f}",
        verbose=True,
        save_top_k=1)

    early_stopping = EarlyStopping(
            monitor='val/loss', 
            patience=10, 
            mode='min', 
            verbose=True)

    trainer = pl.Trainer(
        devices=1, 
        accelerator='auto',
        max_epochs=epochs, 
        logger=logger,
        # logger=wandb_logger, 
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stopping],
        track_grad_norm=2, 
        fast_dev_run=fast_dev_run,
        # detect_anomaly=True, # TODO explore more
        # callbacks=[checkpoint_callback])
        # precision=16,
        # auto_scale_batch_size=True,
    )
    
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
        )
    trainer.fit(
        model, 
        datamodule=MaeMaeDataModule(batch_size=batch_size))


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
    # wandb_logger = WandbLogger(project="Hateful_Memes_Base_Image", log_model=True)
    # checkpoint_callback = ModelCheckpoint(monitor="val/acc", mode="max", dirpath="data/06_models/hateful_memes", save_top_k=1)
