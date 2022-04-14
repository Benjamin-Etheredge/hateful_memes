from matplotlib.pyplot import autoscale
import pytorch_lightning as pl
import torch
import click
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torchvision.models as models
from hateful_memes.data.hateful_memes import MaeMaeDataModule
from hateful_memes.utils import get_project_logger
from torch.nn import functional as F
from torch import nn

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from icecream import ic

from torchvision.transforms import ToPILImage  

class ViTModule(pl.LightningModule):
    """ Pretrained ViT """

    def __init__(
        self, 
        lr=0.003, 
        max_length=512, 
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
    ):
        """ Visual Bert Model """
        super().__init__()
        
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', output_hidden_states = True)

        self.fc1 = nn.Linear(151296, dense_dim)
        self.fc2 = nn.Linear(dense_dim, dense_dim)
        self.fc3 = nn.Linear(dense_dim, 1)

        self.lr = lr
        self.max_length = max_length
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim

        self.save_hyperparameters()
    
    def _shared_step(self, batch):
        y_hat = self.forward(batch)
        y = batch['label']
        loss = F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))
        acc = torch.sum(torch.round(torch.sigmoid(y_hat)) == y.data) / (y.shape[0] * 1.0)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].size(0))
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].size(0))
        return loss
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].size(0))
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].size(0))
        return loss


    
    def forward(self, batch):
        image = batch['image']
        # image = image.cpu()
        image = image.uniform_(0, 1)  
        image = [ToPILImage()(x_) for x_ in image]

        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)        

        x = self.model(**inputs)
        x = x.hidden_states[-1]
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(input=x, p=self.dropout_rate)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(input=x, p=self.dropout_rate)

        x = self.fc3(x)
        x.squeeze_()
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@click.command()
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--max_length', default=128, help='Max length')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--gradient_clip_value', default=1.0, help='Gradient clip')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--log_dir', default="data/08_reporting/vit", help='Log dir')
@click.option('--project', default="vit", help='Project')
def main(batch_size, lr, max_length, dense_dim, dropout_rate, 
         epochs, model_dir, gradient_clip_value, fast_dev_run, 
         log_dir, project):
    """ train model """

    logger = get_project_logger(project=project, save_dir=log_dir, offline=fast_dev_run)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc", 
        mode="max", 
        dirpath=model_dir, 
        filename="{epoch}-{step}-{val_acc:.4f}",
        verbose=True,
        save_top_k=1)
    early_stopping = EarlyStopping(
            monitor='val/acc', 
            patience=10, 
            mode='max', 
            verbose=True)

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=epochs, 
        logger=logger,
        gradient_clip_val=gradient_clip_value,
        callbacks=[checkpoint_callback, early_stopping],
        fast_dev_run=fast_dev_run,
    )
    
    model = ViTModule(
        lr=lr, 
        max_length=max_length, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate)
    trainer.fit(
        model, 
        datamodule=MaeMaeDataModule(batch_size=batch_size))

if __name__ == "__main__":
    pl.seed_everything(42)
    main()