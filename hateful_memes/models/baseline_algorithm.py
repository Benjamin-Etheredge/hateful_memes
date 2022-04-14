import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from hateful_memes.utils import get_project_logger
from hateful_memes.data.hateful_memes import MaeMaeDataModule
from dvclive.lightning import DvcLiveLogger
import pytorch_lightning as pl
from torch.nn import functional as F


class Base(LightningModule):
    """ Simple base model """

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
    
    def _shared_step(self, batch):
        x = batch['image']
        y_hat = self.forward(x)
        y = batch['label']
        loss = F.binary_cross_entropy(y_hat, y.to(y_hat.dtype))
        acc = torch.sum(y_hat == y.data) / (y.shape[0] * 1.0)
        return loss, acc

    def configure_optimizers(self):
        pass
    

class Affirmative(Base):
    """ Return 1 for all inputs """
    def forward(self, x):
        batch_size = x.size(0)
        return torch.ones(batch_size)
    

class Negative(Base):
    def forward(self, x):
        batch_size = x.size(0)
        return torch.zeros(batch_size)
       

from pytorch_lightning.loggers import WandbLogger
if __name__ == '__main__':
    pl.seed_everything(42)
    logger = get_project_logger(project='baseline_algorithm', save_dir='data/08_reporting/baseline', offline=True)

    trainer = pl.Trainer(
        devices=1, 
        accelerator='auto',
        logger=logger,
    )
    
    trainer.validate(Negative(), datamodule=MaeMaeDataModule(batch_size=8))

