
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule, Trainer

from models.baseline import BaseMaeMaeModel
from data.hateful_memes import MaeMaeDataModule
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F
import torch


class BaseImageMaeMaeModel(BaseMaeMaeModel):
    def __init__(self, lr=0.003):
        super().__init__()
        self.lr = lr
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)

        self.l1 = nn.Linear(43264, 128)
        # self.l1 = torch.nn.LazyLinear(128)
        # self.l2 = torch.nn.LazyLinear(1)
        self.l2 = nn.Linear(128, 1)
        self.save_hyperparameters()

    def forward(self, batch):
        x_img = batch['image']
        x = x_img
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.shape[0], -1)

        x = self.l1(x)
        x = F.relu(x)

        x = self.l2(x)
        x = torch.sigmoid(x)
        x = torch.squeeze(x)
        return x


# Model to process text

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="Hateful_Memes_Base_Image", log_model=True)
    checkpoint_callback = ModelCheckpoint(monitor="val/acc", mode="max", dirpath="data/06_models/hateful_memes", save_top_k=1)

    trainer = Trainer(
        gpus=1, 
        max_epochs=100, 
        # logger=wandb_logger, 
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback])
    
    from data.hateful_memes import MaeMaeDataset
    model = BaseImageMaeMaeModel()
    trainer.fit(model, datamodule=MaeMaeDataModule(batch_size=32))
