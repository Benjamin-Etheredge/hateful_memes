
import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision import transforms as T
from icecream import ic
from pytorch_lightning.loggers import WandbLogger



class BaseMaeMaeModel(LightningModule):

    def forward(self, batch):
        raise NotImplemented

    def _shared_step(self, batch):

        y = batch['label']
        y_hat = self(batch)
        loss = F.binary_cross_entropy(y_hat, y.to(y_hat.dtype))

        acc = torch.sum(torch.round(y_hat) == y.data) / (y.shape[0] * 1.0)

        return loss, acc

    def training_step(self, batch, batch_idx):
        # x_img, x_txt = self.preprocess(x_img, x_txt)
        loss, acc = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_nb):
        loss, acc = self._shared_step(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


import transformers

class BaseTextMaeMaeModel(BaseMaeMaeModel):
    def __init__(
        self, 
        lr=0.003, 
        vocab_size=256, 
        embed_dim=512, 
        dense_dim=128, 
        max_length=128,
        batch_size=32):
        super().__init__()

        
        # self.pipeline = transformers.pipeline("feature-extraction", framework='pt')
        # transformers.Pretrainged
        # self.pipeline = transformers.pipeline('feature-extraction', model='bert-base-cased', tokenizer='bert-base-cased')
        # self.pipeline = transformers.pipeline('text-classification', 'Hate-speech-CNERG/bert-base-uncased-hatexplain')
        # self.pipeline = transformers.pipeline('text-classification', 'Hate-speech-CNERG/bert-base-uncased-hatexplain')
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
        # self.modelm = transformers.AutoModel.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
        # TODO could fine tune
        # self.tokenizer

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.lr = lr

        self.l1 = nn.Linear(embed_dim, dense_dim)
        self.l2 = nn.Linear(dense_dim, 1)
        # TODO consider 3 classes for offensive detection

        # self.log("batch_size", batch_size)
        self.batch_size = batch_size

        self.save_hyperparameters()
    
    # def preprocess(self, x_img, x_txt):
    #     x_txt = self.tokenizer(x_txt, return_tensors='pt', padding='max_length', truncation=True)
    #     return x_img, x_txt
    #     # return super().preprocess(x_img, x_txt)
    

    def forward(self, batch):
        text_features = batch['text_features']
        text_offset = batch['text_offset']
        # x = x_txt
        # ic(text_features.shape)
        # ic(text_offset.shape)
        x = self.embedding(text_features, text_offset)
        # ic(x.shape)
        # ic(len(x))
        # ic(x)
        # x = self.pipeline(x, max_length=512, padding='max_length', truncation=True, return_tensors='pt')

        # for key, value in x.items():
        #     x[key] = value.to(self.device)
        # x = self.modelm(**x)
        # ic(x['pooler_output'].shape)
        # ic(x['last_hidden_state'].shape)
        # x = x['pooler_output']
        # ic(x.shape)
        x = self.l1(x)
        # ic(x.shape)
        x = F.relu(x)
        x = self.l2(x)
        # ic(x.shape)
        x = torch.sigmoid(x)
        # x = self.pipeline(x)
        # ic(len(x))
        # ic(len(x[0]))
        # ic(len(x[0]))
        # ic(len(x[3]))
        # ic(x)
        # x = torch.Tensor(x)
        # ic(x.shape)
        # ic(x.shape)
        # x = self.l1(x)
        # ic(x.shape)
        # return [i['score'] for i in x]
        x = torch.squeeze(x)
        return x

from data.hateful_memes import MaeMaeDataModule
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
if __name__ == "__main__":
    wandb_logger = WandbLogger(project="Hateful_Memes_Base", log_model=True)
    # wandb_logger = WandbLogger(log_model=True)
    # cli = LightningCLI(
    #     model_class=MaeMaeModel, 
    #     datamodule_class=MaeMaeDataModule, 
    #     # logger_class=WandbLogger, 
    #     seed_everything_default=4,
    #     # logger=wandb_logger,
    #     trainer_defaults={'logger': wandb_logger},
    #     )
    # data = MaeMaeDataModule(
    #     batch_size=32)

    # train_ds = MeemeesDataset(
    #     root_dir="data/01_raw/hateful_memes",
    #     transforms=create_transformer(img_transforms=img_transforms)
    #     )
    # train_loader = DataLoader(
    #     train_ds, 
    #     batch_size=64,
    #     pin_memory=False, 
    #     num_workers=8)
    checkpoint_callback = ModelCheckpoint(monitor="val/acc", mode="max", dirpath="data/06_models/hateful_memes", save_top_k=1)
    trainer = Trainer(
        gpus=1, 
        max_epochs=100, 
        # logger=wandb_logger, 
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback])
    
    from data.hateful_memes import MaeMaeDataset
    dataset = MaeMaeDataset(
        "data/01_raw/hateful_memes",
        train=True,
    )
    train_dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        # collate_fn=MaeMaeDataset.collate_batch,

    )
    model = BaseTextMaeMaeModel(
         vocab_size=len(dataset.vocab),
         batch_size=32
    )
    # trainer.fit(model, datamodule=MaeMaeDataModule(batch_size=32)) #, ckpt_path='data/06_models/base_text_maemae_model.ckpt')
    # trainer.fit(model=model, train_dataloader=train_dataset)
    # trainer.validate(model, MaeMaeDataModule())