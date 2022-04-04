from gc import callbacks
from models.baseline import BaseTextMaeMaeModel
from hateful_memes.data.hateful_memes import MaeMaeDataset

from hateful_memes.data.hateful_memes import MaeMaeDataModule
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="Hateful_Memes_Base_Text", log_model=True)
    checkpoint_callback = ModelCheckpoint(monitor="val/acc", mode="max", dirpath="data/06_models/hateful_memes", save_top_k=1)
    early_stopping = EarlyStopping(
            monitor='val/acc', 
            patience=10, 
            mode='max', 
            verbose=True)

    trainer = Trainer(
        gpus=1, 
        max_epochs=100, 
        logger=wandb_logger, 
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stopping])
    
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
        collate_fn=MaeMaeDataset.collate_batch,

    )
    model = BaseTextMaeMaeModel(
         vocab_size=len(dataset.vocab),
         batch_size=32
    )
    trainer.fit(model, datamodule=MaeMaeDataModule(batch_size=32))