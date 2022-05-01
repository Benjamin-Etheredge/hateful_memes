import os
from pytorch_lightning.loggers import WandbLogger, CSVLogger


def get_project_logger(*, project=None, save_dir=None, offline=False):
    """ Creates a logger for the project."""
    # return True
    return [
        WandbLogger(project=project, offline=offline, log_model=not offline),
        # DvcLiveLogger(path=save_dir, report=None)
    ]


def get_checkpoint_filename(path: str) -> str:
    """
    Get the latest checkpoint filename.
    """
    files = os.listdir(path)
    files = [f for f in files if f.endswith(".ckpt")]
    assert (len(files) > 0)
    files.sort()
    return files[-1]

def get_checkpoint_path(path: str) -> str:
    """
    Get the latest checkpoint filename.
    """
    files = os.listdir(path)
    files = [f for f in files if f.endswith(".ckpt")]
    assert (len(files) > 0)
    files.sort()
    return os.path.join(path, files[-1])


from pytorch_lightning.callbacks import Callback
import wandb
 
class LogPredictionsCallback(Callback):
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]
            
            # Option 1: log images with `WandbLogger.log_image`
            pl_module.log_image(key='sample_images', images=images, caption=captions)
            # Option 2: log predictions as a Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            pl_module.log_table(key='sample_table', columns=columns, data=data)

# class LogConfusionMatrixCallback(Callback):
#     def on
#     def on(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

#     def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    

from pytorch_lightning.callbacks import BaseFinetuning, BackboneFinetuning
from typing import List
import pytorch_lightning as pl
class Finetuner(BaseFinetuning):
    def __init__(self, sub_models, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.sub_models = sub_models

    def freeze_before_training(self, model: "pl.LightningModule") -> None:
        for model in self.sub_models:
            self.freeze(model)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            print("Unfreezing backbone models")
            for model in self.sub_models:
                self.unfreeze_and_add_param_group(
                    modules=model,
                    optimizer=optimizer,
                    train_bn=True,
                        )
                         

class BackBoneOverrider(BackboneFinetuning):
    # Working around https://github.com/PyTorchLightning/pytorch-lightning/issues/12946
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass