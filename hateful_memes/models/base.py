import imp
import os
from icecream import ic

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision import transforms as T
import torchmetrics

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    StochasticWeightAveraging,
    Timer,
    LearningRateMonitor,
    BackboneFinetuning
)
from pytorch_lightning import Trainer
import wandb

from hateful_memes.utils import get_project_logger, BackBoneOverrider
from hateful_memes.data.hateful_memes import MaeMaeDataModule


class BaseMaeMaeModel(LightningModule):

    def __init__(
        self, 
        lr=0.0003, 
        weight_decay=0.00,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.wd = weight_decay

        # TODO log for each metric through macro
        metrics_kwargs = dict(compute_on_cpu=True)
        # metrics_kwargs = dict()
        self.train_acc = torchmetrics.Accuracy(**metrics_kwargs)
        self.train_f1 = torchmetrics.F1Score(average="micro", **metrics_kwargs)
        self.train_auroc = torchmetrics.AUROC(average="micro", **metrics_kwargs)
        self.val_acc = torchmetrics.Accuracy(**metrics_kwargs)
        self.val_f1 = torchmetrics.F1Score(average="micro", **metrics_kwargs)
        self.val_auroc = torchmetrics.AUROC(average="micro", **metrics_kwargs)

    def forward(self, batch):
        raise NotImplemented
    
    def training_step(self, batch, batch_idx):
        y = batch['label']
        y_hat = self(batch)
        # y_hat = torch.squeeze(y_hat)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))

        self.train_acc.update(y_hat, y)
        self.train_f1.update(y_hat, y)
        self.train_auroc.update(y_hat, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=len(y), sync_dist=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, batch_size=len(y), sync_dist=True)
        self.log("train/f1", self.train_f1, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(y), sync_dist=True)
        self.log("train/auroc", self.train_auroc, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(y), sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['label']
        y_hat = self(batch)
        # y_hat = torch.squeeze(y_hat)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))

        # ic(torch.sigmoid(y_hat))
        self.val_acc.update(y_hat, y)
        self.val_f1.update(y_hat, y)
        self.val_auroc.update(y_hat, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=len(y), sync_dist=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, batch_size=len(y), sync_dist=True)
        self.log("val/f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(y), sync_dist=True)
        self.log("val/auroc", self.val_auroc, on_step=False, prog_bar=True, on_epoch=True, batch_size=len(y), sync_dist=True)

        return loss

    def configure_optimizers(self):
        # filter for thawing
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.wd) 
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr) 

        return {
            "optimizer": optimizer,
            # "lr_scheduler": ReduceLROnPlateau(optimizer, patience=5, verbose=True),
            # "monitor": "train/loss",
        }


def base_train(
        *, 
        model, 
        project, 
        epochs, 
        batch_size=None, 
        model_dir='/tmp', 
        log_dir=None, 
        grad_clip=1.0, 
        fast_dev_run=False,
        monitor_metric="val/loss",
        monitor_metric_mode="min",
        stopping_patience=16,
        mixed_precision=False,
        finetune_epochs=10,
    ):
    logger = get_project_logger(project=project, save_dir=log_dir, offline=fast_dev_run)
    # TODO pull out lr and maybe arg optimizer


    callbacks = [
        ModelCheckpoint(
            monitor=monitor_metric,
            mode=monitor_metric_mode,
            dirpath=model_dir, 
            # filename="{epoch}-{step}-{loss:.4f}",
            verbose=True,
            save_top_k=1),
        EarlyStopping(
            monitor=monitor_metric,
            patience=stopping_patience, 
            mode=monitor_metric_mode,
            min_delta=0.0001,
            verbose=True),
        StochasticWeightAveraging(),
        LearningRateMonitor(),
        # Finetuner(model.backbones, finetune_epochs),
    ]

    try:
        model.backbone
        callbacks.append(BackBoneOverrider(finetune_epochs, verbose=True, backbone_initial_ratio_lr=0.01, lambda_func=lambda epoch: 1.2))
        ic("Adding finetuning")
    except AttributeError:
        ic("no backbone")
        pass


    if batch_size > 0:
        accumulate_grad_batches = max(1, 64//batch_size)
    else:
        accumulate_grad_batches = None
    if accumulate_grad_batches == 1:
        accumulate_grad_batches = None
    ic(accumulate_grad_batches)
    
    trainer = Trainer(
        devices=-1 if not fast_dev_run else 1,
        strategy="ddp",
        # gpus=[1],
        accelerator='auto',
        # replace_sampler_ddp=False,
        # auto_select_gpus=True,
        logger=logger,
        max_epochs=epochs,
        gradient_clip_val=grad_clip,
        # track_grad_norm=2, 
        fast_dev_run=fast_dev_run, 
        # auto_lr_find=True,
        auto_scale_batch_size='power' if batch_size <= 0 else False,
        precision=16 if mixed_precision else 32,
        # amp_backend='native',
        # detect_anomaly=True,
        enable_progress_bar=os.environ.get('ENABLE_PROGRESS_BAR', 1) == 1,
        accumulate_grad_batches=accumulate_grad_batches,
        # profiler="simple",
        # callbacks=[checkpoint_callback, early_stopping])
        callbacks=[*callbacks],
    )

    data = MaeMaeDataModule(batch_size=batch_size if batch_size > 0 else 32)
    ic(model.lr)

    if not fast_dev_run and batch_size <= 0:
        # TODO should I move datamodule inside lightning module?
        result = trainer.tune(
            model, 
            scale_batch_size_kwargs=dict(max_trials=6),
            lr_find_kwargs=dict(
                num_training=100, 
                update_attr=True),
            datamodule=data,
        )
        ic(result)
        batch_size = result['scale_batch_size']
        # lr_find = result['lr_find']
        # plt = lr_find.plot(suggest=True)
        # wandb.log({"lr_plot": plt})
        # new_lr = trainer.tuner.lr_find.suggestion()
        # ic(new_lr)

    #     # new_lr = trainer.tuner.lr_find.suggestion()
    #     # model.hparams.lr = new_lr
    #     # model.lr = new_lr
    data = MaeMaeDataModule(batch_size=batch_size if batch_size > 0 else 32)

    ic(model.lr)
    trainer.fit(
        model, 
        datamodule=data,
        )

    #############################################################
    # Output Results
    #############################################################
    # Setup data for predictions
    data = MaeMaeDataModule(batch_size=batch_size)
    data.setup(None)

    # Load train data
    train_data = data.train_dataloader(shuffle=False, drop_last=False)
    train_labels = []
    train_img_ids = []
    for batch in train_data:
        train_labels += batch['label'].tolist()
        train_img_ids += batch['img_id']

    # Load val data
    val_data = data.val_dataloader()
    val_labels = []
    val_img_ids = []
    for batch in val_data:
        val_labels += batch['label'].tolist()
        val_img_ids += batch['img_id']

    # Load test data
    test_data = data.test_dataloader()
    test_labels = []
    test_img_ids = []
    for batch in test_data:
        test_labels += batch['label'].tolist()
        test_img_ids += batch['img_id']

    # Get Predictions
    train_batched_preds, val_batched_preds, test_batched_preds= trainer.predict(
        model,
        dataloaders=[train_data, val_data, test_data],
        ckpt_path='best',
    )
    # train_pred, val_pred = trainer.predict(model, dataloaders=[train_data, val_data], ckpt_path='best')

    # organize predictions
    import pandas as pd
    import time

    train_preds = []
    for batch_preds in train_batched_preds:
        train_preds += nn.Sigmoid()(batch_preds).tolist()

    val_preds = []
    for batch_preds in val_batched_preds:
        val_preds += nn.Sigmoid()(batch_preds).tolist()

    test_preds = []
    for batch_preds in test_batched_preds:
        test_preds += nn.Sigmoid()(batch_preds).tolist()

    train_results = pd.DataFrame(
        dict(
            img_id=train_img_ids,
            label=train_labels,
            pred=train_preds,
            source='train',
        )
    )
    
    val_results = pd.DataFrame(
        dict(
            img_id=val_img_ids,
            label=val_labels,
            pred=val_preds,
            source='val',
        )
    )

    test_results = pd.DataFrame(
        dict(
            img_id=test_img_ids,
            label=test_labels,
            pred=test_preds,
            source='test',
        )
    )

    curr_time = time.strftime("%Y%m%d-%H%M%S")
    full_results = pd.concat([train_results, val_results, test_results])
    
    out_folder = model_dir.replace('06_models', '07_model_output')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    out_fp = os.path.join(out_folder, curr_time)

    full_results.to_pickle(f'{out_fp}.pkl')
    full_results.to_csv(f'{out_fp}.csv', index=False)
    print(f'Saved full results to {out_fp}.pkl')
    # train_cm = wandb.plot.confusion_matrix(
    #     y_true=train_labels,
    #     preds=train_pred,
    #     class_names=['not hateful', 'hateful'],
    # )
    # val_cm = wandb.plot.confusion_matrix(
    #     y_true=val_labels,
    #     preds=val_pred,
    #     class_names=['not hateful', 'hateful'],
    # )
    # wandb.log({"train_cm": train_cm})
    # wandb.log({"val_cm": val_cm})