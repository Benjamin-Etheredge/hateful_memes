import torch
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging
import math
from typing import Optional, Any, Callable
import numpy as np
import argparse

from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import (tasks, utils, options)
from models.ofa.ofa import OFAModel
from utils import checkpoint_utils
from tasks import OFATask


# For SWA callback in Trainer
class ExponentialMovingAverage:
    def __init__(self, alpha:float):
        self.alpha = alpha
    def __call__(self, averaged_model_parameter: torch.Tensor, model_parameter: torch.Tensor, num_averaged: torch.LongTensor
        ) -> torch.FloatTensor:
        ema_avg = self.alpha * averaged_model_parameter + (1.0 - self.alpha) * model_parameter
        return ema_avg

class HatefulOFADataModule(pl.LightningDataModule):
    def __init__(self, fs_cfg:FairseqConfig, ofa_model:OFAModel, ofa_task=OFATask,
                 train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.fs_cfg = fs_cfg
        self.ofa_task = ofa_task

        self.max_positions = utils.resolve_max_positions(
            self.ofa_task.max_positions(),
            ofa_model.max_positions(),
            self.fs_cfg.dataset.max_tokens,
        )

    def prepare_data(self) -> None:
        self.ofa_task.load_dataset("train", combine=True, epoch=1)
        self.ofa_task.load_dataset("valid", combine=True, epoch=1)     
        return super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = self.ofa_task.dataset("train")
        self.val_dataset = self.ofa_task.dataset("valid")
        return super().setup(stage)

    def train_transforms(self):
        return super().train_transforms

    def train_dataloader(self) -> DataLoader:
        # Create data loader
        train_itr = DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collater,
            num_workers=self.fs_cfg.dataset.num_workers,
            timeout=0,
            pin_memory=True,
        )
        return train_itr

    def val_dataloader(self) -> DataLoader:
        # Create data loader
        val_itr = DataLoader(
            self.val_dataset,
            collate_fn=self.val_dataset.collater,
            num_workers=self.fs_cfg.dataset.num_workers,
            timeout=0,
            pin_memory=True,
        )
        return val_itr

    def teardown(self, stage: Optional[str] = None) -> None:
        return super().teardown(stage)


class HatefulOFA(pl.LightningModule):
    """OFA finetuned for Hateful Memes"""
    def __init__(self, cfg:FairseqConfig, ofa_task:OFATask,                 
                 adamw_eps=1e-8, adamw_betas=(0.9, 0.999), adamw_decay=0.01) -> None:
        super().__init__()
        self.ofa_model = ofa_task.build_model(cfg.model)
        # Injest loss function configuration params
        tgt_dict = ofa_task.target_dictionary
        self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100
        self.ign_prefix_sz = 0
        if hasattr(cfg, 'ignore_prefix_size'):
            self.ign_prefix_sz = cfg.ignore_prefix_size
        self.label_smoothing = 0.0
        #if hasattr(cfg, 'label_smoothing'):
        #    self.label_smoothing = cfg.label_smoothing
        # Hyperparameters
        self.save_hyperparameters("adamw_eps", "adamw_betas", "adamw_decay")
        # Try to load params from checkpoint
        filename = cfg.checkpoint.restore_file   
        load_on_all_ranks = False  # Unless distributed training
        state = checkpoint_utils.load_checkpoint_to_cpu(
            filename, load_on_all_ranks=load_on_all_ranks
        )
        try:
            self.ofa_model.load_state_dict(
                state["model"], strict=True, model_cfg=cfg.model
            )
        except Exception:
            raise Exception(
                "Cannot load model parameters from checkpoint {}; "
                "please ensure that the architectures match.".format(filename)
            )

    """Forward function"""
    def forward(self, batch) -> torch.Tensor:
        net_output = self.ofa_model(**batch['net_input'])
        return net_output

    """Loss function"""
    def _get_lprobs(self, net_output):
        lprobs = self.ofa_model.get_normalized_probs(net_output, log_probs=True)
        return lprobs

    def _get_targets(self, batch):
        targets = batch["target"]
        return targets

    def _prep_lprobs_targets_for_loss(self, ign_prefix_sz, padding_idx, net_output, batch):
        constraint_masks = None
        if "constraint_masks" in batch and batch["constraint_masks"] is not None:
            constraint_masks = batch["constraint_masks"]
            net_output[0].masked_fill_(~constraint_masks, -math.inf)
        lprobs = self._get_lprobs(net_output)
        targets = self._get_targets(batch)
        if ign_prefix_sz > 0:
            lprobs = lprobs[:, ign_prefix_sz :, :].contiguous()
            targets = targets[:, ign_prefix_sz :].contiguous()
            if constraint_masks is not None:
                constraint_masks = constraint_masks[:, ign_prefix_sz :, :].contiguous()
        if constraint_masks is not None:
            constraint_masks = constraint_masks.view(-1, constraint_masks.size(-1))
        lprobs = lprobs.view(-1, lprobs.size(-1))
        targets = targets.view(-1)
        if constraint_masks is not None:
            constraint_masks = constraint_masks[targets != padding_idx]
        lprobs = lprobs[targets != padding_idx]
        targets = targets[targets != padding_idx]
        if targets.dim() == lprobs.dim() - 1:
            targets = targets.unsqueeze(-1)
        return lprobs, targets, constraint_masks

    def _label_smoothed_nll_loss(self, lprobs, target, epsilon=0.0, constraint_masks=None):
        nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
        if constraint_masks is not None:
            smooth_loss = -lprobs.masked_fill(~constraint_masks, 0).sum(dim=-1, keepdim=True).squeeze(-1)
            eps_i = epsilon / (constraint_masks.sum(1) - 1 + 1e-6)
        else:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True).squeeze(-1)
            eps_i = epsilon / (lprobs.size(-1) - 1)
        loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
        loss = loss.sum()
        return loss

    """Training and validation"""
    def _shared_step(self, batch) -> torch.Tensor:
        # Get raw output, y_hat
        net_output = self(batch)
        # Prep for loss function
        ign_prefix_sz = self.ign_prefix_sz
        padding_idx = self.padding_idx
        lprobs, targets, constraints_masks = self._prep_lprobs_targets_for_loss(ign_prefix_sz, 
                                                                                padding_idx,
                                                                                net_output, 
                                                                                batch)
        # Calculate label-smoothed CE loss
        label_smoothing = self.label_smoothing
        loss = self._label_smoothed_nll_loss(lprobs, targets, label_smoothing, constraints_masks)
        return loss

    """Training"""
    def training_step(self, batch) -> torch.Tensor:
        loss = self._shared_step(batch)
        return loss

    """Validation"""
    def validation_step(self, batch) -> torch.Tensor:
        loss = self._shared_step(batch)
        return loss

    """Testing"""
    def test_step(self, *args, **kwargs):
        # TODO EMA
        return super().test_step(*args, **kwargs)

    """Inference"""
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        # TODO EMA
        return super().predict_step(batch, batch_idx, dataloader_idx)

    """Optimizer"""
    def configure_optimizers(self):
        # TODO OFA originally scales gradients by batch size
        
        params = self.parameters()  
        optim = torch.optim.AdamW(
            params,
            betas=self.hparams.adamw_betas, 
            eps=self.hparams.adamw_eps, 
            weight_decay=self.hparams.adamw_decay
        )
        # OFA uses polynomial decay lr scheduler, but cosine annealing with WR was shown to get better results?
        #TODO add these params to YAML
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optim,
            T_0=1,
            T_mult=2    
        )
        optim_dict = {
        "optimizer": optim,
        "lr_scheduler": {
            "scheduler": lr_sched
            # "monitor": "metric_to_track",
            # "frequency": "indicates how often the metric is updated"
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        return optim_dict

# TODO click arguments 
def main(
             modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
        ):
    # Injest CLI arguments
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    cfg = convert_namespace_to_omegaconf(args)
    # utils.import_user_module(cfg.common)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    # User CLI args
    working = False
    if working:
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('--fast-dev-run', dest='fast_dev_run', action='store_true')
        my_parser.add_argument('--fast-dev-run', dest='fast_dev_run', action='store_false')
        my_parser.set_defaults(fast_dev_run=False)
        my_parser.add_argument("--ema-alpha", dest='ema_alpha', type=float)
        my_args = my_parser.parse_known_args(args)
        fast_dev_run = my_args.fast_dev_run
        ema_alpha = my_args.ema_alpha   
    else:
        #TODO
        fast_dev_run = 2
        ema_alpha = 0.1

    # Set up for training
    ## First build the model
    OFA_TASK = tasks.setup_task(cfg.task)
    adamw_eps = cfg.optimizer.adam_eps
    adamw_betas = [float(beta) for beta in cfg.optimizer.adam_betas.split(',')]
    adamw_decay = cfg.optimizer.weight_decay
    hateful_ofa_model = HatefulOFA(cfg, ofa_task=OFA_TASK, 
        adamw_eps=adamw_eps, adamw_betas=adamw_betas, adamw_decay=adamw_decay)
    ## Second load the datasets using the task
    hateful_ofa_data = HatefulOFADataModule(cfg, hateful_ofa_model.ofa_model, ofa_task=OFA_TASK)
    ## Third set up training strategy
    max_epoch = cfg.optimization.max_epoch or math.inf
    ema_fn = ExponentialMovingAverage(alpha=ema_alpha)
    grad_norm_clip = cfg.optimization.clip_norm
    hateful_ofa_trainer = pl.Trainer(
        max_epochs=max_epoch,
        callbacks=[StochasticWeightAveraging(avg_fn=ema_fn)],
        gradient_clip_val=grad_norm_clip,
        fast_dev_run=fast_dev_run)

    # Training
    hateful_ofa_trainer.fit(hateful_ofa_model, datamodule=hateful_ofa_data)


if __name__ == '__main__':
    main()