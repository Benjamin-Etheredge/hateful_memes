import torch
from torch.utils.data import SequentialSampler, BatchSampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging, EarlyStopping, ModelCheckpoint
import math
from typing import Optional, Any, Callable
import numpy as np
import argparse

from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import tasks as fs_tasks
from fairseq import utils as fs_utils
from fairseq import options
from OFA.models.ofa.ofa import OFAModel
from OFA.utils import checkpoint_utils
import OFA.utils.BPE
from OFA.tasks import OFATask
import pathlib
import sys

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
        self.batch_size = fs_cfg.dataset.batch_size

        self.max_positions = fs_utils.resolve_max_positions(
            self.ofa_task.max_positions(),
            ofa_model.max_positions(),
            self.fs_cfg.dataset.max_tokens,
        )

    def prepare_data(self) -> None:
        self.ofa_task.load_dataset("train", combine=True, epoch=1)
        self.ofa_task.load_dataset("valid", combine=True, epoch=1)     

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = self.ofa_task.dataset("train")
        self.val_dataset = self.ofa_task.dataset("valid")
        return super().setup(stage)

    def train_transforms(self):
        return super().train_transforms

    def train_dataloader(self):
        # Create data loader
        train_itr = DataLoader(
            self.train_dataset,
            batch_sampler=BatchSampler(SequentialSampler(self.train_dataset), batch_size=self.batch_size, drop_last=True),
            collate_fn=self.train_dataset.collater,
            num_workers=self.fs_cfg.dataset.num_workers,
            timeout=0,
            pin_memory=False
        )
        return train_itr

    def val_dataloader(self):
        # Create data loader
        val_itr = DataLoader(
            self.val_dataset,
            batch_sampler=BatchSampler(SequentialSampler(self.val_dataset), batch_size=self.batch_size, drop_last=True),
            collate_fn=self.val_dataset.collater,
            num_workers=self.fs_cfg.dataset.num_workers,
            timeout=0,
            pin_memory=False
        )
        return val_itr

    def teardown(self, stage: Optional[str] = None) -> None:
        return super().teardown(stage)


class HatefulOFA(pl.LightningModule):
    """OFA finetuned for Hateful Memes"""
    def __init__(self, cfg:FairseqConfig, ofa_task:OFATask,                 
                 adamw_eps=1e-8, adamw_betas=(0.9, 0.999), adamw_decay=0.01,
                 ema_active=False, label_smoothing=0.0) -> None:
        super().__init__()
        self.ofa_model = ofa_task.build_model(cfg.model)
        # Injest loss function configuration params
        tgt_dict = ofa_task.target_dictionary
        self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100
        self.ign_prefix_sz = 0
        if hasattr(cfg, 'ignore_prefix_size'):
            self.ign_prefix_sz = cfg.ignore_prefix_size
        self.label_smoothing = label_smoothing
        self.ema_active = ema_active
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
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._shared_step(batch)
        return loss

    """Validation"""
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
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
        optim_dict = {}
        
        # TODO OFA originally scales gradients by batch size
        params = self.parameters()  
        optim = torch.optim.AdamW(
            params,
            betas=self.hparams.adamw_betas, 
            eps=self.hparams.adamw_eps, 
            weight_decay=self.hparams.adamw_decay
        )
        
        optim_dict["optimizer"] = optim
        # OFA uses polynomial decay lr scheduler, but cosine annealing with WR was shown to get better results?
        # This actually can't operate alongside SWA, because the SWA callback has it's own LR scheduler baked in.
        # But SWA scheduler also does cosine annealing, so should be totally fine.

        if not self.ema_active: 
            #TODO add these params to YAML
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optim,
                T_0=1,
                T_mult=2    
            )
            optim_dict["lr_scheduler"] = {}
            optim_dict["lr_scheduler"]["scheduler"] = lr_sched
        
        return optim_dict

# TODO click arguments 
def main(
             modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
        ):
    # Dumb hack
    bpe_dir = str(pathlib.Path(utils.BPE.__file__).resolve().parent)
    sys.argv.append("--bpe-dir %s" % (bpe_dir))
    # Injest CLI arguments
    parser = options.get_training_parser()
    args, extra = options.parse_args_and_arch(parser, modify_parser=modify_parser, parse_known=True)
    cfg = convert_namespace_to_omegaconf(args)
    cfg.task.bpe_dir = bpe_dir
    # utils.import_user_module(cfg.common)
    np.random.seed(cfg.common.seed)
    fs_utils.set_torch_seed(cfg.common.seed)
    # Grab project-specific CLI args from extra args left over from OFA parser
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--label-smoothing', dest='label_smoothing', type=float, default=0.0)
    my_parser.add_argument('--fast-dev-run', dest='fast_dev_run', type=int, default=1)
    my_parser.add_argument("--ema-alpha", dest='ema_alpha', type=float, default=0.0)
    my_parser.add_argument("--monitor-metric", dest='monitor_metric', type=str, default="val/loss")
    my_parser.add_argument("--monitor-metric-mode", dest='monitor_metric_mode', type=str, default="min", choices=["min", "max"])
    my_parser.add_argument("--stopping-patience", dest='stopping_patience', type=int, default=10)
    my_args, _ = my_parser.parse_known_args(extra)
    print("parsed args")
    print(my_args)

    # Build the model
    OFA_TASK = fs_tasks.setup_task(cfg.task)
    adamw_eps = cfg.optimizer.adam_eps
    adamw_betas = [float(beta) for beta in cfg.optimizer.adam_betas.split(',')]
    adamw_decay = cfg.optimizer.weight_decay
    ema_alpha = my_args.ema_alpha 
    ema_active = ema_alpha > 0.0
    label_smoothing = my_args.label_smoothing
    hateful_ofa_model = HatefulOFA(cfg, ofa_task=OFA_TASK, 
        adamw_eps=adamw_eps, adamw_betas=adamw_betas, adamw_decay=adamw_decay,
        ema_active=ema_active, label_smoothing=label_smoothing)
    
    # Load the datasets using the task
    hateful_ofa_data = HatefulOFADataModule(cfg, hateful_ofa_model.ofa_model, ofa_task=OFA_TASK)
    
    # Set up training strategy
    #TODO logger
    max_epoch = cfg.optimization.max_epoch or math.inf
    model_dir = cfg.checkpoint.save_dir
    fast_dev_run = my_args.fast_dev_run > 0
    monitor_metric = my_args.monitor_metric
    monitor_metric_mode = my_args.monitor_metric_mode
    stopping_patience = my_args.stopping_patience  
    # Define callbacks
    trainer_callbacks=[]
    # If training:
    if ~fast_dev_run:
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_metric,
            mode=monitor_metric_mode,
            dirpath=model_dir, 
            # filename="{epoch}-{step}-{loss:.4f}",
            verbose=True,
            save_top_k=1)
        trainer_callbacks.append(checkpoint_callback)
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=stopping_patience, 
            mode=monitor_metric_mode,
            verbose=True)
        trainer_callbacks.append(early_stopping)
    else:
        print("FAST DEV RUN")
    # If EMA is specified
    if ema_alpha > 0.0:
        ema_fn = ExponentialMovingAverage(alpha=ema_alpha)
        swa_callback = StochasticWeightAveraging(avg_fn=ema_fn)
        trainer_callbacks.append(swa_callback)
    grad_norm_clip = cfg.optimization.clip_norm
    hateful_ofa_trainer = pl.Trainer(
        max_epochs=max_epoch,
        callbacks=trainer_callbacks,
        gradient_clip_val=grad_norm_clip,
        fast_dev_run=fast_dev_run
    )

    # Training
    hateful_ofa_trainer.fit(hateful_ofa_model, datamodule=hateful_ofa_data)


if __name__ == '__main__':
    main()