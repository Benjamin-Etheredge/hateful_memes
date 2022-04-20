import torch
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader
import torch.nn.functional as torch_func
import pytorch_lightning as pl
import math
from models.OFA.fairseq.fairseq.dataclass.configs import FairseqConfig
from models.OFA.fairseq.fairseq.dataclass.utils import convert_namespace_to_omegaconf
from models.OFA.fairseq.fairseq import (tasks, utils, options)
from models.OFA.models.ofa.ofa import OFAModel
from models.OFA.trainer import Trainer
from models.OFA.utils import checkpoint_utils
from typing import Dict, Optional, Any, List, Tuple, Callable
import numpy as np
import argparse

OFA_TASK = tasks.setup_task("snli_ve")

class HatefulOFADataModule(pl.LightningDataModule):
    def __init__(self, fs_cfg:FairseqConfig, ofa_model:OFAModel, ofa_task=OFA_TASK, 
                 train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims)
        self.fs_cfg = fs_cfg
        self.ofa_task = ofa_task
        self.max_positions = utils.resolve_max_positions(
            self.ofa_task.max_positions(),
            ofa_model.max_positions(),
            self.cfg.dataset.max_tokens,
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
            batch_sampler=SequentialSampler,
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
            batch_sampler=SequentialSampler,
            num_workers=self.fs_cfg.dataset.num_workers,
            timeout=0,
            pin_memory=True,
        )
        return val_itr

    def teardown(self, stage: Optional[str] = None) -> None:
        return super().teardown(stage)

class HatefulOFA(pl.LightningModule):
    """OFA finetuned for Hateful Memes"""
    def __init__(self, cfg:FairseqConfig, ofa_task=OFA_TASK) -> None:
        super().__init__()
        self.ofa_model = ofa_task.build_model(cfg.model)
        # Injest loss function configuration params
        # TODO
        # Try to load params from checkpoint
        filename = cfg.checkpoint.restore_file  # TODO make sure this is right
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
    def _get_lprobs(self, net_output):
        lprobs = self.ofa_model.get_normalized_probs(net_output, log_probs=True)
        return lprobs
    
    def forward(self, batch) -> torch.Tensor:
        net_output = self.ofa_model(**batch['net_input'])
        lprobs = self._get_lprobs(net_output)
        return lprobs

    """Loss function"""
    def _get_targets(self, batch):
        targets = batch["target"]
        return targets

    def _prep_lprobs_targets_for_loss(self, ignore_prefix_size, lprobs, targets):
        if ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, ignore_prefix_size :, :].contiguous()
                targets = targets[:, ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[ignore_prefix_size :, :, :].contiguous()
                targets = targets[ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), targets.view(-1)

    def _label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / (lprobs.size(-1) - 1)
        loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
        return loss

    """Training"""
    def training_step(self, batch) -> torch.Tensor:
        # Get y, y_hat
        lprobs_0 = self(batch)
        targets_0 = self._get_targets(batch)
        # Prep for loss function
        ign_prefix_sz = self.loss_cfg.ignore_prefix_size
        lprobs, targets = self._prep_lprobs_targets_for_loss(ign_prefix_sz, lprobs_0, targets_0)
        # Get loss function params
        label_smoothing = self.loss_cfg.label_smoothing
        ignore_idx = self.loss_cfg.ignore_index
        reduce = self.loss_cfg.reduce
        # Calculate label-smoothed CE loss
        loss = self._label_smoothed_nll_loss(lprobs, targets, 
                                             epsilon=label_smoothing,
                                             ignore_index=ignore_idx,
                                             reduce=reduce)
        return loss

    """Validation"""
    def validation_step(self, batch) -> torch.Tensor:
        # Get y, y_hat
        lprobs_0 = self(batch)
        targets_0 = self._get_targets(batch)
        # Prep for loss function
        ign_prefix_sz = self.loss_cfg.ignore_prefix_size
        lprobs, targets = self._prep_lprobs_targets_for_loss(ign_prefix_sz, lprobs_0, targets_0)
        # Get loss function params
        label_smoothing = self.loss_cfg.label_smoothing
        ignore_idx = self.loss_cfg.ignore_index
        reduce = self.loss_cfg.reduce
        # Calculate label-smoothed CE loss
        loss = self._label_smoothed_nll_loss(lprobs, targets, 
                                             epsilon=label_smoothing,
                                             ignore_index=ignore_idx,
                                             reduce=reduce)
        return loss

    """Testing"""
    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().test_step(*args, **kwargs)

    """Inference"""
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx)

    """Optimizer"""
    def configure_optimizers(self):
        # TODO OFA originally scales gradients by batch size
        return super().configure_optimizers()


# TODO click arguments 
def main(
             modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
        ):
    # Injest CLI arguments
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    cfg = convert_namespace_to_omegaconf(args)
    utils.import_user_module(cfg.common)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    
    # Set up for training
    ## First build the model
    hateful_ofa_model = HatefulOFA(cfg)
    ## Second load the datasets using the task
    hateful_ofa_data = HatefulOFADataModule(cfg, hateful_ofa_model.ofa_model)
    ## Third set up training strategy TODO
    max_epoch = cfg.optimization.max_epoch or math.inf
    hateful_ofa_trainer = pl.Trainer()

    # Training
    hateful_ofa_trainer.fit(hateful_ofa_model, datamodule=hateful_ofa_data)

