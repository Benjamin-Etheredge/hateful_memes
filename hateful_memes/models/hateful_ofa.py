import pytorch_lightning as pl
import math
from models.OFA.fairseq.fairseq.dataclass.configs import FairseqConfig
from models.OFA.fairseq.fairseq.dataclass.utils import convert_namespace_to_omegaconf
from models.OFA.fairseq.fairseq import (tasks, utils, options)
from models.OFA.trainer import Trainer
from models.OFA.utils import checkpoint_utils
from typing import Dict, Optional, Any, List, Tuple, Callable
import numpy as np
import argparse


class hateful_ofa(pl.LightningModule):
    """OFA finetuned for Hateful Memes"""
    def __init__(self, cfg:FairseqConfig) -> None:
        super().__init__()
        # FairSeq task        
        task = tasks.setup_task(cfg.task)
        # OFA model
        model = task.build_model(cfg.model)
        # Training criterion
        criterion = task.build_criterion(cfg.criterion)
        # Load valid dataset (we load training data below, based on the latest checkpoint)
        # We load the valid dataset AFTER building the model
        # data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
        if cfg.dataset.combine_valid_subsets:
            task.load_dataset("valid", combine=True, epoch=1)
        else:
            for valid_sub_split in cfg.dataset.valid_subset.split(","):
                task.load_dataset(valid_sub_split, combine=False, epoch=1) 
        # Custom Trainer class
        quantizer = None  # Unless distributed training
        trainer = Trainer(cfg, task, model, criterion, quantizer)
        # Load the latest checkpoint if one is available and restore the corresponding train iterator
        extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
            cfg.checkpoint,
            trainer,
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=True,
        )
        # Max epochs
        max_epoch = cfg.optimization.max_epoch or math.inf
        if max_epoch > 0 and max_epoch != math.inf:
            total_num_updates = sum(
                math.ceil(len(epoch_itr) / cfg.optimization.update_freq[i])
                if i < len(cfg.optimization.update_freq) else
                math.ceil(len(epoch_itr) / cfg.optimization.update_freq[-1])
                for i in range(max_epoch)
            )
            trainer.lr_reinit(total_num_updates, trainer.get_num_updates())
 
    def forward(self, *args, **kwargs) -> Any:
        return super().forward(*args, **kwargs)

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        # forward and backward
        model.train()
        loss, sample_size, logging_output = criterion(model, sample, update_num=update_num)
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().test_step(*args, **kwargs)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx)

    def configure_optimizers(self):
        return super().configure_optimizers()
    
def main(
             modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
        ):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    cfg = convert_namespace_to_omegaconf(args)
    utils.import_user_module(cfg.common)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

