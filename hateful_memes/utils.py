from pytorch_lightning.loggers import WandbLogger, LightningLoggerBase
from typing import List
class MultiLogger(LightningLoggerBase):
    def __init__(
        self, 
        loggers: List[LightningLoggerBase], 
        *args, **kwargs
    ):
        self.loggers = loggers