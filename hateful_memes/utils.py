from pytorch_lightning.loggers import WandbLogger, CSVLogger
from dvclive.lightning import DvcLiveLogger


def get_project_logger(*, project=None, save_dir=None, offline=False):
    """ Creates a logger for the project."""
    # return True
    return [
        WandbLogger(project=project, offline=offline, log_model=not offline),
        DvcLiveLogger(path=save_dir, report=None)
    ]

