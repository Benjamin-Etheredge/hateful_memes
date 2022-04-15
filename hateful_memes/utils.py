import os
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from dvclive.lightning import DvcLiveLogger


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