from pytorch_lightning.loggers import WandbLogger, CSVLogger

def get_project_logger(*, project=None, save_dir=None, offline=False):
    """ Creates a logger for the project."""
    return [
        WandbLogger(project=project, offline=offline, log_model=not offline),
        CSVLogger(save_dir=save_dir)
    ]

