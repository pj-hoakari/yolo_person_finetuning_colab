"""Enforce runtime storage and disable integrations before importing YOLO."""
import os
import contextlib
import logging
from pathlib import Path

BASE = Path('/content/crowdhuman')


def runtime_path(path):
    path = Path(path).absolute()
    resolved = path.resolve()
    if resolved != BASE and BASE not in resolved.parents:
        raise ValueError('CrowdHuman paths must stay under /content/crowdhuman')
    return path


def initialize(base=None):
    base = runtime_path(BASE if base is None else base)
    if base != BASE:
        raise ValueError('BASE_DIR must be /content/crowdhuman')
    for part in ('raw/archives', 'raw/images', 'raw/annotations', 'processed', 'runs', 'exports', 'weights', 'config'):
        runtime_path(base / part).mkdir(parents=True, exist_ok=True)
    os.environ.update(YOLO_CONFIG_DIR=str(base / 'config'), YOLO_OFFLINE='true',
                      YOLO_AUTOINSTALL='false', YOLO_VERBOSE='false',
                      WANDB_MODE='disabled', COMET_MODE='DISABLED',
                      WANDB_DISABLED='true', HF_HUB_OFFLINE='1')
    return base


def private_yolo():
    initialize()
    from ultralytics import YOLO, settings
    keys = ('sync', 'hub', 'wandb', 'mlflow', 'clearml', 'comet', 'dvc', 'neptune', 'raytune', 'tensorboard')
    settings.update({key: False for key in keys if key in settings})
    settings.update({'datasets_dir': str(BASE / 'processed'),
                     'weights_dir': str(BASE / 'weights'), 'runs_dir': str(BASE / 'runs')})
    # Disable integration registration, including Platform and analytics callbacks.
    from ultralytics.utils import callbacks
    callbacks.add_integration_callbacks = lambda instance: None
    from ultralytics.utils.callbacks import base
    base.add_integration_callbacks = lambda instance: None
    return YOLO


@contextlib.contextmanager
def local_console(path):
    """Capture pre-existing YOLO log handlers as well as stdout/progress bars."""
    path = runtime_path(path)
    logger = logging.getLogger('ultralytics')
    old_handlers, old_propagate = logger.handlers[:], logger.propagate
    with open(path, 'a') as log:
        logger.handlers = [logging.StreamHandler(log)]
        logger.propagate = False
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                yield
        finally:
            logger.handlers, logger.propagate = old_handlers, old_propagate
