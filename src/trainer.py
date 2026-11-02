"""Local-only, reproducible training with explicit checkpoint resumption."""
import importlib.metadata
import json
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from .privacy import BASE, private_yolo, runtime_path, local_console
from .converter import NAMES, validate_labels
from .checkpoints import drive_experiment_dir, save_drive_checkpoint, restore_drive_checkpoint
from .dataset import _download_error_details


@dataclass(frozen=True)
class TrainConfig:
    mode: str = 'vbox'
    model: str = 'yolo26n.pt'
    epochs: int = 50
    batch: int = 16
    imgsz: int = 640
    seed: int = 42
    patience: int = 10
    workers: int = 2
    debug: bool = False
    run_name: str = 'experiment'
    allow_cpu: bool = False

    def validate(self):
        if self.mode not in NAMES or min(self.epochs, self.batch, self.imgsz) <= 0:
            raise ValueError('Invalid training configuration')
        if self.imgsz % 32 or self.workers < 0 or self.patience < 0:
            raise ValueError('Invalid image size, workers or patience')
        if self.model not in ('yolo26n.pt', 'yolo26s.pt'):
            raise ValueError('Use the reference YOLO26n/s model')
        if not self.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in self.run_name):
            raise ValueError('Unsafe run name')

    @property
    def run_dir(self):
        self.validate()
        return runtime_path(BASE / 'runs' / self.mode / (self.run_name + ('_debug' if self.debug else '')))


def versions():
    names = ('ultralytics', 'torch', 'torchvision', 'Pillow', 'PyYAML', 'onnx', 'onnxruntime')
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not installed'
    return result


def device_for(config):
    import torch
    if torch.cuda.is_available():
        return 0
    warnings.warn('GPUがありません。ColabのGPUランタイムを選択してください。', stacklevel=2)
    if not config.allow_cpu:
        raise RuntimeError('GPU required; set allow_cpu=True only for a deliberate CPU test')
    return 'cpu'


def load_model(config):
    config.validate()
    YOLO = private_yolo()
    previous = Path.cwd()
    try:
        os.chdir(runtime_path(BASE / 'weights'))
        return YOLO(str(BASE / 'weights' / config.model))
    finally:
        os.chdir(previous)


def train(config, resume=False, drive_runs_dir=None, checkpoint_status=None):
    config.validate()
    import numpy as np
    import torch
    device = device_for(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    root = runtime_path(BASE / 'processed' / config.mode)
    validate_labels(root, config.mode)
    conversion = json.loads((root / 'conversion.json').read_text())
    if (conversion['limit'] is not None) != config.debug:
        raise ValueError('DEBUG flag and converted data disagree')
    run = config.run_dir
    meta_path = run / 'experiment.json'
    specification = {'config': asdict(config), 'dataset_fingerprint': conversion['fingerprint'], 'versions': versions()}
    if drive_runs_dir is not None:
        drive_run = drive_experiment_dir(config, drive_runs_dir)
        if not resume and (drive_run / 'latest.json').exists():
            raise FileExistsError('Drive experiment exists; resume it or select a new run_name')
        if resume and not meta_path.exists():
            restore_drive_checkpoint(config, drive_runs_dir, specification)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if not resume:
            raise FileExistsError('Run exists; explicitly resume or select a new run_name')
        if meta['specification'] != specification:
            raise ValueError('Checkpoint configuration/dataset/library mismatch')
    else:
        if resume or run.exists():
            raise ValueError('No matching resumable experiment metadata')
        run.mkdir(parents=True)
        meta = {'specification': specification, 'sessions': []}
        meta_path.write_text(json.dumps(meta, indent=2))
    checkpoint = runtime_path(run / 'weights/last.pt')
    YOLO = private_yolo()
    if resume:
        if not checkpoint.is_file():
            raise FileNotFoundError('Missing last.pt checkpoint')
        # Only our trusted runtime checkpoint; never load arbitrary uploaded pickle files.
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ckpt.get('epoch', -1) < 0 or ckpt.get('optimizer') is None:
            raise ValueError('Completed/stripped checkpoint cannot resume; restore the raw Drive snapshot into a fresh local run')
        if ckpt['epoch'] + 1 >= config.epochs:
            raise ValueError('Checkpoint already reached the configured final epoch')
        del ckpt
    started = time.perf_counter()
    status = 'interrupted'
    session = {'seconds': 0.0, 'status': 'running', 'resume': resume}
    meta['sessions'].append(session)
    meta_path.write_text(json.dumps(meta, indent=2))

    def persist_epoch(trainer_instance):
        epoch = int(trainer_instance.epoch) + 1
        session['seconds'] = time.perf_counter() - started
        meta_path.write_text(json.dumps(meta, indent=2))
        try:
            saved = save_drive_checkpoint(config, drive_runs_dir, epoch)
        except Exception as error:
            failure = {'epoch': epoch, 'type': type(error).__name__,
                       'message': 'Drive checkpoint save failed', 'errors': _download_error_details(error)}
            (run / 'drive_backup_error.json').write_text(json.dumps(failure, indent=2))
            raise RuntimeError(f'Drive checkpoint save failed at epoch {epoch}; training stopped so unsaved epochs do not accumulate') from error
        if checkpoint_status is not None:
            checkpoint_status(saved)

    try:
        # Keep epoch progress and diagnostic output out of the notebook document.
        with local_console(run / 'console.log'):
            model = YOLO(str(checkpoint)) if resume else load_model(config)
            if drive_runs_dir is not None:
                model.add_callback('on_model_save', persist_epoch)
            if resume:
                model.train(resume=True, data=str(root / 'data.yaml'), plots=False,
                            save=True, save_period=-1, save_json=False, save_txt=False, device=device)
            else:
                model.train(data=str(root / 'data.yaml'), epochs=config.epochs, batch=config.batch,
                            imgsz=config.imgsz, seed=config.seed, deterministic=True,
                            device=device, project=str(run.parent), name=run.name, exist_ok=True,
                            patience=config.patience, workers=config.workers, cache=False,
                            single_cls=False, cos_lr=True, close_mosaic=min(10, config.epochs),
                            shear=0.0, perspective=0.0, amp=True, plots=False,
                            max_det=1000, save=True, save_period=-1,
                            save_json=False, save_txt=False, verbose=False)
        status = 'completed'
    finally:
        session.update(seconds=time.perf_counter()-started, status=status)
        meta_path.write_text(json.dumps(meta, indent=2))
    return run
