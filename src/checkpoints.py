"""Verified private Drive snapshots, committed only after every file has copied."""
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from .privacy import runtime_path

CHECKPOINT_FILES = ('weights/last.pt', 'weights/best.pt', 'experiment.json', 'results.csv', 'args.yaml')


def _checkpoint_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_drive_path(path, root):
    path = Path(path).absolute()
    if root.resolve() != root or root not in path.resolve().parents:
        raise ValueError('Checkpoint path escapes its private Drive directory')
    return path


def drive_experiment_dir(config, drive_runs_dir):
    config.validate()
    root = Path(drive_runs_dir).expanduser().absolute()
    if root.resolve() != root:
        raise ValueError('Drive checkpoint root must not traverse symlinks')
    return _checkpoint_drive_path(root / config.mode / config.run_dir.name, root)


def save_drive_checkpoint(config, drive_runs_dir, epoch):
    """Preserve an unstripped optimizer checkpoint before Ultralytics final_eval."""
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError('Checkpoint epoch must be positive')
    run = config.run_dir
    experiment = json.loads(runtime_path(run / 'experiment.json').read_text())
    if not runtime_path(run / 'weights/last.pt').is_file():
        raise FileNotFoundError('No last.pt to back up')
    root = drive_experiment_dir(config, drive_runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_name = f'epoch_{epoch:06d}_{uuid.uuid4().hex[:8]}'
    staging = _checkpoint_drive_path(root / (snapshot_name + '.incomplete'), root)
    final = _checkpoint_drive_path(root / snapshot_name, root)
    staging.mkdir()
    try:
        files = {}
        for relative in CHECKPOINT_FILES:
            source = runtime_path(run / relative)
            if not source.exists():
                continue
            target = _checkpoint_drive_path(staging / relative, root)
            target.parent.mkdir(parents=True, exist_ok=True)
            before = _checkpoint_hash(source)
            shutil.copyfile(source, target)
            if _checkpoint_hash(target) != before:
                raise ValueError('Drive checkpoint copy checksum mismatch')
            files[relative] = {'sha256': before, 'size': target.stat().st_size}
        manifest = {'epoch': epoch, 'specification': experiment['specification'], 'files': files}
        manifest_path = staging / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2))
        manifest_hash = _checkpoint_hash(manifest_path)
        staging.replace(final)
        # The previous latest pointer remains valid until the entire snapshot is ready.
        pointer = _checkpoint_drive_path(root / 'latest.json', root)
        temporary_pointer = _checkpoint_drive_path(root / ('latest.' + uuid.uuid4().hex + '.tmp'), root)
        try:
            temporary_pointer.write_text(json.dumps({'snapshot': snapshot_name, 'epoch': epoch,
                                                     'manifest_sha256': manifest_hash}, indent=2))
            temporary_pointer.replace(pointer)
        finally:
            temporary_pointer.unlink(missing_ok=True)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {'epoch': epoch, 'directory': str(final)}


def restore_drive_checkpoint(config, drive_runs_dir, specification):
    """Restore latest complete, matching snapshot into a fresh Colab run directory."""
    root = drive_experiment_dir(config, drive_runs_dir)
    pointer = json.loads(_checkpoint_drive_path(root / 'latest.json', root).read_text())
    name = pointer.get('snapshot', '')
    if not isinstance(name, str) or not re.fullmatch(r'epoch_[0-9]{6,}_[a-f0-9]{8}', name):
        raise ValueError('Invalid Drive snapshot pointer')
    snapshot = _checkpoint_drive_path(root / name, root)
    manifest_path = _checkpoint_drive_path(snapshot / 'manifest.json', root)
    if _checkpoint_hash(manifest_path) != pointer.get('manifest_sha256'):
        raise ValueError('Drive checkpoint manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('specification') != specification:
        raise ValueError('Drive checkpoint configuration/dataset/library mismatch')
    if manifest.get('epoch') != pointer.get('epoch'):
        raise ValueError('Drive checkpoint epoch mismatch')
    files = manifest.get('files', {})
    if not isinstance(files, dict) or not {'weights/last.pt', 'experiment.json'} <= files.keys() or not files.keys() <= set(CHECKPOINT_FILES):
        raise ValueError('Invalid Drive checkpoint file list')
    run = config.run_dir
    if run.exists() and any(run.iterdir()):
        raise FileExistsError('Restore requires a missing or empty local run directory')
    staging = runtime_path(run.with_name(run.name + '.restore-' + uuid.uuid4().hex))
    staging.mkdir(parents=True)
    try:
        for relative, expected in files.items():
            source = _checkpoint_drive_path(snapshot / relative, root)
            target = runtime_path(staging / relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if target.stat().st_size != expected['size'] or _checkpoint_hash(target) != expected['sha256']:
                raise ValueError('Drive checkpoint file checksum mismatch')
        meta = json.loads((staging / 'experiment.json').read_text())
        if meta.get('specification') != specification:
            raise ValueError('Drive experiment metadata mismatch')
        for session in meta.get('sessions', []):
            if session.get('status') == 'running':
                session['status'] = 'interrupted'
        (staging / 'experiment.json').write_text(json.dumps(meta, indent=2))
        if run.exists():
            run.rmdir()  # Only an empty directory was allowed above.
        staging.replace(run)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return runtime_path(run / 'weights/last.pt')
