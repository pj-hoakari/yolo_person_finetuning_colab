"""Save prepared private data once; restore without downloading or converting."""
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from .privacy import BASE, runtime_path


def _prepared_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_root(directory):
    root = Path(directory).absolute()
    if root.resolve() != root:
        raise ValueError('Prepared data directory must not traverse symlinks')
    return root


def _prepared_parts(settings):
    modes = settings['modes']
    if not modes or len(set(modes)) != len(modes) or any(m not in ('vbox', 'hbox', 'both') for m in modes):
        raise ValueError('Invalid prepared modes')
    return ['raw/images', 'raw/annotations'] + ['processed/' + mode for mode in modes]


def save_prepared_data(directory, settings):
    """Publish latest.json only after a complete archive has been verified."""
    root = _prepared_root(directory)
    parts = _prepared_parts(settings)
    root.mkdir(parents=True, exist_ok=True)
    name = 'prepared_' + uuid.uuid4().hex + '.tar'
    archive = root / name
    temporary = root / (name + '.incomplete')
    try:
        with tarfile.open(temporary, 'w') as tar:
            for part in parts:
                source = runtime_path(BASE / part)
                if not source.is_dir():
                    raise FileNotFoundError('Prepared data is missing: ' + part)
                for path in [source, *sorted(source.rglob('*'))]:
                    relative = path.relative_to(BASE).as_posix()
                    runtime_path(path)
                    info = tar.gettarinfo(str(path), arcname=relative)
                    if info.issym():
                        target = path.resolve()
                        image_root = BASE / 'raw/images'
                        if image_root not in target.parents or not target.is_file():
                            raise ValueError('Prepared link must point to an image')
                        info.linkname = os.path.relpath(target, path.parent)
                        tar.addfile(info)
                    elif info.isfile():
                        with open(path, 'rb') as handle:
                            tar.addfile(info, handle)
                    elif info.isdir():
                        tar.addfile(info)
                    else:
                        raise ValueError('Unsupported prepared file')
        checksum = _prepared_hash(temporary)
        temporary.rename(archive)
        manifest = {'schema': 1, 'archive': name, 'sha256': checksum,
                    'size': archive.stat().st_size, 'settings': settings}
        pointer = root / ('latest.' + uuid.uuid4().hex + '.tmp')
        pointer.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        pointer.replace(root / 'latest.json')
        return archive
    finally:
        temporary.unlink(missing_ok=True)


def restore_prepared_data(directory, settings):
    """Verify paths and checksum, extract in staging, then install into empty dirs."""
    root = _prepared_root(directory)
    parts = _prepared_parts(settings)
    manifest = json.loads((root / 'latest.json').read_text())
    if manifest.get('schema') != 1 or manifest.get('settings') != settings:
        raise ValueError('Prepared data settings mismatch; use the saved modes/DEBUG settings')
    name = manifest['archive']
    if not isinstance(name, str) or not name.startswith('prepared_') or not name.endswith('.tar') or Path(name).name != name:
        raise ValueError('Invalid prepared archive name')
    archive = _prepared_root(root / name)
    if archive.stat().st_size != manifest['size'] or _prepared_hash(archive) != manifest['sha256']:
        raise ValueError('Prepared archive checksum mismatch')
    for part in parts:
        target = runtime_path(BASE / part)
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError('Restore requires an empty local directory: ' + part)
    with tarfile.open(archive, 'r:') as tar:
        members = tar.getmembers()
        seen, links = set(), set()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.name in seen:
                raise ValueError('Unsafe prepared archive path')
            if not any(member.name == part or member.name.startswith(part + '/') for part in parts):
                raise ValueError('Unexpected prepared archive path')
            seen.add(member.name)
            if member.issym():
                target = (BASE / member.name).parent / member.linkname
                resolved = target.resolve()
                if (BASE / 'raw/images') not in resolved.parents or Path(member.linkname).is_absolute():
                    raise ValueError('Unsafe prepared archive link')
                links.add(member.name)
            elif not (member.isfile() or member.isdir()):
                raise ValueError('Unsupported prepared archive entry')
        for member in members:
            if any(parent.as_posix() in links for parent in PurePosixPath(member.name).parents):
                raise ValueError('Prepared archive traverses a link')
        required = sum(member.size for member in members if member.isfile())
        if shutil.disk_usage(BASE).free < required + 1024**2:
            raise OSError('Insufficient local space for prepared data')
        with tempfile.TemporaryDirectory(prefix='.prepared-', dir=BASE) as temporary:
            stage = Path(temporary)
            # Extract regular entries first; links are created only after all writes.
            for member in members:
                target = stage / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.isdir():
                    target.mkdir(exist_ok=True)
                elif member.isfile():
                    with tar.extractfile(member) as source, open(target, 'wb') as output:
                        shutil.copyfileobj(source, output, 8 * 1024**2)
            for member in members:
                if member.issym():
                    (stage / member.name).symlink_to(member.linkname)
            for part in parts:
                if not (stage / part).is_dir():
                    raise ValueError('Prepared archive is incomplete')
            installed = []
            try:
                for part in parts:
                    target = runtime_path(BASE / part)
                    if target.exists():
                        target.rmdir()
                    (stage / part).rename(target)
                    installed.append(target)
            except Exception:
                for target in reversed(installed):
                    target.rename(stage / target.relative_to(BASE))
                raise
    return {mode: BASE / 'processed' / mode / 'data.yaml' for mode in settings['modes']}
