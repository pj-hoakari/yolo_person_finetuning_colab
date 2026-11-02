"""Public Hugging Face downloads, local validation and safe extraction."""
import json
import re
import shutil
import stat
import urllib.request
import zipfile
import zlib
from pathlib import Path
from .privacy import BASE, runtime_path

HF_REPO_ID = 'sshao0516/CrowdHuman'
HF_REVISION = 'd97203da87e348ea69f7a7633a57c21a956120a6'
ARCHIVES = ('CrowdHuman_train01.zip', 'CrowdHuman_train02.zip', 'CrowdHuman_train03.zip', 'CrowdHuman_val.zip')
ANNOTATIONS = ('annotation_train.odgt', 'annotation_val.odgt')
EXPECTED = {'train': 15000, 'val': 4370}


def destination(name):
    if name not in ARCHIVES + ANNOTATIONS:
        raise ValueError('Unexpected dataset filename')
    return BASE / 'raw' / ('archives' if name in ARCHIVES else 'annotations') / name


def read_odgt(path):
    """Never include raw annotations or IDs in exception messages."""
    runtime_path(path)
    with open(path, encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
                image_id = record['ID']
                if not isinstance(image_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', image_id):
                    raise ValueError()
                if not isinstance(record['gtboxes'], list):
                    raise ValueError()
            except (ValueError, TypeError, KeyError):
                raise ValueError(f'Invalid ODGT record at line {line_no}') from None
            yield record


def validate_file(path, name):
    runtime_path(path)
    if not Path(path).is_file():
        raise FileNotFoundError(f'Missing required file: {name}')
    if name in ARCHIVES:
        try:
            with zipfile.ZipFile(path) as z:
                if z.testzip() is not None:
                    raise ValueError('ZIP CRC check failed')
                if not z.infolist():
                    raise ValueError('Empty ZIP')
        except zipfile.BadZipFile:
            raise ValueError(f'Invalid ZIP: {name}') from None
        return None
    count = sum(1 for _ in read_odgt(path))
    if count == 0:
        raise ValueError('Empty annotation file')
    return count


def huggingface_links(repo_id=HF_REPO_ID, revision=HF_REVISION):
    """Pin all six original files to one public dataset commit, without Hub upload APIs."""
    if not isinstance(repo_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', repo_id):
        raise ValueError('Invalid Hugging Face dataset repository ID')
    if not isinstance(revision, str) or not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Use a full Hugging Face dataset commit SHA as revision')
    root = f'https://huggingface.co/datasets/{repo_id}/resolve/{revision}'
    return {name: f'{root}/{name}?download=true' for name in ARCHIVES + ANNOTATIONS}


def download_huggingface(repo_id=HF_REPO_ID, revision=HF_REVISION):
    """Stream public resolve URLs straight to /content; no Hub cache, login or uploads."""
    links = huggingface_links(repo_id, revision)
    needed = []
    for name in ARCHIVES + ANNOTATIONS:
        path = runtime_path(destination(name))
        if path.exists():
            validate_file(path, name)  # Corrupt files require explicit removal/replacement.
        else:
            needed.append(name)
    for name in needed:
        path = runtime_path(destination(name))
        temp = runtime_path(path.with_suffix(path.suffix + '.part'))
        try:
            # Standard HTTPS redirects support Hugging Face's public file storage.
            # No stored tokens, cookies, auth headers or gate/limit bypass are used.
            with urllib.request.urlopen(links[name], timeout=120) as response, open(temp, 'wb') as out:
                if 'text/html' in response.headers.get('Content-Type', '').lower():
                    raise RuntimeError('Hugging Face returned an access page instead of a file')
                length = int(response.headers.get('Content-Length', 0))
                if length and shutil.disk_usage(BASE).free < length + 2 * 1024**3:
                    raise OSError('Insufficient download disk space')
                received = 0
                while chunk := response.read(1024**2):
                    if shutil.disk_usage(BASE).free < len(chunk) + 1024**3:
                        raise OSError('Insufficient download disk space')
                    out.write(chunk)
                    received += len(chunk)
                if length and received != length:
                    raise ValueError('Incomplete Hugging Face response')
            validate_file(temp, name)
            temp.replace(path)
        except Exception:
            temp.unlink(missing_ok=True)
            raise RuntimeError(f'Hugging Face download failed: {name}; use lawful manual upload if access is unavailable') from None
    source = runtime_path(BASE / 'raw/download_source.json')
    source.write_text(json.dumps({'requested_repo_id': repo_id, 'requested_revision': revision,
                                  'downloaded': needed,
                                  'reused_existing': [name for name in ARCHIVES + ANNOTATIONS if name not in needed]}, indent=2))


def upload_missing():
    """Colab upload widget writes files directly to raw; it never prints file contents."""
    from google.colab import files
    import os
    previous = Path.cwd()
    try:
        os.chdir(runtime_path(BASE / 'raw'))
        uploaded = files.upload()
        del uploaded  # Browser upload may use substantial RAM for large archives.
        for name in ARCHIVES + ANNOTATIONS:
            staged = BASE / 'raw' / name
            if staged.exists():
                target = destination(name)
                if target.exists():
                    validate_file(target, name)
                    staged.unlink()
                else:
                    validate_file(staged, name)
                    staged.replace(target)
    finally:
        os.chdir(previous)


def inventory():
    counts = {}
    for name in ARCHIVES + ANNOTATIONS:
        count = validate_file(destination(name), name)
        if count is not None:
            counts[name] = count
    for split in EXPECTED:
        if counts[f'annotation_{split}.odgt'] != EXPECTED[split]:
            raise ValueError('Official annotation count mismatch')
    return {'annotation_counts': counts, 'free_gib': round(shutil.disk_usage(BASE).free / 1024**3, 2)}


def extract_archives():
    root = runtime_path(BASE / 'raw/images')
    entries = []
    seen = set()
    for name in ARCHIVES:
        archive = destination(name)
        validate_file(archive, name)
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if item.is_dir():
                    continue
                parts = Path(item.filename).parts
                if len(parts) != 2 or parts[0] != 'Images' or Path(parts[1]).suffix.lower() not in ('.jpg', '.jpeg'):
                    raise ValueError('Unexpected archive member')
                if stat.S_ISLNK(item.external_attr >> 16) or parts[1] in seen:
                    raise ValueError('Duplicate/unsafe archive member')
                seen.add(parts[1])
                target = runtime_path(root / parts[1])
                entries.append((archive, item.filename, target, item.file_size, item.CRC))
    required = sum(size for _, _, p, size, _ in entries if not p.exists() or p.stat().st_size != size)
    if shutil.disk_usage(BASE).free < required + 2 * 1024**3:
        raise OSError('Insufficient extraction space (plus 2 GiB reserve)')
    # Compare CRCs before reuse; a same-size corrupt image must also be replaced.
    for archive in map(destination, ARCHIVES):
        with zipfile.ZipFile(archive) as z:
            for source_archive, member, target, size, crc in entries:
                if source_archive != archive:
                    continue
                if target.exists() and target.stat().st_size == size:
                    actual = 0
                    with open(target, 'rb') as existing:
                        for chunk in iter(lambda: existing.read(1024**2), b''):
                            actual = zlib.crc32(chunk, actual)
                    if actual == crc:
                        continue
                part = target.with_suffix('.part')
                with z.open(member) as source, open(part, 'wb') as out:
                    shutil.copyfileobj(source, out)
                part.replace(target)
    return {'archive_images': len(entries)}


def image_size(path):
    runtime_path(path)
    from PIL import Image
    try:
        with Image.open(path) as image:
            size = image.size
            image.verify()
        if min(size) <= 0:
            raise ValueError()
        return size
    except Exception:
        raise ValueError('Missing or corrupt dataset image') from None


def inspect_dataset():
    """Check every image, including DEBUG runs; reject split overlap and duplicates."""
    splits = {}
    sizes = {}
    for split in ('train', 'val'):
        ids = set()
        for record in read_odgt(destination(f'annotation_{split}.odgt')):
            key = record['ID']
            if key in ids:
                raise ValueError('Duplicate image ID within split')
            ids.add(key)
            sizes[key] = image_size(BASE / 'raw/images' / f'{key}.jpg')
        splits[split] = ids
    if splits['train'] & splits['val']:
        raise ValueError('Train/val overlap detected')
    return sizes, {split: len(ids) for split, ids in splits.items()}
