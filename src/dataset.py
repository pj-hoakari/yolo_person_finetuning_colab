"""Public Hugging Face downloads, local validation and safe extraction."""
import json
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
import re
import shutil
import stat
import urllib.request
import urllib.error
from urllib.parse import urlsplit
import zipfile
import zlib
from pathlib import Path
from .privacy import BASE, runtime_path, storage_root

HF_REPO_ID = 'sshao0516/CrowdHuman'
HF_REVISION = 'd97203da87e348ea69f7a7633a57c21a956120a6'
ARCHIVES = ('CrowdHuman_train01.zip', 'CrowdHuman_train02.zip', 'CrowdHuman_train03.zip', 'CrowdHuman_val.zip')
ANNOTATIONS = ('annotation_train.odgt', 'annotation_val.odgt')
EXPECTED = {'train': 15000, 'val': 4370}


class DownloadProgress:
    """Throttled text progress for a terminal or one updating notebook display."""
    def __init__(self, stream=None, interval=1.0, notebook=False):
        self.stream = sys.stderr if stream is None else stream
        self.interval = interval
        self.notebook = notebook
        self.last_update = None
        self.task = None
        self.started = None
        self.display = None

    def __call__(self, event):
        now = time.monotonic()
        stage = event['stage']
        task = (stage, event.get('filename'))
        changed = task != self.task
        if changed:
            self.task, self.started = task, now
        finished = event.get('finished', False) or stage in ('reused', 'file_complete', 'complete', 'error', 'interrupted')
        if not changed and not finished and self.last_update is not None and now - self.last_update < self.interval:
            return
        self.last_update = now
        done = event.get('processed', 0)
        total = event.get('total', 0)
        prefix = f"[{event.get('completed_files', 0)}/{event.get('file_total', 6)} 完了]"
        filename = event.get('filename', '')
        labels = {'request': '接続中', 'validation': '検証中', 'transfer': 'ダウンロード中',
                  'reused': '検証OK・再利用', 'file_complete': '取得・検証完了',
                  'complete': '全ファイル準備完了', 'error': '失敗', 'interrupted': '中断'}
        text = f"{prefix} {labels.get(stage, stage)} {filename}"
        if stage in ('transfer', 'validation'):
            percent = f'{100 * done / total:5.1f}%' if total else '  -- %'
            elapsed = max(0.0, now - self.started)
            speed = done / elapsed if elapsed else 0
            remaining = max(0, total - done) / speed if total and speed else None
            eta = f'{int(remaining)//60:02d}:{int(remaining)%60:02d}' if remaining is not None else '--:--'
            if event.get('unit') == 'records':
                amount = f'{done:,}/{total:,} 件' if total else f'{done:,} 件'
                rate = f'{speed:,.0f} 件/s' if speed else '-- 件/s'
            else:
                divisor, unit = (1024**3, 'GiB') if max(done, total) >= 1024**3 else (1024**2, 'MiB')
                amount = f'{done/divisor:.1f}/{total/divisor:.1f} {unit}' if total else f'{done/divisor:.1f} {unit}'
                rate = f'{speed/1024**2:.1f} MiB/s' if speed else '-- MiB/s'
            text += f' | {percent} | {amount} | {rate} | 残り {eta}'
        if self.notebook:
            from html import escape
            from IPython.display import HTML, display
            content = HTML('<pre>' + escape(text) + '</pre>')
            if self.display is None:
                self.display = display(content, display_id=True)
            else:
                self.display.update(content)
        elif self.stream.isatty():
            print('\r' + text + '\033[K', end='\n' if finished else '', file=self.stream, flush=True)
        else:
            print(text, file=self.stream, flush=True)


def _notify_progress(progress, **event):
    if progress is not None:
        progress(event)


def _file_progress(progress, index, completed, total_files):
    if progress is None:
        return None
    return lambda event: progress({**event, 'file_index': index, 'completed_files': completed,
                                   'file_total': total_files})


def destination(name):
    if name not in ARCHIVES + ANNOTATIONS:
        raise ValueError('Unexpected dataset filename')
    return storage_root() / 'raw' / ('archives' if name in ARCHIVES else 'annotations') / name


def read_odgt(path):
    """Never include raw annotations or IDs in exception messages."""
    runtime_path(path)
    with open(path, encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                raise ValueError(f'Invalid ODGT record at line {line_no}: invalid JSON') from None
            if not isinstance(record, dict):
                raise ValueError(f'Invalid ODGT record at line {line_no}: expected a JSON object')
            image_id = record.get('ID')
            # Original CrowdHuman image stems contain a comma. Keep separators,
            # dots, whitespace and control characters excluded from filesystem IDs.
            if not isinstance(image_id, str) or not re.fullmatch(r'[A-Za-z0-9_,\-]+', image_id):
                raise ValueError(f'Invalid ODGT record at line {line_no}: invalid image ID')
            if not isinstance(record.get('gtboxes'), list):
                raise ValueError(f'Invalid ODGT record at line {line_no}: gtboxes must be a list')
            yield record


def validate_file(path, name, progress=None):
    runtime_path(path)
    if not Path(path).is_file():
        raise FileNotFoundError(f'Missing required file: {name}')
    if name in ARCHIVES:
        try:
            with zipfile.ZipFile(path) as z:
                entries = z.infolist()
                if not entries:
                    raise ValueError('Empty ZIP')
                if progress is None:
                    if z.testzip() is not None:
                        raise ValueError('ZIP CRC check failed')
                else:
                    total = sum(entry.file_size for entry in entries if not entry.is_dir())
                    processed = 0
                    _notify_progress(progress, stage='validation', filename=name, processed=0, total=total, unit='bytes')
                    for entry in entries:
                        if entry.is_dir():
                            continue
                        with z.open(entry) as handle:
                            while chunk := handle.read(1024**2):
                                processed += len(chunk)
                                _notify_progress(progress, stage='validation', filename=name,
                                                 processed=processed, total=total, unit='bytes')
                    _notify_progress(progress, stage='validation', filename=name, processed=processed,
                                     total=total, unit='bytes', finished=True)
        except zipfile.BadZipFile:
            raise ValueError(f'Invalid ZIP: {name}') from None
        return None
    count = 0
    expected = EXPECTED['train' if name == 'annotation_train.odgt' else 'val']
    _notify_progress(progress, stage='validation', filename=name, processed=0, total=expected, unit='records')
    for _ in read_odgt(path):
        count += 1
        if count % 100 == 0:
            _notify_progress(progress, stage='validation', filename=name, processed=count, total=expected, unit='records')
    if count == 0:
        raise ValueError('Empty annotation file')
    _notify_progress(progress, stage='validation', filename=name, processed=count, total=count, unit='records', finished=True)
    return count


def huggingface_links(repo_id=HF_REPO_ID, revision=HF_REVISION):
    """Pin all six original files to one public dataset commit, without Hub upload APIs."""
    if not isinstance(repo_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', repo_id):
        raise ValueError('Invalid Hugging Face dataset repository ID')
    if not isinstance(revision, str) or not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Use a full Hugging Face dataset commit SHA as revision')
    root = f'https://huggingface.co/datasets/{repo_id}/resolve/{revision}'
    return {name: f'{root}/{name}?download=true' for name in ARCHIVES + ANNOTATIONS}


def _safe_download_message(message):
    """Never retain signed URLs, credentials, response bodies or annotation content."""
    text = re.sub(r'https?://[^\s\"\'<>]+', '[URL removed]', str(message))
    text = re.sub(r'(?i)\b(?:hf_[A-Za-z0-9]+|Bearer\s+[^\s,;]+)', '[credential removed]', text)
    text = re.sub(r'(?i)((?:token|password|authorization|signature|credential|api[_-]?key)\s*[:=]\s*)[^\s,;]+', r'\1[redacted]', text)
    return text[:1000]


def _download_event(log_path, context, event, **details):
    path = runtime_path(log_path)
    runtime_path(path.parent).mkdir(parents=True, exist_ok=True)
    entry = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
             'event': event, **context, **details}
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + '\n')


def _download_error_details(error):
    chain, seen = [], set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        item = {'type': type(current).__name__, 'message': _safe_download_message(current)}
        if isinstance(current, urllib.error.HTTPError):
            item['http_status'] = current.code
            item['reason'] = _safe_download_message(current.reason)
            if current.headers:
                item['response_headers'] = {
                    name: _safe_download_message(current.headers[name])
                    for name in ('X-Request-Id', 'Retry-After') if name in current.headers}
        elif isinstance(current, urllib.error.URLError):
            item['reason'] = _safe_download_message(current.reason)
        if isinstance(current, OSError) and current.errno is not None:
            item['errno'] = current.errno
        # Stack locations only: no source lines, locals, response payloads or data IDs.
        item['stack'] = [{'file': Path(frame.filename).name, 'line': frame.lineno,
                          'function': frame.name}
                         for frame in traceback.extract_tb(current.__traceback__)]
        chain.append(item)
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return chain


def download_huggingface(repo_id=HF_REPO_ID, revision=HF_REVISION, progress=None):
    """Download public files and retain sanitized diagnostics inside the runtime."""
    log_path = runtime_path(storage_root() / 'logs/download.jsonl')
    context = {'attempt_id': uuid.uuid4().hex, 'stage': 'configuration'}
    temp = None
    completed = 0
    total_files = len(ARCHIVES + ANNOTATIONS)
    try:
        links = huggingface_links(repo_id, revision)
        context.update(repo_id=repo_id, revision=revision)
        _download_event(log_path, context, 'start')
        needed = []
        for index, name in enumerate(ARCHIVES + ANNOTATIONS, 1):
            context.update(filename=name, stage='existing_file_validation')
            path = runtime_path(destination(name))
            if path.exists():
                validate_file(path, name, progress=_file_progress(progress, index, completed, total_files))
                completed += 1
                _notify_progress(progress, stage='reused', filename=name, completed_files=completed, file_total=total_files)
                _download_event(log_path, context, 'reused', bytes_on_disk=path.stat().st_size)
            else:
                needed.append(name)
        for name in needed:
            context = {'attempt_id': context['attempt_id'], 'repo_id': repo_id,
                       'revision': revision, 'filename': name, 'stage': 'http_request',
                       'bytes_received': 0}
            path = runtime_path(destination(name))
            temp = runtime_path(path.with_suffix(path.suffix + '.part'))
            index = (ARCHIVES + ANNOTATIONS).index(name) + 1
            current_progress = _file_progress(progress, index, completed, total_files)
            _notify_progress(current_progress, stage='request', filename=name)
            _download_event(log_path, context, 'file_start')
            # No auth headers, cookies, response bodies or signed redirect URLs are logged.
            with urllib.request.urlopen(links[name], timeout=120) as response, open(temp, 'wb') as out:
                context['http_status'] = getattr(response, 'status', None)
                context['response_host'] = urlsplit(response.geturl()).hostname if hasattr(response, 'geturl') else 'huggingface.co'
                content_type = response.headers.get('Content-Type', '')
                context['content_type'] = _safe_download_message(content_type)
                if 'text/html' in content_type.lower():
                    raise RuntimeError('Hugging Face returned an access page instead of a file')
                length = int(response.headers.get('Content-Length', 0))
                context.update(expected_bytes=length, stage='disk_space_check',
                               free_bytes=shutil.disk_usage(storage_root()).free)
                _download_event(log_path, context, 'response')
                if length and context['free_bytes'] < length + 2 * 1024**3:
                    raise OSError('Insufficient download disk space')
                context['stage'] = 'transfer'
                last_progress = 0
                _notify_progress(current_progress, stage='transfer', filename=name, processed=0, total=length, unit='bytes')
                while chunk := response.read(1024**2):
                    context['free_bytes'] = shutil.disk_usage(storage_root()).free
                    if context['free_bytes'] < len(chunk) + 1024**3:
                        raise OSError('Insufficient download disk space')
                    out.write(chunk)
                    context['bytes_received'] += len(chunk)
                    _notify_progress(current_progress, stage='transfer', filename=name,
                                     processed=context['bytes_received'], total=length, unit='bytes')
                    if context['bytes_received'] - last_progress >= 64 * 1024**2:
                        _download_event(log_path, context, 'progress')
                        last_progress = context['bytes_received']
                if length and context['bytes_received'] != length:
                    raise ValueError('Incomplete Hugging Face response')
            _notify_progress(current_progress, stage='transfer', filename=name,
                             processed=context['bytes_received'], total=length, unit='bytes', finished=True)
            context['stage'] = 'downloaded_file_validation'
            validate_file(temp, name, progress=current_progress)
            temp.replace(path)
            temp = None
            completed += 1
            _notify_progress(progress, stage='file_complete', filename=name, completed_files=completed, file_total=total_files)
            _download_event(log_path, context, 'file_complete')
        context['stage'] = 'save_source_metadata'
        source = runtime_path(storage_root() / 'raw/download_source.json')
        source.write_text(json.dumps({'requested_repo_id': repo_id, 'requested_revision': revision,
                                      'downloaded': needed,
                                      'reused_existing': [name for name in ARCHIVES + ANNOTATIONS if name not in needed]}, indent=2))
        _download_event(log_path, context, 'complete', downloaded_files=len(needed),
                        reused_files=len(ARCHIVES + ANNOTATIONS) - len(needed))
        _notify_progress(progress, stage='complete', completed_files=completed, file_total=total_files)
        return log_path
    except (Exception, KeyboardInterrupt) as error:
        cleanup_error = None
        if temp is not None:
            try:
                temp.unlink(missing_ok=True)
            except OSError as cleanup:
                cleanup_error = _safe_download_message(cleanup)
        log_note = f'Log: {log_path}'
        try:
            _download_event(log_path, context, 'interrupted' if isinstance(error, KeyboardInterrupt) else 'error',
                            errors=_download_error_details(error), cleanup_error=cleanup_error)
        except OSError as log_error:
            log_note = f'Could not write log {log_path}: {_safe_download_message(log_error)}'
        _notify_progress(progress, stage='interrupted' if isinstance(error, KeyboardInterrupt) else 'error',
                         filename=context.get('filename', ''), completed_files=completed, file_total=total_files)
        if isinstance(error, KeyboardInterrupt):
            raise
        raise RuntimeError(f'Hugging Face download failed: {context.get("filename", "configuration")}; '
                           f'{type(error).__name__}: {_safe_download_message(error)}. '
                           + log_note) from None


def upload_missing():
    """Colab upload widget writes files directly to raw; it never prints file contents."""
    from google.colab import files
    import os
    previous = Path.cwd()
    try:
        os.chdir(runtime_path(storage_root() / 'raw'))
        uploaded = files.upload()
        del uploaded  # Browser upload may use substantial RAM for large archives.
        for name in ARCHIVES + ANNOTATIONS:
            staged = storage_root() / 'raw' / name
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


def inventory(progress=None):
    counts = {}
    files = ARCHIVES + ANNOTATIONS
    for index, name in enumerate(files, 1):
        count = validate_file(destination(name), name, progress=_file_progress(progress, index, index - 1, len(files)))
        if count is not None:
            counts[name] = count
    for split in EXPECTED:
        if counts[f'annotation_{split}.odgt'] != EXPECTED[split]:
            raise ValueError('Official annotation count mismatch')
    _notify_progress(progress, stage='complete', completed_files=len(files), file_total=len(files))
    return {'annotation_counts': counts, 'free_gib': round(shutil.disk_usage(storage_root()).free / 1024**3, 2)}


def extract_archives():
    root = runtime_path(storage_root() / 'raw/images')
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
    if shutil.disk_usage(storage_root()).free < required + 2 * 1024**3:
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
            sizes[key] = image_size(storage_root() / 'raw/images' / f'{key}.jpg')
        splits[split] = ids
    if splits['train'] & splits['val']:
        raise ValueError('Train/val overlap detected')
    return sizes, {split: len(ids) for split, ids in splits.items()}
