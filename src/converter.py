"""Separate body/head ignore policy and streaming YOLO label conversion."""
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from .privacy import BASE, runtime_path
from .dataset import destination, read_odgt

NAMES = {'vbox': {0: 'person'}, 'hbox': {0: 'head'}, 'both': {0: 'person', 1: 'head'}}


def normalize_box(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError('Invalid bbox')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise ValueError('Invalid bbox')
    x, y, w, h = box
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        raise ValueError('Nonpositive dimensions')
    x1, y1 = max(0, min(width, x)), max(0, min(height, y))
    x2, y2 = max(0, min(width, x + w)), max(0, min(height, y + h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Empty clipped bbox')
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)


def ignored(annotation, attr):
    value = annotation.get(attr, {})
    if not isinstance(value, dict) or value.get('ignore', 0) not in (0, 1):
        raise ValueError('Invalid ignore attribute')
    return value.get('ignore', 0) == 1


def convert_boxes(boxes, mode, width, height):
    if mode not in NAMES:
        raise ValueError('Unsupported mode')
    labels, stats = [], Counter()
    fields = ('vbox', 'hbox') if mode == 'both' else (mode,)
    for annotation in boxes:
        if not isinstance(annotation, dict):
            stats['invalid_annotation'] += 1
            continue
        if annotation.get('tag') != 'person':
            stats['non_person'] += 1
            continue
        for field in fields:
            try:
                if ignored(annotation, 'extra' if field == 'vbox' else 'head_attr'):
                    stats[f'{field}_ignored'] += 1
                    continue
                box = annotation.get(field)
                normalized = normalize_box(box, width, height)
                class_id = 1 if mode == 'both' and field == 'hbox' else 0
                labels.append((class_id, *normalized))
                stats[f'{field}_accepted'] += 1
                x, y, w, h = box
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    stats[f'{field}_clipped'] += 1
            except (ValueError, TypeError):
                stats[f'{field}_invalid'] += 1
    return labels, stats


def fingerprint(sizes, limit):
    digest = hashlib.sha256()
    for split in ('train', 'val'):
        with open(destination(f'annotation_{split}.odgt'), 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024**2), b''):
                digest.update(chunk)
    digest.update(json.dumps(sorted(sizes.items())).encode())
    digest.update(str(limit).encode())
    return digest.hexdigest()


def convert_dataset(mode, sizes, limit=None):
    if mode not in NAMES or (limit is not None and limit <= 0):
        raise ValueError('Invalid conversion configuration')
    root = runtime_path(BASE / 'processed' / mode)
    signature = fingerprint(sizes, limit)
    manifest = root / 'conversion.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['fingerprint'] == signature:
            validate_labels(root, mode)
            return root / 'data.yaml'
        raise ValueError('Dataset changed; remove processed mode and matching run explicitly')
    staging = root.with_name(mode + '.building')
    if staging.exists():
        shutil.rmtree(runtime_path(staging))
    counts = Counter()
    selected = set()
    try:
        for split in ('train', 'val'):
            images = staging / 'images' / split
            labels_dir = staging / 'labels' / split
            images.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            for index, record in enumerate(read_odgt(destination(f'annotation_{split}.odgt'))):
                if limit is not None and index >= limit:
                    break
                key = record['ID']
                if key in selected:
                    raise ValueError('Duplicate image or split overlap')
                selected.add(key)
                width, height = sizes[key]
                source = runtime_path(BASE / 'raw/images' / f'{key}.jpg')
                if not source.is_file():
                    raise FileNotFoundError('Missing source image')
                rows, stats = convert_boxes(record['gtboxes'], mode, width, height)
                counts.update({f'{split}_{k}': v for k, v in stats.items()})
                counts[f'{split}_images'] += 1
                (images / source.name).symlink_to(source)
                (labels_dir / f'{key}.txt').write_text(''.join(' '.join([str(row[0]), *(f'{v:.10f}' for v in row[1:])]) + '\n' for row in rows))
        if not counts['train_images'] or not counts['val_images']:
            raise ValueError('Both splits must be nonempty')
        (staging / 'conversion.json').write_text(json.dumps({'fingerprint': signature, 'limit': limit, 'counts': counts}, indent=2))
        if root.exists():
            raise ValueError('Incomplete processed directory; remove it explicitly')
        staging.rename(root)
        write_yaml(root, mode)
        validate_labels(root, mode)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return root / 'data.yaml'


def write_yaml(root, mode):
    import yaml
    root = runtime_path(root)
    data = {'path': str(root), 'train': 'images/train', 'val': 'images/val', 'names': NAMES[mode]}
    (root / 'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False))


def validate_labels(root, mode):
    import yaml
    root = runtime_path(root)
    data = yaml.safe_load((root / 'data.yaml').read_text())
    if data != {'path': str(root), 'train': 'images/train', 'val': 'images/val', 'names': NAMES[mode]}:
        raise ValueError('data.yaml mismatch')
    ids = []
    totals = {}
    for split in ('train', 'val'):
        images = list((root / 'images' / split).glob('*.jpg'))
        label_files = list((root / 'labels' / split).glob('*.txt'))
        if not images or {p.stem for p in images} != {p.stem for p in label_files}:
            raise ValueError('Images/labels mismatch')
        boxes = 0
        ids.append({p.stem for p in images})
        for image in images:
            runtime_path(image)
            if not image.is_file():
                raise FileNotFoundError('Broken image symlink')
        for path in label_files:
            runtime_path(path)
            for line in path.read_text().splitlines():
                fields = line.split()
                if len(fields) != 5 or fields[0] not in map(str, NAMES[mode]):
                    raise ValueError('Invalid class or label columns')
                values = list(map(float, fields[1:]))
                x, y, w, h = values
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or min(w, h) <= 0:
                    raise ValueError('Invalid normalized label')
                if x-w/2 < -1e-8 or x+w/2 > 1+1e-8 or y-h/2 < -1e-8 or y+h/2 > 1+1e-8:
                    raise ValueError('Label crosses image boundary')
                boxes += 1
        if not boxes:
            raise ValueError('Split has no usable labels')
        totals[split] = {'images': len(images), 'boxes': boxes}
    if ids[0] & ids[1]:
        raise ValueError('Split overlap')
    return totals
