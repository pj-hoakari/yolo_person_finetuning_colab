"""Scalar-only evaluation, comparable CSVs and explicitly selected local exports."""
from dataclasses import asdict
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from .privacy import BASE, private_yolo, runtime_path, local_console
from .trainer import device_for


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(config):
    run = config.run_dir
    best = runtime_path(run / 'weights/best.pt')
    if not best.is_file():
        raise FileNotFoundError('No trained best.pt')
    experiment = json.loads((run / 'experiment.json').read_text())
    if experiment['specification']['config'] != asdict(config):
        raise ValueError('Evaluation must use the saved training configuration')
    from .converter import validate_labels
    root = runtime_path(BASE / 'processed' / config.mode)
    validate_labels(root, config.mode)
    conversion = json.loads((root / 'conversion.json').read_text())
    if conversion['fingerprint'] != experiment['specification']['dataset_fingerprint']:
        raise ValueError('Evaluation dataset differs from the training dataset')
    YOLO = private_yolo()
    with local_console(run / 'evaluation.log'):
        model = YOLO(str(best))
        metrics = model.val(data=str(BASE / 'processed' / config.mode / 'data.yaml'),
                            split='val', imgsz=config.imgsz, batch=1, device=device_for(config),
                            workers=config.workers, seed=config.seed, conf=0.001, iou=0.7,
                            max_det=1000, half=False, plots=False, save=False,
                            save_json=False, save_txt=False, verbose=False,
                            project=str(run), name='validation', exist_ok=True)
    import torch
    identity = {'mode': config.mode, 'run_name': run.name, 'weight_sha256': sha256(best)}
    common = dict(precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                  map50=float(metrics.box.map50), map50_95=float(metrics.box.map),
                  training_seconds=sum(s['seconds'] for s in experiment['sessions']),
                  inference_ms_per_image=float(metrics.speed['inference']),
                  model_mib=best.stat().st_size/1024**2,
                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    record = {**identity, **common, 'specification': experiment['specification'],
              'evaluation': {'batch': 1, 'imgsz': config.imgsz, 'half': False, 'conf': 0.001, 'iou': 0.7, 'max_det': 1000},
              'classes': []}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(index)
        record['classes'].append({'class_id': int(class_id), 'name': model.names[int(class_id)],
                                  'precision': float(p), 'recall': float(r),
                                  'map50': float(ap50), 'map50_95': float(ap)})
    (run / 'evaluation.json').write_text(json.dumps(record, indent=2))
    return {**identity, **common}


def comparison(output=BASE / 'exports/comparison.csv', configs=None):
    rows, signatures = [], []
    paths = ((BASE / 'runs').glob('*/*/evaluation.json') if configs is None
             else (config.run_dir / 'evaluation.json' for config in configs))
    for path in sorted(paths):
        runtime_path(path)
        row = json.loads(path.read_text())
        best = runtime_path(path.parent / 'weights/best.pt')
        if not best.is_file() or sha256(best) != row['weight_sha256']:
            raise ValueError('Stale evaluation: evaluate the current checkpoint again')
        spec = json.loads(json.dumps(row['specification']))
        for key in ('mode', 'run_name'):
            spec['config'].pop(key)
        signatures.append(json.dumps({'spec': spec, 'device': row['device'], 'evaluation': row['evaluation']}, sort_keys=True))
        scalar = {k: v for k, v in row.items() if k not in ('specification', 'evaluation', 'classes')}
        rows.append({**scalar, 'scope': 'all', 'class_name': 'all'})
        for cls in row['classes']:
            rows.append({**scalar, **{k: cls[k] for k in ('precision', 'recall', 'map50', 'map50_95')},
                         'scope': 'class', 'class_name': cls['name']})
    if not rows:
        raise FileNotFoundError('No evaluation results; train and evaluate at least one mode')
    comparable = len(set(signatures)) == 1
    for row in rows:
        row['matched_conditions'] = comparable
    output = runtime_path(output)
    with open(output, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def training_history(config):
    """Summarize the CSV by actual epoch numbers, ignoring non-finite metrics."""
    with open(runtime_path(config.run_dir / 'results.csv'), newline='', encoding='utf-8') as handle:
        rows = [{key.strip(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError('Training CSV contains no epochs')
    metric = next((key for key in rows[0] if 'mAP50-95' in key and '(B)' in key), None)
    best = None
    for row in rows:
        if metric is not None:
            try:
                value = float(row[metric])
            except (ValueError, TypeError):
                continue
            if math.isfinite(value) and (best is None or value > float(best[metric])):
                best = row
    summary = {'epochs_run': len(rows), 'last_epoch': int(float(rows[-1]['epoch'])),
               'best_epoch_by_map50_95': int(float(best['epoch'])) if best else None,
               'best_map50_95': float(best[metric]) if best else None,
               'last_metrics': rows[-1]}
    return summary, rows


def plot_training_history(config):
    """Display scalar curves only; never read or display dataset images."""
    import matplotlib.pyplot as plt
    _, rows = training_history(config)
    epochs = [float(row['epoch']) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key in rows[0]:
        axis = axes[0] if 'loss' in key else axes[1] if 'mAP' in key else None
        if axis is not None:
            axis.plot(epochs, [float(row[key]) for row in rows], label=key)
    for axis, title in zip(axes, ('Loss', 'mAP')):
        axis.set(title=title, xlabel='Epoch')
        axis.grid(alpha=.3)
        if axis.lines:
            axis.legend(fontsize=8)
    figure.suptitle(f'{config.mode} / {config.run_dir.name}')
    figure.tight_layout()
    try:
        from IPython.display import display
        display(figure)
    finally:
        plt.close(figure)


def available_exports(configs, include_onnx=True):
    names = []
    for config in configs:
        prefix = f'{config.mode}_{config.run_dir.name}'
        candidates = [prefix + '.pt', prefix + '_training.csv']
        if include_onnx:
            candidates.insert(1, prefix + '.onnx')
        names.extend(name for name in candidates if runtime_path(BASE / 'exports' / name).is_file())
    if runtime_path(BASE / 'exports/comparison.csv').is_file():
        names.append('comparison.csv')
    return names


def inspect_onnx(path, image_size, contract=None):
    """Run synthetic input only. A supplied application contract is required for matching."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    path = runtime_path(path)
    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].type != 'tensor(float)' or len(inputs[0].shape) != 4:
        raise ValueError('Unexpected ONNX image input')
    tensor = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
    outputs = session.run(None, {inputs[0].name: tensor})
    if not outputs or any(not np.isfinite(value).all() for value in outputs):
        raise ValueError('Invalid ONNX synthetic inference output')
    interface = {'input': {'name': inputs[0].name, 'dtype': inputs[0].type, 'shape': list(tensor.shape)},
                 'outputs': [{'name': meta.name, 'dtype': meta.type, 'shape': list(value.shape)}
                             for meta, value in zip(session.get_outputs(), outputs)]}
    if contract is not None and interface != contract:
        raise ValueError('Web application input/output contract mismatch')
    return {'interface': interface, 'onnx_checker': 'passed', 'synthetic_inference': 'passed',
            'application_contract': 'matched' if contract is not None else 'unverified',
            'browser_integration': 'unverified',
            'preprocessing': 'RGB float32 NCHW /255; letterbox and inverse coordinate mapping required',
            'postprocessing': 'Confirm YOLO26 end-to-end output vs raw detections before connecting tracker'}


def export_model(config, onnx_export=True, contract=None):
    YOLO = private_yolo()
    best = runtime_path(config.run_dir / 'weights/best.pt')
    if not best.is_file():
        raise FileNotFoundError('No trained best.pt')
    prefix = f'{config.mode}_{config.run_dir.name}'
    out = BASE / 'exports'
    weight = runtime_path(out / f'{prefix}.pt')
    shutil.copyfile(best, weight)
    csv_source = runtime_path(config.run_dir / 'results.csv')
    if csv_source.is_file():
        shutil.copyfile(csv_source, out / f'{prefix}_training.csv')
    if onnx_export:
        with local_console(config.run_dir / 'export.log'):
            model = YOLO(str(weight))
            exported = runtime_path(model.export(format='onnx', imgsz=config.imgsz, opset=13,
                                                 dynamic=True, simplify=False, nms=False,
                                                 half=False, batch=1, device='cpu'))
            report = inspect_onnx(exported, config.imgsz, contract)
        report['names'] = model.names
        report['weight_sha256'] = sha256(weight)
        (out / f'{prefix}_onnx_interface.json').write_text(json.dumps(report, indent=2))
    # Deliberately no zip of the run directory or image/annotation inclusion.
    return [p.name for p in sorted(out.glob(prefix + '*')) if p.suffix in ('.pt', '.onnx', '.csv')]


def export_results_txt(configs):
    """Export evaluated runs and epoch metrics as a timestamped UTF-8 report."""
    created = datetime.now(timezone(timedelta(hours=9)))
    sections = ['CrowdHuman 学習結果', f'出力日時: {created.isoformat()} (日本時間)']
    if not configs:
        raise ValueError('No training configurations selected')
    for config in configs:
        run = config.run_dir
        record = json.loads(runtime_path(run / 'evaluation.json').read_text())
        if record['specification']['config'] != asdict(config):
            raise ValueError('Evaluation must use the saved training configuration')
        if sha256(runtime_path(run / 'weights/best.pt')) != record['weight_sha256']:
            raise ValueError('Stale evaluation: evaluate the current checkpoint again')
        sections.extend(['', f'モード: {config.mode} / 実験名: {run.name}', '学習設定:',
                         json.dumps(asdict(config), ensure_ascii=False, indent=2), '学習サマリー:',
                         json.dumps(training_history(config)[0], ensure_ascii=False, indent=2), '評価結果:'])
        for key in ('precision', 'recall', 'map50', 'map50_95', 'training_seconds',
                    'inference_ms_per_image', 'model_mib', 'device'):
            sections.append(f'{key}: {record[key]}')
        sections.extend(['クラス別評価:', json.dumps(record['classes'], ensure_ascii=False, indent=2),
                         'epochごとの学習結果（タブ区切り）:'])
        with open(runtime_path(run / 'results.csv'), newline='', encoding='utf-8') as handle:
            sections.extend('\t'.join(value.strip() for value in row) for row in csv.reader(handle))
    output = runtime_path(BASE / 'exports' / f'training_results_{created:%Y%m%d_%H%M%S_%f}.txt')
    output.write_text('\n'.join(sections) + '\n', encoding='utf-8')
    return output.name


def download_selected(names):
    """Called only from an opt-in notebook cell; no external publication."""
    from google.colab import files
    for name in names:
        if Path(name).name != name or Path(name).suffix not in ('.pt', '.onnx', '.csv', '.txt'):
            raise ValueError('Only explicitly selected pt/onnx/csv/txt exports may be downloaded')
        path = runtime_path(BASE / 'exports' / name)
        if not path.is_file():
            raise FileNotFoundError('Selected export does not exist')
        files.download(str(path))
