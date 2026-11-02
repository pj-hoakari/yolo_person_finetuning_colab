"""Optional real CPU smoke test with synthetic data and random YOLO26 weights only.

Run: python3 tools/smoke_training.py
Needs torch/ultralytics/Pillow/PyYAML. No pretrained weights or real data download.
The production /content restriction is replaced ONLY inside this synthetic test.
"""
import json
import argparse
import sys
import tempfile
import shutil
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import converter, dataset, evaluator, privacy, trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-check', action='store_true', help='also export ONNX and run checker (no ONNX Runtime required)')
    parser.add_argument('--drive-resume', action='store_true', help='interrupt vbox after epoch 1 and resume after deleting its local run')
    args = parser.parse_args()
    from PIL import Image, ImageDraw
    with tempfile.TemporaryDirectory(prefix='synthetic-yolo-smoke-') as directory, ExitStack() as stack:
        root = Path(directory)
        for module in (privacy, converter, dataset, evaluator, trainer):
            stack.enter_context(patch.object(module, 'BASE', root))
        privacy.initialize()
        with privacy.local_console(root / 'setup.log'):
            YOLO = privacy.private_yolo()
        # Multiple train images permit batch=2 through the smallest feature map.
        for split, count in (('train', 2), ('val', 1)):
            records = []
            for index in range(count):
                key = f'synthetic_{split}_{index}'
                image = Image.new('RGB', (64, 64), (20, 30, 50))
                ImageDraw.Draw(image).rectangle((16, 8, 40, 56), fill=(180, 100, 90))
                image.save(root / 'raw/images' / f'{key}.jpg')
                records.append({'ID': key, 'gtboxes': [{'tag': 'person', 'vbox': [16, 8, 24, 48], 'hbox': [20, 8, 12, 12]}]})
            dataset.destination(f'annotation_{split}.odgt').write_text(''.join(json.dumps(row) + '\n' for row in records))
        sizes, _ = dataset.inspect_dataset()
        for mode in ('vbox', 'hbox', 'both'):
            converter.convert_dataset(mode, sizes)
            config = trainer.TrainConfig(mode=mode, epochs=2 if args.drive_resume and mode == 'vbox' else 1, batch=2, imgsz=64,
                                         workers=0, allow_cpu=True, run_name='synthetic')
            try:
                # Random architecture proves API plumbing; this is not fine-tuning evidence.
                with patch.object(trainer, 'load_model', side_effect=lambda _: YOLO('yolo26n.yaml')):
                    if args.drive_resume and mode == 'vbox':
                        drive = root / 'private_drive/checkpoints'
                        def interrupt_after_backup(saved):
                            raise KeyboardInterrupt('synthetic runtime interruption after saved epoch')
                        try:
                            trainer.train(config, drive_runs_dir=drive, checkpoint_status=interrupt_after_backup)
                        except KeyboardInterrupt:
                            pass
                        assert (drive / mode / config.run_dir.name / 'latest.json').is_file()
                        shutil.rmtree(config.run_dir)
                        run = trainer.train(config, resume=True, drive_runs_dir=drive)
                        manifest = json.loads((drive / mode / run.name / 'latest.json').read_text())
                        assert manifest['epoch'] == 2
                        print('Synthetic Drive: epoch 1 backup / local run deletion / optimizer resume to epoch 2 passed', flush=True)
                    else:
                        run = trainer.train(config)
                evaluator.evaluate(config)
                evaluator.export_model(config, onnx_export=False)
                from ultralytics.utils import callbacks
                model = YOLO(str(run / 'weights/best.pt'))
                self_callbacks = model.callbacks
                assert all('integrations' not in getattr(fn, '__module__', '') and
                           '.callbacks.' not in getattr(fn, '__module__', '').replace('.callbacks.base', '')
                           for fns in self_callbacks.values() for fn in fns)
                assert not list((root / 'runs').rglob('*.jpg'))
                assert not list((root / 'runs').rglob('*.png'))
                if args.onnx_check and mode == 'vbox':
                    with privacy.local_console(root / 'onnx_export.log'):
                        exported = model.export(format='onnx', imgsz=64, opset=13, dynamic=True,
                                                simplify=False, nms=False, half=False, batch=1, device='cpu')
                        import onnx
                        onnx.checker.check_model(onnx.load(exported))
                    print('Synthetic ONNX export/checker passed; ONNX Runtime/browser integration unverified', flush=True)
                print(f'{mode}: synthetic CPU train / evaluate / PT and CSV export passed', flush=True)
            except Exception:
                for log in root.rglob('*.log'):
                    print(f'--- synthetic log {log.name} ---\n{log.read_text()[-6000:]}', file=sys.stderr)
                raise
        evaluator.comparison(root / 'exports/comparison.csv')
        print('Synthetic comparison passed; temporary data deleted on exit', flush=True)


if __name__ == '__main__':
    main()
