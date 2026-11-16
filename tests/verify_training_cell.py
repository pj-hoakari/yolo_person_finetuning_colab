"""学習設定・保存無効化・NaN停止・ログZIPの検証。

PyTorchとPyYAMLのある環境で python tests/verify_training_cell.py を実行。
Trainerは代替クラスを使用し、実際のモデル学習・推論は実行しません。
"""
import ast
import csv
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch
import yaml


NOTEBOOK = json.loads((Path(__file__).resolve().parents[1] / 'yolo_person_finetuning.ipynb').read_text())


def load_definitions(cell_id, namespace):
    source = ''.join(next(cell for cell in NOTEBOOK['cells'] if cell.get('id') == cell_id)['source'])
    definitions = [node for node in ast.parse(source).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), cell_id, 'exec'), namespace)


class RecordingDetectionTrainer:
    def __init__(self, *, overrides):
        self.overrides = overrides
        self.callbacks = {}
        self.save_dir = Path(overrides['project']) / overrides['name']

    def add_callback(self, event, callback):
        self.callbacks[event] = callback

    def save_model(self):
        raise AssertionError('標準の重み保存が呼び出されました')


class TrainingCellTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        data = self.root / 'data'
        data.mkdir()
        (data / 'data.yaml').write_text('names: {0: person, 1: head}\n')
        for name in ('.build_config.json', 'preparation_report.json'):
            (data / name).write_text('{}')
        pretrained = self.root / 'initial.pt'
        pretrained.write_bytes(b'pretrained placeholder')
        self.args = (data / 'data.yaml', pretrained, self.root / 'local',
                     'test_run', 20, 640, 8, 2)
        self.ns = dict(torch=torch, yaml=yaml, Path=Path,
                       ultralytics=SimpleNamespace(__version__='8.4.0'),
                       DetectionTrainer=RecordingDetectionTrainer)
        load_definitions('training', self.ns)

    def test_new_training_uses_same_optimization_settings_without_checkpoint_io(self):
        with patch.object(torch, 'load', side_effect=AssertionError('チェックポイントを読み込んだ')):
            with patch.object(torch, 'save', side_effect=AssertionError('重みを保存した')):
                trainer = self.ns['create_training'](*self.args, require_cuda=False)
                # 標準側から最終エポックに保存を要求されても、重みを書き出しません。
                trainer.save_model()
        self.assertEqual(trainer.overrides, {
            'model': str(self.args[1].resolve()), 'data': str(self.args[0].resolve()),
            'project': str(self.args[2].resolve()), 'name': 'test_run', 'exist_ok': False,
            'epochs': 20, 'imgsz': 640, 'batch': 8, 'workers': 2, 'device': 0,
            'optimizer': 'AdamW', 'lr0': 0.001, 'seed': 0, 'patience': 0,
            'val': True, 'plots': True, 'amp': True, 'cache': False, 'close_mosaic': 10,
            'save': False, 'save_period': -1, 'resume': False,
        })
        self.assertEqual(set(trainer.callbacks), {'on_train_batch_end', 'on_train_epoch_end'})
        self.assertFalse(self.args[2].exists())

    def test_nan_and_inf_stop_training(self):
        trainer = self.ns['create_training'](*self.args, require_cuda=False)
        for field in ('loss', 'loss_items', 'tloss'):
            for abnormal in (float('nan'), float('inf')):
                state = SimpleNamespace(epoch=1, loss=torch.tensor(1.),
                                        loss_items=torch.tensor([1., 2., 3.]), tloss=torch.tensor([1., 2., 3.]))
                setattr(state, field, torch.tensor([1., abnormal]))
                for callback in trainer.callbacks.values():
                    with self.subTest(field=field, abnormal=abnormal):
                        with self.assertRaises(FloatingPointError):
                            callback(state)
        state = SimpleNamespace(epoch=0, loss=torch.tensor(1.),
                                loss_items=torch.tensor([1., 2., 3.]), tloss=torch.tensor([1., 2., 3.]))
        for callback in trainer.callbacks.values():
            callback(state)

    def test_archives_local_logs_without_weights_or_drive_copy(self):
        logs = self.root / 'local_logs'
        logs.mkdir()
        (logs / 'args.yaml').write_text('epochs: 20\ndata: /data/data.yaml\n')
        (logs / 'results.csv').write_text('epoch, metrics/mAP50-95(B), train/box_loss\n1,0.7,2\n2,0.3,1\n')
        (logs / 'results.png').write_bytes(b'plot')
        ns = dict(Path=Path, datetime=datetime, timezone=timezone, timedelta=timedelta,
                  csv=csv, json=json, math=math, shutil=shutil, tempfile=tempfile, yaml=yaml)
        load_definitions('download-results', ns)
        archive = ns['archive_training_results'](logs)
        self.addCleanup(shutil.rmtree, archive.parent)
        with zipfile.ZipFile(archive) as zipped:
            self.assertIsNone(zipped.testzip())
            names = zipped.namelist()
            self.assertIn('results.png', names)
            self.assertFalse(any(name.endswith('.pt') for name in names))
            summary = json.loads(zipped.read(next(name for name in names if name.startswith('summary_'))))
            self.assertEqual(summary['total_epochs_run'], 2)
            self.assertEqual(summary['best_epoch'], 1)
            self.assertEqual(summary['last_epoch'], 2)
            self.assertTrue(any(name.startswith('args_') and name.endswith('.txt') for name in names))
        self.assertEqual({path.name for path in logs.iterdir()}, {'args.yaml', 'results.csv', 'results.png'})


if __name__ == '__main__':
    unittest.main()
