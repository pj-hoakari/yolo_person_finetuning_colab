"""合成テンソルによる自動保存・再開・NaN停止・ログZIPの検証。

PyTorchとPyYAMLのある環境で python tests/verify_training_cell.py を実行。
Trainerは代替クラスを使用し、実モデルの学習・推論は実行しません。
"""
import ast
from copy import deepcopy
import csv
from datetime import datetime, timezone, timedelta
import hashlib
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


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path, text):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


class RecordingDetectionTrainer:
    def __init__(self, *, overrides):
        self.overrides = overrides
        self.args = SimpleNamespace(**overrides)
        self.callbacks = {}
        self.save_dir = Path(overrides['project']) / overrides['name']

    def add_callback(self, event, callback):
        self.callbacks[event] = callback

    def validate(self):
        # 8.4.0のValidatorと同じEMAのdtype変換。推論や学習は実行しません。
        model = self.ema.ema
        model.half() if self.amp else model.float()
        model.float()
        return {'metrics/mAP50-95(B)': 0.5}, 0.5

    def run_callbacks(self, event):
        callback = self.callbacks.get(event)
        if callback is not None:
            callback(self)


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
        self.args = (data / 'data.yaml', pretrained, self.root / 'drive', self.root / 'local',
                     'test_run', 20, 640, 8, 2)
        self.ns = dict(torch=torch, yaml=yaml, Path=Path, hashlib=hashlib, json=json,
                       math=math, shutil=shutil, deepcopy=deepcopy, datetime=datetime,
                       sha256_file=sha256_file, atomic_text=atomic_text,
                       unwrap_model=lambda model: model,
                       ultralytics=SimpleNamespace(__version__='8.4.0'),
                       DetectionTrainer=RecordingDetectionTrainer)
        load_definitions('training', self.ns)
        self.run = self.root / 'drive' / 'test_run'

    def trainer(self, epoch=0):
        trainer = self.ns['create_training'](*self.args, require_cuda=False)
        trainer.epoch = epoch
        trainer.loss = torch.tensor(1.)
        trainer.loss_items = trainer.tloss = torch.tensor([1., 2., 3.])
        trainer.model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            trainer.model.weight.fill_(100_000.)
        trainer.ema = SimpleNamespace(ema=deepcopy(trainer.model), updates=50)
        with torch.no_grad():
            trainer.ema.ema.weight.fill_(90_000.)
        state = {'state': {0: {'step': torch.tensor(5.), 'exp_avg': torch.tensor([1e-4]),
                               'exp_avg_sq': torch.tensor([1e-12])}}, 'param_groups': []}
        trainer.optimizer = SimpleNamespace(state_dict=lambda: state)
        trainer.scaler = SimpleNamespace(state_dict=lambda: {'scale': 512.})
        trainer.last = trainer.save_dir / 'weights' / 'last.pt'
        trainer.best = trainer.save_dir / 'weights' / 'best.pt'
        trainer.csv = trainer.save_dir / 'results.csv'
        trainer.csv.write_text('epoch,loss\n1,1\n')
        (trainer.save_dir / 'args.yaml').write_text('epochs: 20\n')
        trainer.best_fitness = trainer.fitness = 0.5
        trainer.metrics = {'metrics/mAP50-95(B)': 0.5}
        trainer.read_results_csv = lambda: {'epoch': [1]}
        return trainer

    def commit(self, trainer):
        trainer.save_model()
        trainer.callbacks['on_model_save'](trainer)

    def test_settings_and_fp32_save(self):
        trainer = self.trainer()
        self.assertEqual(trainer.overrides['optimizer'], 'AdamW')
        self.assertEqual(trainer.overrides['lr0'], 0.001)
        self.assertTrue(trainer.overrides['amp'])
        self.assertEqual(trainer.overrides['patience'], 0)
        self.assertTrue(trainer.overrides['save'])
        self.assertFalse(trainer.overrides['resume'])
        self.commit(trainer)
        snapshot = self.run / 'epochs/epoch_000001'
        checkpoint = torch.load(snapshot / 'checkpoint.pt', map_location='cpu', weights_only=False)
        self.assertEqual(checkpoint['ema'].weight.dtype, torch.float32)
        self.assertEqual(checkpoint['ema'].weight.item(), 90_000.)
        self.assertEqual(checkpoint['training_model_state']['weight'].item(), 100_000.)
        moment = checkpoint['optimizer']['state'][0]['exp_avg_sq']
        self.assertEqual(moment.dtype, torch.float32)
        torch.testing.assert_close(moment, torch.tensor([1e-12]), rtol=0, atol=0)
        self.assertEqual(checkpoint['scaler'], {'scale': 512.})
        self.assertEqual(json.loads((self.run / 'latest.json').read_text())['completed_epochs'], 1)

    def test_resume_restores_training_model_separately_from_ema(self):
        trainer = self.trainer()
        self.commit(trainer)
        # ランタイムのローカル出力が失われた状態を再現。
        (self.root / 'local').rename(self.root / 'previous_session')
        resumed = self.ns['create_training'](*self.args, require_cuda=False)
        self.assertEqual(resumed.overrides['resume'], str(resumed.save_dir / 'weights/last.pt'))
        resumed.model = torch.nn.Linear(1, 1)
        resumed.callbacks['on_pretrain_routine_end'](resumed)
        self.assertEqual(resumed.model.weight.item(), 100_000.)
        checkpoint = torch.load(resumed.save_dir / 'weights/last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(checkpoint['ema'].weight.item(), 90_000.)
        self.assertTrue((resumed.save_dir / 'results.csv').is_file())

    def test_save_and_finalization_preserve_original_dtypes_and_values(self):
        trainer = self.trainer()
        trainer.model.double()
        trainer.ema.ema.double()
        trainer.ema.ema.register_buffer('half_buffer', torch.tensor([0.125], dtype=torch.float16))
        trainer.ema.ema.register_buffer('counter', torch.tensor([3], dtype=torch.int64))
        state = trainer.optimizer.state_dict()['state'][0]
        state['exp_avg_sq'] = state['exp_avg_sq'].double()
        expected = deepcopy(trainer.ema.ema.state_dict())
        self.commit(trainer)
        checkpoint = torch.load(trainer.last, map_location='cpu', weights_only=False)
        self.assertEqual(checkpoint['resume_precision'], 'preserved')
        self.assertEqual(checkpoint['training_model_state']['weight'].dtype, torch.float64)
        self.assertEqual(checkpoint['optimizer']['state'][0]['exp_avg_sq'].dtype, torch.float64)
        for name, value in checkpoint['ema'].state_dict().items():
            self.assertEqual(value.dtype, expected[name].dtype)
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        self.ns['finalize_checkpoint'](trainer.last)
        finalized = torch.load(trainer.last, map_location='cpu', weights_only=False)
        for name, value in finalized['model'].state_dict().items():
            self.assertEqual(value.dtype, expected[name].dtype)
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_saving_does_not_mutate_live_training_state(self):
        trainer = self.trainer()
        trainer.model.register_buffer('running_value', torch.tensor([0.1, 0.2]))
        states = {
            'model': trainer.model.state_dict(),
            'ema': trainer.ema.ema.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'scaler': trainer.scaler.state_dict(),
        }
        before = deepcopy(states)
        rng_before = torch.get_rng_state().clone()
        flags = (trainer.model.training, trainer.ema.ema.training)
        self.commit(trainer)

        def assert_same(left, right):
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertEqual(left.dtype, right.dtype)
                self.assertEqual(left.device, right.device)
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    assert_same(left[key], right[key])
            elif isinstance(left, (tuple, list)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    assert_same(a, b)
            else:
                self.assertEqual(left, right)

        assert_same(before, states)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(flags, (trainer.model.training, trainer.ema.ema.training))

    def test_fp16_validation_overflow_reproduced_and_fixed(self):
        trainer = self.trainer()
        trainer.ema.ema.add_module('bn', torch.nn.BatchNorm1d(1))
        trainer.ema.ema.bn.running_var.fill_(100_000.)
        trainer.amp = True
        unprotected = SimpleNamespace(ema=deepcopy(trainer.ema), amp=True)
        RecordingDetectionTrainer.validate(unprotected)
        self.assertTrue(torch.isinf(unprotected.ema.ema.bn.running_var).all())
        expected = deepcopy(trainer.ema.ema.state_dict())
        trainer.validate()
        for name, value in trainer.ema.ema.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        self.assertTrue(trainer.amp)
        self.assertTrue(trainer.args.amp)
        self.commit(trainer)
        checkpoint = torch.load(trainer.last, map_location='cpu', weights_only=False)
        self.assertEqual(checkpoint['ema'].bn.running_var.item(), 100_000.)
        self.assertEqual(checkpoint['ema'].bn.running_var.dtype, torch.float32)

    def test_training_amp_restored_after_validation_error(self):
        trainer = self.trainer()
        trainer.amp = True
        with patch.object(RecordingDetectionTrainer, 'validate', side_effect=RuntimeError('validation failure')):
            with self.assertRaisesRegex(RuntimeError, 'validation failure'):
                trainer.validate()
        self.assertTrue(trainer.amp)
        self.assertTrue(trainer.args.amp)

    def test_final_eval_does_not_half_cast_batchnorm_statistics(self):
        trainer = self.trainer()
        trainer.ema.ema.add_module('bn', torch.nn.BatchNorm1d(1))
        trainer.ema.ema.bn.running_var.fill_(100_000.)
        self.commit(trainer)

        class RecordingValidator:
            args = SimpleNamespace(half=True, plots=False, compile=True)

            def __call__(self, *, model):
                self.checkpoint = torch.load(model, map_location='cpu', weights_only=False)
                return {'fitness': 0.5, 'metrics/mAP50-95(B)': 0.5}

        trainer.validator = RecordingValidator()
        trainer.final_eval()
        trainer.callbacks['on_train_end'](trainer)
        for name in ('best.pt', 'last.pt'):
            path = self.run / 'final/weights' / name
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['model'].bn.running_var.item(), 100_000.)
            self.assertEqual(checkpoint['model'].bn.running_var.dtype, torch.float32)
            self.assertIsNone(checkpoint['optimizer'])
            self.assertNotIn('training_model_state', checkpoint)
        self.assertFalse(trainer.validator.args.half)
        # 再開用のDriveスナップショットは最終処理後も残ります。
        checkpoint = torch.load(self.run / 'epochs/epoch_000001/checkpoint.pt',
                                map_location='cpu', weights_only=False)
        self.assertIsNotNone(checkpoint['optimizer'])

    def test_live_and_copy_anomalies_are_distinguished(self):
        trainer = self.trainer()
        with torch.no_grad():
            trainer.ema.ema.weight.fill_(float('inf'))
        with self.assertRaisesRegex(FloatingPointError, r'保存開始前.ema.weight.*Inf=1'):
            trainer.save_model()
        self.assertFalse(trainer.last.exists())
        with torch.no_grad():
            trainer.ema.ema.weight.fill_(90_000.)

        def corrupted_copy(value):
            copied = deepcopy(value)
            if isinstance(copied, torch.nn.Module):
                with torch.no_grad():
                    copied.weight.fill_(float('nan'))
            return copied

        with patch.dict(self.ns, deepcopy=corrupted_copy):
            with self.assertRaisesRegex(FloatingPointError, r'保存用コピー.ema.weight.*NaN=1'):
                trainer.save_model()
        self.assertEqual(trainer.ema.ema.weight.item(), 90_000.)
        self.assertFalse(trainer.last.exists())
        self.assertFalse((self.run / 'latest.json').exists())

    def test_nan_or_inf_never_advances_latest(self):
        trainer = self.trainer()
        self.commit(trainer)
        latest = (self.run / 'latest.json').read_bytes()
        trainer.epoch = 1
        for key in ('loss', 'loss_items', 'tloss'):
            original = getattr(trainer, key)
            for abnormal in (float('nan'), float('inf')):
                setattr(trainer, key, torch.tensor([1., abnormal]))
                with self.assertRaises(FloatingPointError):
                    trainer.callbacks['on_train_batch_end'](trainer)
                with self.assertRaises(FloatingPointError):
                    trainer.callbacks['on_model_save'](trainer)
            setattr(trainer, key, original)
        trainer.optimizer.state_dict()['state'][0]['exp_avg_sq'].fill_(float('nan'))
        before = trainer.last.read_bytes()
        with self.assertRaises(FloatingPointError):
            trainer.save_model()
        self.assertEqual(trainer.last.read_bytes(), before)
        self.assertEqual((self.run / 'latest.json').read_bytes(), latest)
        self.assertFalse((self.run / 'epochs/epoch_000002').exists())

    def test_save_failures_keep_latest(self):
        trainer = self.trainer()
        self.commit(trainer)
        latest = (self.run / 'latest.json').read_bytes()
        trainer.epoch = 1
        with patch.object(torch, 'save', side_effect=IOError('local save failure')):
            with self.assertRaisesRegex(IOError, 'local save failure'):
                trainer.save_model()
        original_copy = self.ns['copy_verified']

        def fail_copy(source, target):
            if 'epoch_000002' in str(target) and target.name == 'best.pt':
                raise IOError('Drive copy failure')
            return original_copy(source, target)

        trainer.save_model()
        with patch.dict(self.ns, copy_verified=fail_copy):
            with self.assertRaisesRegex(IOError, 'Drive copy failure'):
                trainer.callbacks['on_model_save'](trainer)
        self.assertEqual((self.run / 'latest.json').read_bytes(), latest)
        resumed = self.ns['create_training'](*self.args, require_cuda=False)
        checkpoint = torch.load(resumed.save_dir / 'weights/last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(checkpoint['epoch'], 0)

    def test_completed_run_and_config_change(self):
        trainer = self.trainer(epoch=19)
        self.commit(trainer)
        self.assertIsNone(self.ns['create_training'](*self.args, require_cuda=False))
        changed = list(self.args)
        changed[5] = 21
        with self.assertRaisesRegex(RuntimeError, '別設定'):
            self.ns['create_training'](*changed, require_cuda=False)

    def test_nonfinite_checkpoint_rejected_even_when_hash_matches(self):
        trainer = self.trainer()
        self.commit(trainer)
        snapshot = self.run / 'epochs/epoch_000001'
        path = snapshot / 'checkpoint.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        checkpoint['training_model_state']['weight'].fill_(float('inf'))
        torch.save(checkpoint, path)
        metadata = json.loads((snapshot / 'snapshot.json').read_text())
        metadata['files']['checkpoint.pt'] = sha256_file(path)
        (snapshot / 'snapshot.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(FloatingPointError, '再開前.training_model_state'):
            self.ns['create_training'](*self.args, require_cuda=False)

    def test_previous_best_preserved_and_final_artifacts_saved(self):
        trainer = self.trainer()
        self.commit(trainer)
        best_bytes = trainer.best.read_bytes()
        trainer.epoch = 1
        trainer.fitness = 0.3
        trainer.save_model()
        self.assertEqual(trainer.best.read_bytes(), best_bytes)
        trainer.callbacks['on_model_save'](trainer)
        trainer.callbacks['on_train_end'](trainer)
        self.assertTrue((self.run / 'final/weights/best.pt').is_file())
        self.assertTrue((self.run / 'final/results.csv').is_file())
        self.assertEqual(json.loads((self.run / 'final/completed.json').read_text())['completed_epochs'], 2)

    def test_corrupt_file_rejected_by_hash(self):
        trainer = self.trainer()
        self.commit(trainer)
        path = self.run / 'epochs/epoch_000001/checkpoint.pt'
        with path.open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(IOError, '検証に失敗'):
            self.ns['create_training'](*self.args, require_cuda=False)

    def test_archives_local_logs(self):
        logs = self.root / 'local_logs'
        logs.mkdir()
        (logs / 'args.yaml').write_text('epochs: 20\ndata: /data/data.yaml\n')
        (logs / 'results.csv').write_text('epoch, metrics/mAP50-95(B), train/box_loss\n1,0.7,2\n2,0.3,1\n')
        ns = dict(Path=Path, datetime=datetime, timezone=timezone, timedelta=timedelta,
                  csv=csv, json=json, math=math, shutil=shutil, tempfile=tempfile, yaml=yaml)
        load_definitions('download-results', ns)
        archive = ns['archive_training_results'](logs)
        self.addCleanup(shutil.rmtree, archive.parent)
        with zipfile.ZipFile(archive) as zipped:
            self.assertIsNone(zipped.testzip())
            summary = json.loads(zipped.read(next(name for name in zipped.namelist() if name.startswith('summary_'))))
            self.assertEqual(summary['total_epochs_run'], 2)
            self.assertEqual(summary['best_epoch'], 1)


if __name__ == '__main__':
    unittest.main()
