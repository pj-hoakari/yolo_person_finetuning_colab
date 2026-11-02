"""Synthetic model bytes only; simulate a private Drive with a temporary folder."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src import checkpoints, privacy, trainer


class DriveCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-checkpoint-')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for module in (privacy, trainer):
            scope = patch.object(module, 'BASE', root / 'runtime')
            scope.start()
            self.addCleanup(scope.stop)
        self.drive = root / 'drive/checkpoints'
        self.config = trainer.TrainConfig(epochs=3, run_name='synthetic')
        self.run = self.config.run_dir
        (self.run / 'weights').mkdir(parents=True)
        self.spec = {'config': __import__('dataclasses').asdict(self.config),
                     'dataset_fingerprint': 'synthetic-only', 'versions': {'torch': 'synthetic'}}
        meta = {'specification': self.spec, 'sessions': [{'status': 'running', 'seconds': 3}]}
        (self.run / 'experiment.json').write_text(json.dumps(meta))
        (self.run / 'weights/last.pt').write_bytes(b'synthetic-optimizer-checkpoint')
        (self.run / 'weights/best.pt').write_bytes(b'synthetic-best-model')
        (self.run / 'results.csv').write_text('epoch,loss\n1,0.5\n')
        (self.run / 'args.yaml').write_text('epochs: 3\n')
        (self.run / 'training_image.jpg').write_bytes(b'synthetic-image-must-not-be-copied')

    def snapshot(self, epoch=1):
        saved = checkpoints.save_drive_checkpoint(self.config, self.drive, epoch)
        return Path(saved['directory'])

    def test_snapshot_allowlist_and_restore_after_runtime_loss(self):
        snapshot = self.snapshot()
        manifest = json.loads((snapshot / 'manifest.json').read_text())
        self.assertEqual(set(manifest['files']), set(checkpoints.CHECKPOINT_FILES))
        self.assertFalse(list(self.drive.rglob('*.jpg')))
        shutil.rmtree(self.run)
        restored = checkpoints.restore_drive_checkpoint(self.config, self.drive, self.spec)
        self.assertEqual(restored.read_bytes(), b'synthetic-optimizer-checkpoint')
        meta = json.loads((self.run / 'experiment.json').read_text())
        self.assertEqual(meta['sessions'][0]['status'], 'interrupted')
        self.assertEqual((self.run / 'results.csv').read_text(), 'epoch,loss\n1,0.5\n')

    def test_epoch_history_and_latest_pointer(self):
        self.snapshot(1)
        self.snapshot(2)
        root = checkpoints.drive_experiment_dir(self.config, self.drive)
        pointer = json.loads((root / 'latest.json').read_text())
        self.assertEqual(pointer['epoch'], 2)
        self.assertEqual(len(list(root.glob('epoch_*'))), 2)

    def test_failed_copy_preserves_previous_latest(self):
        self.snapshot(1)
        root = checkpoints.drive_experiment_dir(self.config, self.drive)
        before = (root / 'latest.json').read_bytes()
        original = shutil.copyfile
        def fail_best(source, target):
            if str(source).endswith('best.pt'):
                raise OSError('synthetic Drive disconnected')
            return original(source, target)
        with patch.object(checkpoints.shutil, 'copyfile', side_effect=fail_best):
            with self.assertRaisesRegex(OSError, 'disconnected'):
                self.snapshot(2)
        self.assertEqual((root / 'latest.json').read_bytes(), before)
        self.assertFalse(list(root.glob('*.incomplete')))

    def test_tampered_checkpoint_rejected_before_restoring(self):
        snapshot = self.snapshot()
        (snapshot / 'weights/last.pt').write_bytes(b'corrupt')
        shutil.rmtree(self.run)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            checkpoints.restore_drive_checkpoint(self.config, self.drive, self.spec)
        self.assertFalse(self.run.exists())
        self.assertFalse(list(self.run.parent.glob('*.restore-*')))

    def test_tampered_manifest_rejected(self):
        snapshot = self.snapshot()
        (snapshot / 'manifest.json').write_text('{}')
        shutil.rmtree(self.run)
        with self.assertRaisesRegex(ValueError, 'manifest checksum mismatch'):
            checkpoints.restore_drive_checkpoint(self.config, self.drive, self.spec)

    def test_configuration_dataset_or_versions_mismatch_rejected(self):
        self.snapshot()
        shutil.rmtree(self.run)
        for field in ('config', 'dataset_fingerprint', 'versions'):
            spec = copy.deepcopy(self.spec)
            spec[field] = 'different'
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                checkpoints.restore_drive_checkpoint(self.config, self.drive, spec)
        self.assertFalse(self.run.exists())

    def test_pointer_path_traversal_rejected(self):
        self.snapshot()
        root = checkpoints.drive_experiment_dir(self.config, self.drive)
        (root / 'latest.json').write_text(json.dumps({'snapshot': '../outside'}))
        shutil.rmtree(self.run)
        with self.assertRaisesRegex(ValueError, 'pointer'):
            checkpoints.restore_drive_checkpoint(self.config, self.drive, self.spec)

    def test_nonempty_local_run_preserved(self):
        self.snapshot()
        before = (self.run / 'weights/last.pt').read_bytes()
        with self.assertRaises(FileExistsError):
            checkpoints.restore_drive_checkpoint(self.config, self.drive, self.spec)
        self.assertEqual((self.run / 'weights/last.pt').read_bytes(), before)

    def test_unexpected_snapshot_file_rejected(self):
        snapshot = self.snapshot()
        manifest_path = snapshot / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['../../outside'] = {'size': 1, 'sha256': 'fake'}
        manifest_path.write_text(json.dumps(manifest))
        root = checkpoints.drive_experiment_dir(self.config, self.drive)
        pointer = json.loads((root / 'latest.json').read_text())
        pointer['manifest_sha256'] = checkpoints._checkpoint_hash(manifest_path)
        (root / 'latest.json').write_text(json.dumps(pointer))
        shutil.rmtree(self.run)
        with self.assertRaisesRegex(ValueError, 'file list'):
            checkpoints.restore_drive_checkpoint(self.config, self.drive, self.spec)
