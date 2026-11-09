"""Synthetic preparation and restore follow the same notebook entry point."""
import shutil
import unittest
from unittest.mock import Mock, patch
from src import preparation, workflow
import test_converter


class WorkflowTests(unittest.TestCase):
    write_records = test_converter.SyntheticRuntime.write_records

    def setUp(self):
        test_converter.SyntheticRuntime.setUp(self)
        guard = patch.object(preparation, 'BASE', self.base)
        guard.start()
        self.addCleanup(guard.stop)
        self.options = dict(modes=['vbox', 'hbox'], debug=True, debug_limit=1,
                            drive_dataset_dir=self.base / 'drive/dataset',
                            drive_prepared_dir=self.base / 'drive/prepared',
                            hf_repo='synthetic', hf_revision='0' * 40,
                            download_missing=False, status=Mock())

    def prepare(self, **changes):
        with patch.object(workflow, 'copy_drive_dataset') as copy, \
             patch.object(workflow, 'extract_archives') as extract:
            paths = workflow.prepare_training_data(**dict(self.options, **changes))
        return paths, copy, extract

    def test_prepare_save_clear_and_restore_without_acquisition(self):
        paths, copy, extract = self.prepare()
        copy.assert_called_once()
        extract.assert_called_once()
        self.assertEqual(set(paths), {'vbox', 'hbox'})
        self.assertTrue((self.options['drive_prepared_dir'] / 'latest.json').is_file())
        for part in ('raw/images', 'raw/annotations', 'processed'):
            shutil.rmtree(self.base / part)
            (self.base / part).mkdir()
        with patch.object(workflow, 'download_huggingface', side_effect=AssertionError('no download')):
            paths, copy, extract = self.prepare(restore=True)
        copy.assert_not_called()
        extract.assert_not_called()
        self.assertTrue(all(path.is_file() for path in paths.values()))

    def test_save_can_be_disabled_and_download_precedes_copy(self):
        events = []
        with patch.object(workflow, 'download_huggingface', side_effect=lambda *a, **k: events.append('download')), \
             patch.object(workflow, 'copy_drive_dataset', side_effect=lambda *a, **k: events.append('copy')), \
             patch.object(workflow, 'extract_archives', side_effect=lambda: events.append('extract')):
            workflow.prepare_training_data(**dict(self.options, download_missing=True, save_prepared=False))
        self.assertEqual(events, ['download', 'copy', 'extract'])
        self.assertFalse(self.options['drive_prepared_dir'].exists())

    def test_prepare_failure_does_not_publish_snapshot(self):
        with patch.object(workflow, 'copy_drive_dataset', side_effect=OSError('copy failed')):
            with self.assertRaisesRegex(OSError, 'copy failed'):
                workflow.prepare_training_data(**self.options)
        self.assertFalse(self.options['drive_prepared_dir'].exists())

    def test_restore_rejects_changed_debug_settings(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'settings mismatch'):
            self.prepare(restore=True, debug_limit=2)
