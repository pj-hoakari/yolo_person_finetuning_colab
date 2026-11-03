"""Prepared data survives runtime deletion; all inputs are synthetic."""
import io
import json
import shutil
import tarfile
import unittest
from unittest.mock import patch

from src import converter, preparation
import test_converter


class PreparedDataTests(unittest.TestCase):
    write_records = test_converter.SyntheticRuntime.write_records

    def setUp(self):
        test_converter.SyntheticRuntime.setUp(self)
        guard = patch.object(preparation, 'BASE', self.base)
        guard.start()
        self.addCleanup(guard.stop)
        self.drive = self.base / 'drive/prepared'
        self.settings = {'modes': ['vbox'], 'debug': False, 'debug_limit': None}
        sizes = {'synthetic_train': (100, 100), 'synthetic_val': (100, 100)}
        converter.convert_dataset('vbox', sizes)

    def clear_runtime_data(self):
        for part in ('raw/images', 'raw/annotations', 'processed/vbox'):
            shutil.rmtree(self.base / part)
            (self.base / part).mkdir()

    def test_restore_images_labels_yaml_and_links_after_reset(self):
        preparation.save_prepared_data(self.drive, self.settings)
        self.clear_runtime_data()
        paths = preparation.restore_prepared_data(self.drive, self.settings)
        self.assertTrue(paths['vbox'].is_file())
        converter.validate_labels(paths['vbox'].parent, 'vbox')
        links = list((paths['vbox'].parent / 'images').rglob('*.jpg'))
        self.assertEqual(len(links), 2)
        self.assertTrue(all(link.is_symlink() and link.is_file() for link in links))

    def test_settings_mismatch_and_existing_local_data_rejected(self):
        preparation.save_prepared_data(self.drive, self.settings)
        with self.assertRaisesRegex(ValueError, 'settings mismatch'):
            preparation.restore_prepared_data(self.drive, dict(self.settings, debug=True))
        with self.assertRaises(FileExistsError):
            preparation.restore_prepared_data(self.drive, self.settings)

    def test_corrupted_archive_rejected(self):
        archive = preparation.save_prepared_data(self.drive, self.settings)
        with archive.open('ab') as output:
            output.write(b'corrupt')
        self.clear_runtime_data()
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            preparation.restore_prepared_data(self.drive, self.settings)
        self.assertFalse(any((self.base / 'raw/images').iterdir()))

    def test_interrupted_save_keeps_previous_pointer(self):
        preparation.save_prepared_data(self.drive, self.settings)
        previous = (self.drive / 'latest.json').read_bytes()
        with patch.object(preparation.tarfile.TarFile, 'addfile', side_effect=OSError('Drive disconnected')):
            with self.assertRaises(OSError):
                preparation.save_prepared_data(self.drive, self.settings)
        self.assertEqual(previous, (self.drive / 'latest.json').read_bytes())
        self.assertFalse(list(self.drive.glob('*.incomplete')))

    def test_archive_traversal_rejected_even_with_matching_checksum(self):
        archive = preparation.save_prepared_data(self.drive, self.settings)
        self.clear_runtime_data()
        with tarfile.open(archive, 'w') as tar:
            entry = tarfile.TarInfo('../escape')
            entry.size = 1
            tar.addfile(entry, io.BytesIO(b'x'))
        pointer = self.drive / 'latest.json'
        manifest = json.loads(pointer.read_text())
        manifest.update(sha256=preparation._prepared_hash(archive), size=archive.stat().st_size)
        pointer.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Unsafe prepared archive path'):
            preparation.restore_prepared_data(self.drive, self.settings)
