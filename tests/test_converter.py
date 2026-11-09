"""Synthetic-only tests. No CrowdHuman data is downloaded or used."""
import ast
import copy
import io
import json
import logging
import math
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from PIL import Image
from src import converter, dataset, evaluator, privacy, trainer

PROJECT = Path(__file__).resolve().parents[1]


def person(**changes):
    box = {'tag': 'person', 'vbox': [10, 20, 30, 40], 'hbox': [20, 20, 10, 10]}
    box.update(changes)
    return box


class BoxTests(unittest.TestCase):
    def test_vbox(self):
        rows, stats = converter.convert_boxes([person()], 'vbox', 100, 100)
        self.assertEqual(rows, [(0, .25, .4, .3, .4)])
        self.assertEqual(stats['vbox_accepted'], 1)

    def test_hbox(self):
        rows, _ = converter.convert_boxes([person()], 'hbox', 100, 100)
        self.assertEqual(rows, [(0, .25, .25, .1, .1)])

    def test_both_classes(self):
        rows, _ = converter.convert_boxes([person()], 'both', 100, 100)
        self.assertEqual([row[0] for row in rows], [0, 1])

    def test_ignore_independent(self):
        for extra, head, expected in [(0, 0, [0, 1]), (1, 0, [1]), (0, 1, [0]), (1, 1, [])]:
            with self.subTest(extra=extra, head=head):
                rows, _ = converter.convert_boxes([person(extra={'ignore': extra}, head_attr={'ignore': head})], 'both', 100, 100)
                self.assertEqual([row[0] for row in rows], expected)

    def test_non_person_and_bad_annotation(self):
        rows, stats = converter.convert_boxes([person(tag='mask'), None], 'both', 100, 100)
        self.assertEqual(rows, [])
        self.assertEqual(stats, {'non_person': 1, 'invalid_annotation': 1})

    def test_clipping(self):
        self.assertEqual(converter.normalize_box([-20, -10, 150, 120], 100, 100), (.5, .5, 1, 1))
        rows, stats = converter.convert_boxes([person(vbox=[-10, 0, 20, 20])], 'vbox', 100, 100)
        self.assertEqual(rows[0][1:], (.05, .1, .1, .2))
        self.assertEqual(stats['vbox_clipped'], 1)

    def test_invalid_boxes(self):
        for box in [None, [], [0, 0, 0, 1], [0, 0, 1, -1], [101, 0, 5, 5], [0, 0, math.nan, 1], [0, 0, math.inf, 1], ['0', 0, 1, 1], [True, 0, 1, 1]]:
            with self.subTest(box=box):
                rows, stats = converter.convert_boxes([person(vbox=box)], 'vbox', 100, 100)
                self.assertEqual(rows, [])
                self.assertEqual(stats['vbox_invalid'], 1)

    def test_invalid_ignore(self):
        for attr in [None, [], {'ignore': '1'}, {'ignore': 2}]:
            rows, stats = converter.convert_boxes([person(extra=attr)], 'both', 100, 100)
            self.assertEqual([row[0] for row in rows], [1])
            self.assertEqual(stats['vbox_invalid'], 1)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            converter.convert_boxes([], 'fbox', 100, 100)


class SyntheticRuntime(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-crowdhuman-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        for module in (privacy, dataset, converter, trainer, evaluator):
            guard = patch.object(module, 'BASE', self.base)
            guard.start()
            self.addCleanup(guard.stop)
        for relative in ('raw/archives', 'raw/annotations', 'raw/images', 'exports', 'runs', 'processed'):
            (self.base / relative).mkdir(parents=True)
        self.records = {'train': [{'ID': 'synthetic_train', 'gtboxes': [person()]}],
                        'val': [{'ID': 'synthetic_val', 'gtboxes': [person()]}]}
        for split in self.records:
            Image.new('RGB', (100, 100), (30, 60, 90)).save(self.base / 'raw/images' / f'synthetic_{split}.jpg')
        self.write_records()

    def write_records(self):
        for split, records in self.records.items():
            dataset.destination(f'annotation_{split}.odgt').write_text(''.join(json.dumps(r) + '\n' for r in records))

    def archives(self, malicious=False):
        for index, name in enumerate(dataset.ARCHIVES):
            with zipfile.ZipFile(dataset.destination(name), 'w') as z:
                if malicious and index == 0:
                    z.writestr('../escape.jpg', b'unsafe')
                else:
                    buffer = io.BytesIO()
                    Image.new('RGB', (16, 16), (index, 20, 40)).save(buffer, format='JPEG')
                    z.writestr(f'Images/synthetic_archive{index}.jpg', buffer.getvalue())

    def test_streaming_parser_and_sizes(self):
        sizes, counts = dataset.inspect_dataset()
        self.assertEqual(counts, {'train': 1, 'val': 1})
        self.assertEqual(sizes['synthetic_train'], (100, 100))

    def test_conversion_separation_yaml_and_reuse(self):
        sizes, _ = dataset.inspect_dataset()
        for mode in ('vbox', 'hbox', 'both'):
            target = converter.convert_dataset(mode, sizes)
            data = yaml.safe_load(target.read_text())
            self.assertEqual(data['names'], converter.NAMES[mode])
            root = target.parent
            self.assertEqual(converter.validate_labels(root, mode)['train']['images'], 1)
            self.assertTrue((root / 'images/train/synthetic_train.jpg').is_symlink())
            self.assertFalse((root / 'images/val/synthetic_train.jpg').exists())
            before = (root / 'labels/train/synthetic_train.txt').stat().st_mtime_ns
            self.assertEqual(converter.convert_dataset(mode, sizes), target)
            self.assertEqual((root / 'labels/train/synthetic_train.txt').stat().st_mtime_ns, before)

    def test_changed_debug_rejected(self):
        sizes, _ = dataset.inspect_dataset()
        converter.convert_dataset('vbox', sizes, limit=1)
        with self.assertRaises(ValueError):
            converter.convert_dataset('vbox', sizes)

    def test_changed_annotation_rejected(self):
        sizes, _ = dataset.inspect_dataset()
        converter.convert_dataset('vbox', sizes)
        self.records['train'][0]['gtboxes'][0]['vbox'] = [1, 1, 5, 5]
        self.write_records()
        with self.assertRaises(ValueError):
            converter.convert_dataset('vbox', sizes)

    def test_background_image_retained(self):
        self.records['train'].append({'ID': 'synthetic_background', 'gtboxes': [person(tag='mask')]})
        Image.new('RGB', (100, 100)).save(self.base / 'raw/images/synthetic_background.jpg')
        self.write_records()
        sizes, _ = dataset.inspect_dataset()
        root = converter.convert_dataset('vbox', sizes).parent
        self.assertEqual((root / 'labels/train/synthetic_background.txt').read_text(), '')
        self.assertEqual(converter.validate_labels(root, 'vbox')['train']['images'], 2)

    def test_yaml_tamper_rejected(self):
        sizes, _ = dataset.inspect_dataset()
        root = converter.convert_dataset('vbox', sizes).parent
        (root / 'data.yaml').write_text('path: /tmp\n')
        with self.assertRaises(ValueError):
            converter.validate_labels(root, 'vbox')

    def test_class_tamper_rejected(self):
        sizes, _ = dataset.inspect_dataset()
        root = converter.convert_dataset('vbox', sizes).parent
        (root / 'labels/train/synthetic_train.txt').write_text('1 0.5 0.5 0.1 0.1\n')
        with self.assertRaises(ValueError):
            converter.validate_labels(root, 'vbox')

    def test_missing_label_rejected(self):
        sizes, _ = dataset.inspect_dataset()
        root = converter.convert_dataset('vbox', sizes).parent
        (root / 'labels/train/synthetic_train.txt').unlink()
        with self.assertRaises(ValueError):
            converter.validate_labels(root, 'vbox')

    def test_overlap_and_duplicates(self):
        self.records['val'] = copy.deepcopy(self.records['train'])
        self.write_records()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            dataset.inspect_dataset()
        self.records['val'] = []
        self.records['train'] *= 2
        self.write_records()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            dataset.inspect_dataset()

    def test_malformed_and_unsafe_odgt(self):
        path = dataset.destination('annotation_train.odgt')
        for value in ['not json', '{}', '[]', '{"ID":"../private", "gtboxes":[]}']:
            path.write_text(value)
            with self.assertRaisesRegex(ValueError, 'Invalid ODGT record'):
                list(dataset.read_odgt(path))

    def test_original_style_comma_ids_parse_and_convert(self):
        for split in ('train', 'val'):
            old_id = f'synthetic_{split}'
            new_id = f'synthetic,{split}_001'
            self.records[split][0]['ID'] = new_id
            (self.base / 'raw/images' / f'{old_id}.jpg').rename(
                self.base / 'raw/images' / f'{new_id}.jpg')
        self.write_records()
        sizes, counts = dataset.inspect_dataset()
        self.assertEqual(counts, {'train': 1, 'val': 1})
        for mode in ('vbox', 'hbox', 'both'):
            root = converter.convert_dataset(mode, sizes).parent
            for split in ('train', 'val'):
                self.assertTrue((root / 'images' / split / f'synthetic,{split}_001.jpg').is_symlink())
                self.assertTrue((root / 'labels' / split / f'synthetic,{split}_001.txt').is_file())
            self.assertEqual(converter.validate_labels(root, mode)['val']['images'], 1)

    def test_comma_support_still_rejects_unsafe_ids(self):
        path = dataset.destination('annotation_train.odgt')
        for image_id in ('', '..', '../synthetic,001', '/tmp/synthetic,001',
                         'folder/synthetic,001', 'folder\\synthetic,001',
                         'synthetic,001\n', 'synthetic,001.jpg', None, 123):
            with self.subTest(image_id=image_id):
                path.write_text(json.dumps({'ID': image_id, 'gtboxes': []}))
                with self.assertRaisesRegex(ValueError, 'invalid image ID') as error:
                    list(dataset.read_odgt(path))
                self.assertNotIn('synthetic', str(error.exception))

    def test_odgt_validation_reports_specific_reason(self):
        path = dataset.destination('annotation_train.odgt')
        for payload, reason in (('not-json', 'invalid JSON'), ('[]', 'expected a JSON object'),
                                ('{"ID":"synthetic,001","gtboxes":{}}', 'gtboxes must be a list')):
            path.write_text(payload)
            with self.assertRaisesRegex(ValueError, reason):
                list(dataset.read_odgt(path))

    def test_missing_files(self):
        missing = self.base / 'raw/annotations/missing.odgt'
        with self.assertRaises(FileNotFoundError):
            list(dataset.read_odgt(missing))
        with self.assertRaises(FileNotFoundError):
            dataset.validate_file(self.base / 'raw/archives/missing.zip', 'CrowdHuman_val.zip')
        with self.assertRaises(ValueError):
            dataset.image_size(self.base / 'raw/images/missing.jpg')

    def test_missing_corrupt_image(self):
        path = self.base / 'raw/images/synthetic_train.jpg'
        path.write_bytes(b'not an image')
        with self.assertRaisesRegex(ValueError, 'Missing or corrupt'):
            dataset.inspect_dataset()

    def test_archive_integrity_reuse_and_corrupt_repair(self):
        self.archives()
        self.assertEqual(dataset.extract_archives()['archive_images'], 4)
        image = self.base / 'raw/images/synthetic_archive0.jpg'
        before = image.stat().st_mtime_ns
        dataset.extract_archives()
        self.assertEqual(image.stat().st_mtime_ns, before)
        original = image.read_bytes()
        image.write_bytes(b'x' * len(original))
        dataset.extract_archives()
        self.assertEqual(image.read_bytes(), original)

    def test_zip_traversal_rejected(self):
        self.archives(malicious=True)
        with self.assertRaisesRegex(ValueError, 'Unexpected archive'):
            dataset.extract_archives()
        self.assertFalse((self.base / 'escape.jpg').exists())

    def test_bad_zip(self):
        path = dataset.destination('CrowdHuman_val.zip')
        path.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'Invalid ZIP'):
            dataset.validate_file(path, path.name)

    def test_free_space_check(self):
        self.archives()
        with patch.object(dataset.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 1})()):
            with self.assertRaises(OSError):
                dataset.extract_archives()

    def test_download_reuses_existing_without_network(self):
        self.archives()
        with patch.object(dataset.urllib.request, 'urlopen', side_effect=AssertionError('Network must not be called')):
            dataset.download_huggingface()

    def test_huggingface_links_pin_original_files(self):
        links = dataset.huggingface_links()
        self.assertEqual(set(links), set(dataset.ARCHIVES + dataset.ANNOTATIONS))
        for name, url in links.items():
            self.assertEqual(url, f'https://huggingface.co/datasets/{dataset.HF_REPO_ID}/resolve/{dataset.HF_REVISION}/{name}?download=true')
        self.assertNotIn('CrowdHuman_test.zip', links)

    def test_huggingface_source_validation(self):
        for repo, revision in (('../unsafe', dataset.HF_REVISION), ('org/repo?token=x', dataset.HF_REVISION), (dataset.HF_REPO_ID, 'main'), (dataset.HF_REPO_ID, '../main')):
            with self.assertRaises(ValueError):
                dataset.huggingface_links(repo, revision)

    def test_download_rejects_html_and_cleans_partial(self):
        dataset.destination('annotation_train.odgt').unlink()
        self.archives()
        response = io.BytesIO(b'<html>login</html>')
        response.headers = {'Content-Type': 'text/html'}
        with patch.object(dataset, 'huggingface_links', return_value={'annotation_train.odgt': 'https://huggingface.co/datasets/owner/repo/resolve/revision/annotation_train.odgt'}), patch.object(dataset.urllib.request, 'urlopen', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'Hugging Face download failed'):
                dataset.download_huggingface()
        self.assertFalse(dataset.destination('annotation_train.odgt').exists())
        self.assertEqual(list((self.base / 'raw/annotations').glob('*.part')), [])

    def test_huggingface_download_validates_and_records_source(self):
        self.archives()
        name = 'annotation_train.odgt'
        destination = dataset.destination(name)
        payload = destination.read_bytes()
        destination.unlink()
        response = io.BytesIO(payload)
        response.headers = {'Content-Type': 'application/octet-stream', 'Content-Length': str(len(payload))}
        with patch.object(dataset.urllib.request, 'urlopen', return_value=response) as request:
            dataset.download_huggingface()
        self.assertEqual(destination.read_bytes(), payload)
        request.assert_called_once_with(dataset.huggingface_links()[name], timeout=120)
        source = json.loads((self.base / 'raw/download_source.json').read_text())
        self.assertEqual(source['downloaded'], [name])
        self.assertEqual(source['requested_revision'], dataset.HF_REVISION)
        self.assertEqual(len(source['reused_existing']), 5)

    def test_huggingface_rejects_truncated_response(self):
        self.archives()
        name = 'annotation_train.odgt'
        destination = dataset.destination(name)
        payload = destination.read_bytes()
        destination.unlink()
        response = io.BytesIO(payload)
        response.headers = {'Content-Type': 'application/octet-stream', 'Content-Length': str(len(payload) + 1)}
        with patch.object(dataset.urllib.request, 'urlopen', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'Hugging Face download failed'):
                dataset.download_huggingface()
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_suffix('.odgt.part').exists())

    def prepare_download_failure(self):
        self.archives()
        dataset.destination('annotation_train.odgt').unlink()

    def download_log_entries(self):
        return [json.loads(line) for line in (self.base / 'logs/download.jsonl').read_text().splitlines()]

    def test_http_error_log_is_diagnostic_and_redacted(self):
        self.prepare_download_failure()
        error = dataset.urllib.error.HTTPError(
            'https://cdn.example/file?token=private-signature', 403,
            'Forbidden https://cdn.example/file?token=private-signature Bearer hf_privatecredential',
            {'X-Request-Id': 'synthetic-request', 'Authorization': 'Bearer never-log-this'},
            io.BytesIO(b'private response body'))
        with patch.object(dataset.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'HTTPError.*403.*Log:') as caught:
                dataset.download_huggingface()
        entry = self.download_log_entries()[-1]
        self.assertEqual(entry['event'], 'error')
        self.assertEqual(entry['filename'], 'annotation_train.odgt')
        self.assertEqual(entry['stage'], 'http_request')
        self.assertEqual(entry['errors'][0]['http_status'], 403)
        self.assertEqual(entry['errors'][0]['response_headers']['X-Request-Id'], 'synthetic-request')
        self.assertTrue(entry['errors'][0]['stack'])
        self.assertTrue(entry['timestamp_utc'])
        text = (self.base / 'logs/download.jsonl').read_text() + str(caught.exception)
        for secret in ('private-signature', 'hf_privatecredential', 'never-log-this', 'private response body'):
            self.assertNotIn(secret, text)
        self.assertEqual(error.fp.tell(), 0)

    def test_dns_error_log(self):
        self.prepare_download_failure()
        error = dataset.urllib.error.URLError(OSError(-3, 'Temporary failure in name resolution'))
        with patch.object(dataset.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'URLError.*name resolution'):
                dataset.download_huggingface()
        entry = self.download_log_entries()[-1]
        self.assertEqual(entry['errors'][0]['type'], 'URLError')
        self.assertIn('name resolution', entry['errors'][0]['reason'])

    def test_existing_corrupt_file_error_is_logged(self):
        self.archives()
        dataset.destination('CrowdHuman_train01.zip').write_bytes(b'synthetic-corrupt-zip')
        with patch.object(dataset.urllib.request, 'urlopen', side_effect=AssertionError('must not download')):
            with self.assertRaisesRegex(RuntimeError, 'Invalid ZIP'):
                dataset.download_huggingface()
        entry = self.download_log_entries()[-1]
        self.assertEqual(entry['stage'], 'existing_file_validation')
        self.assertEqual(entry['filename'], 'CrowdHuman_train01.zip')
        self.assertEqual(entry['errors'][0]['type'], 'ValueError')

    def test_truncated_response_log_has_byte_counts(self):
        self.prepare_download_failure()
        payload = json.dumps({'ID': 'synthetic', 'gtboxes': []}).encode()
        response = io.BytesIO(payload)
        response.headers = {'Content-Length': str(len(payload) + 10)}
        with patch.object(dataset.urllib.request, 'urlopen', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'Incomplete Hugging Face response'):
                dataset.download_huggingface()
        entry = self.download_log_entries()[-1]
        self.assertEqual(entry['stage'], 'transfer')
        self.assertEqual(entry['bytes_received'], len(payload))
        self.assertEqual(entry['expected_bytes'], len(payload) + 10)

    def test_success_logs_append_attempts_and_reuse(self):
        self.archives()
        first_path = dataset.download_huggingface()
        dataset.download_huggingface()
        entries = self.download_log_entries()
        self.assertEqual(first_path, self.base / 'logs/download.jsonl')
        self.assertEqual(len({entry['attempt_id'] for entry in entries}), 2)
        self.assertEqual(sum(entry['event'] == 'reused' for entry in entries), 12)
        self.assertEqual(sum(entry['event'] == 'complete' for entry in entries), 2)

    def test_redirect_url_credentials_are_not_logged(self):
        self.prepare_download_failure()
        payload = json.dumps({'ID': 'synthetic', 'gtboxes': []}).encode()
        response = io.BytesIO(payload)
        response.headers = {'Content-Length': str(len(payload)), 'Content-Type': 'application/octet-stream'}
        response.status = 200
        response.geturl = lambda: 'https://storage.example/private?X-Amz-Signature=synthetic-secret'
        with patch.object(dataset.urllib.request, 'urlopen', return_value=response):
            dataset.download_huggingface()
        entry = next(item for item in self.download_log_entries() if item['event'] == 'response')
        self.assertEqual(entry['response_host'], 'storage.example')
        self.assertEqual(entry['http_status'], 200)
        self.assertNotIn('synthetic-secret', (self.base / 'logs/download.jsonl').read_text())

    def test_log_write_failure_is_reported(self):
        with patch.object(dataset, '_download_event', side_effect=OSError(28, 'No space left on device')):
            with self.assertRaisesRegex(RuntimeError, 'Could not write log.*No space left'):
                dataset.download_huggingface()

    def test_interruption_is_logged_and_preserved(self):
        self.prepare_download_failure()
        with patch.object(dataset.urllib.request, 'urlopen', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                dataset.download_huggingface()
        self.assertEqual(self.download_log_entries()[-1]['event'], 'interrupted')

    def test_invalid_configuration_logged_without_raw_input(self):
        with self.assertRaisesRegex(RuntimeError, 'Invalid Hugging Face'):
            dataset.download_huggingface('unsafe?token=synthetic-secret')
        entry = self.download_log_entries()[-1]
        self.assertEqual(entry['stage'], 'configuration')
        self.assertNotIn('synthetic-secret', json.dumps(entry))

    def test_progress_reports_existing_validation_and_six_reused_files(self):
        self.archives()
        events = []
        with patch.object(dataset.urllib.request, 'urlopen', side_effect=AssertionError('must reuse')):
            dataset.download_huggingface(progress=events.append)
        self.assertEqual([event['completed_files'] for event in events if event['stage'] == 'reused'], list(range(1, 7)))
        self.assertEqual(events[-1]['stage'], 'complete')
        self.assertEqual(events[-1]['completed_files'], 6)
        checks = [event for event in events if event['stage'] == 'validation' and event.get('finished')]
        self.assertEqual(len(checks), 6)
        self.assertTrue(all(event['processed'] == event['total'] for event in checks))
        self.assertEqual({event['unit'] for event in checks}, {'bytes', 'records'})

    def test_progress_reports_download_with_unknown_size(self):
        self.prepare_download_failure()
        payload = json.dumps({'ID': 'synthetic,001', 'gtboxes': []}).encode()
        response = io.BytesIO(payload)
        response.headers = {'Content-Type': 'application/octet-stream'}
        events = []
        with patch.object(dataset.urllib.request, 'urlopen', return_value=response):
            dataset.download_huggingface(progress=events.append)
        transfers = [event for event in events if event['stage'] == 'transfer']
        self.assertEqual(transfers[-1]['processed'], len(payload))
        self.assertEqual(transfers[-1]['total'], 0)
        self.assertTrue(transfers[-1]['finished'])
        self.assertEqual(events[-1]['completed_files'], 6)
        self.assertTrue(any(event['stage'] == 'file_complete' for event in events))

    def test_streamed_zip_validation_detects_bad_crc(self):
        self.archives()
        path = dataset.destination('CrowdHuman_train01.zip')
        with zipfile.ZipFile(path) as archive:
            entry = archive.infolist()[0]
            offset = entry.header_offset + 30 + len(entry.filename.encode()) + len(entry.extra)
        data = bytearray(path.read_bytes())
        data[offset] ^= 1
        path.write_bytes(data)
        with self.assertRaises(ValueError):
            dataset.validate_file(path, path.name, progress=lambda event: None)

    def test_console_progress_throttles_and_finishes(self):
        stream = io.StringIO()
        display = dataset.DownloadProgress(stream=stream, interval=1)
        common = {'stage': 'transfer', 'filename': 'CrowdHuman_val.zip', 'completed_files': 4,
                  'file_total': 6, 'total': 3 * 1024**2, 'unit': 'bytes'}
        with patch.object(dataset.time, 'monotonic', side_effect=[0, .25, 1.5, 2]):
            for index in range(4):
                display({**common, 'processed': index * 1024**2, 'finished': index == 3})
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn('[4/6 完了]', lines[-1])
        self.assertIn('100.0%', lines[-1])
        self.assertIn('MiB/s', lines[-1])
        self.assertIn('残り 00:00', lines[-1])

    def test_console_progress_handles_unknown_length(self):
        stream = io.StringIO()
        display = dataset.DownloadProgress(stream=stream)
        display({'stage': 'transfer', 'filename': 'annotation_val.odgt',
                 'processed': 1024, 'total': 0, 'finished': True})
        self.assertIn('-- %', stream.getvalue())
        self.assertIn('残り --:--', stream.getvalue())

    def test_notebook_progress_updates_one_display(self):
        handle, show = Mock(), Mock()
        show.return_value = handle
        ipython = SimpleNamespace(HTML=lambda text: text, display=show)
        display = dataset.DownloadProgress(notebook=True)
        with patch.dict('sys.modules', {'IPython.display': ipython}):
            display({'stage': 'transfer', 'filename': 'CrowdHuman_val.zip', 'processed': 0, 'total': 100})
            display({'stage': 'transfer', 'filename': 'CrowdHuman_val.zip', 'processed': 100, 'total': 100, 'finished': True})
        show.assert_called_once()
        handle.update.assert_called_once()
        self.assertIn('100.0%', handle.update.call_args.args[0])

    def prepare_drive_cache(self):
        self.archives()
        source_root = self.base / 'drive_cache'
        for name in dataset.ARCHIVES + dataset.ANNOTATIONS:
            original = dataset.destination(name)
            target = source_root / original.relative_to(self.base)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original.read_bytes())
            original.unlink()
        return source_root

    def test_drive_copy_and_reuse_without_network(self):
        source = self.prepare_drive_cache()
        originals = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
        events = []
        with patch.object(dataset.urllib.request, 'urlopen', side_effect=AssertionError('copy must not use network')):
            log = dataset.copy_drive_dataset(source, progress=events.append)
            timestamps = {name: dataset.destination(name).stat().st_mtime_ns for name in dataset.ARCHIVES + dataset.ANNOTATIONS}
            dataset.copy_drive_dataset(source)
        self.assertEqual(log, self.base / 'logs/drive_copy.jsonl')
        manifest = json.loads((self.base / 'raw/drive_copy_source.json').read_text())
        self.assertEqual(len(manifest['files']), 6)
        self.assertTrue(all(row['action'] == 'reused' for row in manifest['files']))
        for relative, data in originals.items():
            self.assertEqual((source / relative).read_bytes(), data)
            self.assertEqual((self.base / relative).read_bytes(), data)
        self.assertEqual({name: dataset.destination(name).stat().st_mtime_ns for name in timestamps}, timestamps)
        self.assertTrue({'copy', 'checksum', 'validation', 'complete'} <= {event['stage'] for event in events})
        self.assertEqual(events[-1]['completed_files'], 6)

    def test_drive_copy_missing_file_does_not_partially_copy(self):
        source = self.prepare_drive_cache()
        (source / 'raw/annotations/annotation_val.odgt').unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing Drive cache file'):
            dataset.copy_drive_dataset(source)
        self.assertFalse(any(dataset.destination(name).exists() for name in dataset.ARCHIVES + dataset.ANNOTATIONS))

    def test_drive_copy_corruption_leaves_no_final_or_temporary_file(self):
        source = self.prepare_drive_cache()
        archive = source / 'raw/archives/CrowdHuman_train01.zip'
        archive.write_bytes(b'synthetic-invalid-archive')
        with self.assertRaisesRegex(RuntimeError, 'Invalid ZIP'):
            dataset.copy_drive_dataset(source)
        self.assertEqual(archive.read_bytes(), b'synthetic-invalid-archive')
        self.assertFalse(dataset.destination(archive.name).exists())
        self.assertFalse(list((self.base / 'raw').rglob('*.copying')))
        entry = json.loads((self.base / 'logs/drive_copy.jsonl').read_text().splitlines()[-1])
        self.assertEqual(entry['stage'], 'copied_file_validation')

    def test_drive_copy_different_local_file_is_not_overwritten(self):
        source = self.prepare_drive_cache()
        target = dataset.destination('CrowdHuman_train01.zip')
        with zipfile.ZipFile(target, 'w') as archive:
            archive.writestr('Images/synthetic_other.jpg', b'synthetic-other-image')
        before = target.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'differs from Drive cache'):
            dataset.copy_drive_dataset(source)
        self.assertEqual(target.read_bytes(), before)

    def test_drive_copy_disk_space_check(self):
        source = self.prepare_drive_cache()
        with patch.object(dataset.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(RuntimeError, 'Insufficient runtime space'):
                dataset.copy_drive_dataset(source)
        self.assertFalse(list((self.base / 'raw').rglob('*.copying')))

    def test_drive_copy_rejects_source_symlink_escape(self):
        source = self.prepare_drive_cache()
        target = source / 'raw/archives/CrowdHuman_train01.zip'
        target.unlink()
        outside = self.base / 'outside.zip'
        outside.write_bytes(b'synthetic-outside')
        target.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, 'escapes its dataset directory'):
            dataset.copy_drive_dataset(source)

    def test_drive_copy_reports_checksum_mismatch(self):
        source = self.prepare_drive_cache()
        with patch.object(dataset, '_copy_checksum', return_value='incorrect-hash'):
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                dataset.copy_drive_dataset(source)
        self.assertFalse(list((self.base / 'raw').rglob('*.copying')))

    def test_runtime_boundary_and_symlink(self):
        with self.assertRaises(ValueError):
            privacy.runtime_path(self.base.parent / 'outside')
        (self.base / 'escape').symlink_to(self.base.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            privacy.runtime_path(self.base / 'escape/file')

    def test_explicit_local_download_storage_is_scoped(self):
        root = self.base / 'crowdhuman'
        with privacy.local_download_storage(root):
            for relative in ('raw/archives', 'raw/annotations', 'logs'):
                (root / relative).mkdir(parents=True)
            self.archives()
            self.write_records()
            with patch.object(dataset.urllib.request, 'urlopen', side_effect=AssertionError('no network expected')):
                log = dataset.download_huggingface()
            self.assertEqual(log, root / 'logs/download.jsonl')
            self.assertTrue(log.is_file())
            self.assertEqual(dataset.destination('annotation_train.odgt'), root / 'raw/annotations/annotation_train.odgt')
            with self.assertRaises(ValueError):
                privacy.runtime_path(self.base / 'outside')
        self.assertEqual(privacy.storage_root(), self.base)
        self.assertEqual(dataset.destination('annotation_train.odgt'), self.base / 'raw/annotations/annotation_train.odgt')

    def test_local_download_scope_restored_after_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            with privacy.local_download_storage(self.base / 'crowdhuman'):
                raise RuntimeError('synthetic failure')
        self.assertEqual(privacy.storage_root(), self.base)

    def test_local_download_scope_rejects_symlink(self):
        link = self.base / 'local_symlink'
        link.symlink_to(self.base.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            with privacy.local_download_storage(link):
                self.fail('symlink destination must be rejected')

    def test_logs_are_local(self):
        logger = logging.getLogger('ultralytics')
        handler = logging.StreamHandler()
        with patch.object(logger, 'handlers', [handler]), patch.object(logger, 'level', logging.INFO):
            with privacy.local_console(self.base / 'console.log'):
                print('synthetic progress')
                logger.info('synthetic metric')
            self.assertEqual(logger.handlers, [handler])
        text = (self.base / 'console.log').read_text()
        self.assertIn('synthetic progress', text)
        self.assertIn('synthetic metric', text)

    def make_evaluation(self, mode, batch=16):
        config = trainer.TrainConfig(mode=mode, batch=batch)
        run = config.run_dir
        (run / 'weights').mkdir(parents=True)
        best = run / 'weights/best.pt'
        best.write_bytes(b'synthetic-not-a-model')
        record = {'mode': mode, 'run_name': 'experiment', 'weight_sha256': evaluator.sha256(best),
                  'precision': .5, 'recall': .6, 'map50': .4, 'map50_95': .3,
                  'training_seconds': 2, 'inference_ms_per_image': 1, 'model_mib': .001,
                  'device': 'synthetic-device', 'classes': [], 'evaluation': {'batch': 1},
                  'specification': {'config': __import__('dataclasses').asdict(config),
                                    'dataset_fingerprint': 'synthetic', 'versions': {}}}
        (run / 'evaluation.json').write_text(json.dumps(record))
        return best

    def test_comparison_matching_conditions(self):
        self.make_evaluation('vbox')
        self.make_evaluation('hbox')
        rows = evaluator.comparison(self.base / 'exports/comparison.csv')
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['matched_conditions'] for row in rows))

    def test_comparison_mismatched_conditions(self):
        self.make_evaluation('vbox')
        self.make_evaluation('hbox', batch=8)
        rows = evaluator.comparison(self.base / 'exports/comparison.csv')
        self.assertFalse(any(row['matched_conditions'] for row in rows))

    def test_comparison_can_select_current_experiment(self):
        self.make_evaluation('vbox')
        stale = self.make_evaluation('hbox')
        stale.write_bytes(b'changed-synthetic')
        rows = evaluator.comparison(self.base / 'exports/comparison.csv', configs=[trainer.TrainConfig()])
        self.assertEqual({row['mode'] for row in rows}, {'vbox'})

    def test_history_uses_epoch_numbers_and_ignores_nonfinite_scores(self):
        best = self.make_evaluation('vbox')
        (best.parent.parent / 'results.csv').write_text(
            ' epoch, metrics/mAP50-95(B), train/box_loss\n'
            ' 5, 0.4, 0.5\n 6, nan, 0.4\n 7, 0.3, 0.2\n')
        summary, rows = evaluator.training_history(trainer.TrainConfig())
        self.assertEqual(summary['epochs_run'], 3)
        self.assertEqual(summary['last_epoch'], 7)
        self.assertEqual(summary['best_epoch_by_map50_95'], 5)
        self.assertEqual(summary['best_map50_95'], .4)
        self.assertEqual(rows[-1]['train/box_loss'], '0.2')

    def test_available_exports_selects_only_current_runs_and_enabled_formats(self):
        for name in ('vbox_experiment.pt', 'vbox_experiment.onnx', 'vbox_experiment_training.csv',
                     'vbox_experiment_old.pt', 'hbox_other.pt', 'comparison.csv', 'training_results_old.txt'):
            (self.base / 'exports' / name).touch()
        self.assertEqual(evaluator.available_exports([trainer.TrainConfig()], include_onnx=False),
                         ['vbox_experiment.pt', 'vbox_experiment_training.csv', 'comparison.csv'])

    def test_stale_evaluation(self):
        best = self.make_evaluation('vbox')
        best.write_bytes(b'changed-synthetic')
        with self.assertRaisesRegex(ValueError, 'Stale'):
            evaluator.comparison(self.base / 'exports/comparison.csv')

    def test_results_txt_contains_selected_runs_and_japan_timestamp(self):
        from datetime import datetime, timezone
        for mode in ('vbox', 'hbox'):
            best = self.make_evaluation(mode)
            (best.parent.parent / 'results.csv').write_text('epoch, train/box_loss\n1, 0.25\n')
        instant = datetime(2026, 11, 5, 18, 4, 3, 123456, tzinfo=timezone.utc)
        with patch.object(evaluator, 'datetime') as clock:
            clock.now.side_effect = lambda tz: instant.astimezone(tz)
            name = evaluator.export_results_txt([trainer.TrainConfig(mode=mode) for mode in ('vbox', 'hbox')])
        self.assertEqual(name, 'training_results_20261106_030403_123456.txt')
        text = (self.base / 'exports' / name).read_text(encoding='utf-8')
        self.assertIn('モード: vbox', text)
        self.assertIn('モード: hbox', text)
        self.assertIn('map50: 0.4', text)
        self.assertIn('epoch\ttrain/box_loss\n1\t0.25', text)
        files = SimpleNamespace(download=Mock())
        with patch.dict('sys.modules', {'google.colab': SimpleNamespace(files=files)}):
            evaluator.download_selected([name])
        files.download.assert_called_once_with(str(self.base / 'exports' / name))

    def test_results_txt_rejects_stale_or_missing_evaluation(self):
        config = trainer.TrainConfig()
        with self.assertRaises(FileNotFoundError):
            evaluator.export_results_txt([config])
        best = self.make_evaluation('vbox')
        best.write_bytes(b'changed-synthetic')
        with self.assertRaisesRegex(ValueError, 'Stale'):
            evaluator.export_results_txt([config])

    def test_resume_and_checkpoint_guards(self):
        sizes, _ = dataset.inspect_dataset()
        converter.convert_dataset('vbox', sizes)
        config = trainer.TrainConfig(epochs=2, batch=2, imgsz=64, allow_cpu=True)
        model = Mock()
        torch = SimpleNamespace(manual_seed=Mock(), cuda=SimpleNamespace(is_available=lambda: False),
                                load=Mock(return_value={'epoch': 0, 'optimizer': {'synthetic': True}}))
        numpy = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules', torch=torch, numpy=numpy))
            stack.enter_context(patch.object(trainer, 'device_for', return_value='cpu'))
            stack.enter_context(patch.object(trainer, 'versions', return_value={'ultralytics': 'synthetic'}))
            stack.enter_context(patch.object(trainer, 'load_model', return_value=model))
            factory = stack.enter_context(patch.object(trainer, 'private_yolo', return_value=Mock(return_value=model)))
            run = trainer.train(config)
            self.assertFalse(model.train.call_args.kwargs['plots'])
            self.assertFalse(model.train.call_args.kwargs['single_cls'])
            self.assertEqual(model.train.call_args.kwargs['seed'], 42)
            with self.assertRaises(FileExistsError):
                trainer.train(config)
            with self.assertRaises(FileNotFoundError):
                trainer.train(config, resume=True)
            (run / 'weights').mkdir()
            (run / 'weights/last.pt').write_bytes(b'synthetic-placeholder-only')
            trainer.train(config, resume=True)
            self.assertTrue(model.train.call_args.kwargs['resume'])
            self.assertEqual(len(json.loads((run / 'experiment.json').read_text())['sessions']), 2)
            changed = trainer.TrainConfig(epochs=3, batch=2, imgsz=64, allow_cpu=True)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                trainer.train(changed, resume=True)
            torch.load.return_value = {'epoch': -1, 'optimizer': None}
            with self.assertRaisesRegex(ValueError, 'cannot resume'):
                trainer.train(config, resume=True)

    def test_train_restores_drive_snapshot_after_local_run_loss(self):
        sizes, _ = dataset.inspect_dataset()
        converter.convert_dataset('vbox', sizes)
        config = trainer.TrainConfig(epochs=3, batch=2, imgsz=64, allow_cpu=True)
        model = Mock()
        torch = SimpleNamespace(manual_seed=Mock(), cuda=SimpleNamespace(is_available=lambda: False),
                                load=Mock(return_value={'epoch': 0, 'optimizer': {'synthetic': True}}))
        numpy = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        drive = self.base / 'private_drive/checkpoints'
        def interrupt_after_epoch(**kwargs):
            (config.run_dir / 'weights').mkdir()
            (config.run_dir / 'weights/last.pt').write_bytes(b'synthetic-checkpoint-not-a-model')
            callback = model.add_callback.call_args.args[1]
            callback(SimpleNamespace(epoch=0))
            raise KeyboardInterrupt()
        model.train.side_effect = interrupt_after_epoch
        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules', torch=torch, numpy=numpy))
            stack.enter_context(patch.object(trainer, 'device_for', return_value='cpu'))
            stack.enter_context(patch.object(trainer, 'versions', return_value={'ultralytics': 'synthetic'}))
            stack.enter_context(patch.object(trainer, 'load_model', return_value=model))
            stack.enter_context(patch.object(trainer, 'private_yolo', return_value=Mock(return_value=model)))
            with self.assertRaises(KeyboardInterrupt):
                trainer.train(config, drive_runs_dir=drive)
            self.assertTrue((drive / 'vbox/experiment/latest.json').is_file())
            with self.assertRaisesRegex(FileExistsError, 'Drive experiment exists'):
                trainer.train(config, drive_runs_dir=drive)
            dataset.shutil.rmtree(config.run_dir)
            model.train.side_effect = None
            trainer.train(config, resume=True, drive_runs_dir=drive)
            self.assertTrue(model.train.call_args.kwargs['resume'])
            meta = json.loads((config.run_dir / 'experiment.json').read_text())
            self.assertEqual([session['status'] for session in meta['sessions']], ['interrupted', 'completed'])
            self.assertEqual(model.add_callback.call_args.args[0], 'on_model_save')

    def test_drive_save_failure_stops_training_and_records_error(self):
        sizes, _ = dataset.inspect_dataset()
        converter.convert_dataset('vbox', sizes)
        config = trainer.TrainConfig(epochs=3, batch=2, imgsz=64, allow_cpu=True)
        model = Mock()
        torch = SimpleNamespace(manual_seed=Mock(), cuda=SimpleNamespace(is_available=lambda: False))
        numpy = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        def save_epoch(**kwargs):
            model.add_callback.call_args.args[1](SimpleNamespace(epoch=0))
        model.train.side_effect = save_epoch
        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules', torch=torch, numpy=numpy))
            stack.enter_context(patch.object(trainer, 'device_for', return_value='cpu'))
            stack.enter_context(patch.object(trainer, 'versions', return_value={}))
            stack.enter_context(patch.object(trainer, 'load_model', return_value=model))
            stack.enter_context(patch.object(trainer, 'private_yolo', return_value=Mock(return_value=model)))
            stack.enter_context(patch.object(trainer, 'save_drive_checkpoint', side_effect=OSError(28, 'synthetic Drive full')))
            with self.assertRaisesRegex(RuntimeError, 'training stopped'):
                trainer.train(config, drive_runs_dir=self.base / 'drive')
        failure = json.loads((config.run_dir / 'drive_backup_error.json').read_text())
        self.assertEqual(failure['epoch'], 1)
        self.assertEqual(failure['errors'][0]['errno'], 28)
        self.assertIn('Drive full', failure['errors'][0]['message'])
        model.train.assert_called_once()

    def test_initialization_rejects_escaped_directory(self):
        (self.base / 'config').symlink_to(self.base.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            privacy.initialize()

    def test_onnx_contract_and_unverified_status(self):
        image = SimpleNamespace(shape=(1, 3, 64, 64))
        output = SimpleNamespace(shape=(1, 300, 6))
        numpy = SimpleNamespace(float32='float32', zeros=Mock(return_value=image),
                                isfinite=Mock(return_value=SimpleNamespace(all=lambda: True)))
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name='images', type='tensor(float)', shape=['batch', 3, 'height', 'width'])]
        session.get_outputs.return_value = [SimpleNamespace(name='output0', type='tensor(float)')]
        session.run.return_value = [output]
        onnx = SimpleNamespace(checker=SimpleNamespace(check_model=Mock()), load=Mock())
        runtime = SimpleNamespace(InferenceSession=Mock(return_value=session))
        path = self.base / 'exports/synthetic.onnx'
        path.write_bytes(b'synthetic-placeholder-only')
        with patch.dict('sys.modules', numpy=numpy, onnx=onnx, onnxruntime=runtime):
            report = evaluator.inspect_onnx(path, 64)
            self.assertEqual(report['application_contract'], 'unverified')
            matched = evaluator.inspect_onnx(path, 64, report['interface'])
            self.assertEqual(matched['application_contract'], 'matched')
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                evaluator.inspect_onnx(path, 64, {'input': 'incompatible'})
            numpy.isfinite.return_value = SimpleNamespace(all=lambda: False)
            with self.assertRaisesRegex(ValueError, 'Invalid ONNX'):
                evaluator.inspect_onnx(path, 64)

    def test_download_allowlist_and_missing_export(self):
        files = SimpleNamespace(download=Mock())
        with patch.dict('sys.modules', {'google.colab': SimpleNamespace(files=files)}):
            for name in ('../raw/annotation_train.odgt', 'sample.jpg', 'sample.zip', '../weights/best.pt'):
                with self.assertRaises(ValueError):
                    evaluator.download_selected([name])
            with self.assertRaises(FileNotFoundError):
                evaluator.download_selected(['missing.pt'])
            (self.base / 'exports/synthetic.pt').write_bytes(b'synthetic-not-a-model')
            evaluator.download_selected(['synthetic.pt'])
            files.download.assert_called_once_with(str(self.base / 'exports/synthetic.pt'))


class ProjectTests(unittest.TestCase):
    def test_invalid_training_configuration(self):
        for config in [trainer.TrainConfig(mode='fbox'), trainer.TrainConfig(batch=0), trainer.TrainConfig(imgsz=641), trainer.TrainConfig(run_name='../x'), trainer.TrainConfig(model='untrusted.pt')]:
            with self.assertRaises(ValueError):
                config.validate()

    def test_ignore_rules(self):
        # Build a fresh synthetic Git repo so this check also works inside Colab.
        with tempfile.TemporaryDirectory(prefix='synthetic-policy-') as directory:
            root = Path(directory)
            (root / '.gitignore').write_text((PROJECT / '.gitignore').read_text())
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            for name in ('CrowdHuman/a.txt', 'crowdhuman/a.txt', 'datasets/a', 'data/a', 'sample.odgt', 'sample.zip', 'a.jpg', 'a.jpeg', 'a.png', 'a.webp', 'runs/a.txt', 'weights/a', 'checkpoints/a', 'a.pt', 'a.pth', 'a.onnx', '.env', '.env.local', 'a.key', '__pycache__/a', '.ipynb_checkpoints/a', 'logs/download.jsonl', 'console.log'):
                with self.subTest(name=name):
                    result = subprocess.run(['git', '-C', str(root), 'check-ignore', '-q', '--', name])
                    self.assertEqual(result.returncode, 0)
            self.assertEqual(subprocess.run(['git', '-C', str(root), 'check-ignore', '-q', 'src/converter.py']).returncode, 1)

    def test_notebook_has_no_outputs_and_compiles(self):
        path = PROJECT / 'crowdhuman_finetuning.ipynb'
        if not path.exists():
            self.skipTest('Notebook structure checked in development repository only')
        notebook = json.loads(path.read_text())
        self.assertTrue(notebook['metadata']['colab']['private_outputs'])
        headings = [c for c in notebook['cells'] if c['cell_type'] == 'markdown']
        self.assertEqual(len(headings), 10)
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
                compile(''.join(cell['source']), f'cell-{index}', 'exec')

    def test_source_privacy_policy(self):
        for path in (PROJECT / 'src').glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, ('push_to_hub', 'mount', 'imshow', 'show'))
        text = (PROJECT / 'src/trainer.py').read_text()
        self.assertIn('plots=False', text)
        self.assertIn('save_json=False', text)
        privacy_text = (PROJECT / 'src/privacy.py').read_text()
        self.assertIn('add_integration_callbacks = lambda instance: None', privacy_text)


if __name__ == '__main__':
    unittest.main()
