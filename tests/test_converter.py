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

    def test_runtime_boundary_and_symlink(self):
        with self.assertRaises(ValueError):
            privacy.runtime_path(self.base.parent / 'outside')
        (self.base / 'escape').symlink_to(self.base.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            privacy.runtime_path(self.base / 'escape/file')

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

    def test_stale_evaluation(self):
        best = self.make_evaluation('vbox')
        best.write_bytes(b'changed-synthetic')
        with self.assertRaisesRegex(ValueError, 'Stale'):
            evaluator.comparison(self.base / 'exports/comparison.csv')

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
            for name in ('CrowdHuman/a.txt', 'crowdhuman/a.txt', 'datasets/a', 'data/a', 'sample.odgt', 'sample.zip', 'a.jpg', 'a.jpeg', 'a.png', 'a.webp', 'runs/a.txt', 'weights/a', 'checkpoints/a', 'a.pt', 'a.pth', 'a.onnx', '.env', '.env.local', 'a.key', '__pycache__/a', '.ipynb_checkpoints/a'):
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
        self.assertEqual(len(headings), 19)
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
                compile(''.join(cell['source']), f'cell-{index}', 'exec')
                self.assertEqual(notebook['cells'][index-1]['cell_type'], 'markdown')

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
