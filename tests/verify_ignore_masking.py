"""ignore領域の黒塗りを合成画像・ZIPで検証。実データ取得や学習は行いません。"""
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from PIL import Image


class IgnoreMaskTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        notebook = json.loads((Path(__file__).resolve().parents[1] / 'yolo_person_finetuning.ipynb').read_text())
        source = ''.join(next(c for c in notebook['cells'] if c.get('id') == 'dataset-prepare')['source'])
        definitions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
        self.ns = dict(Path=Path, json=json, hashlib=hashlib, math=math, shutil=shutil,
                       tempfile=tempfile, zipfile=zipfile, Counter=Counter, Image=Image,
                       ThreadPoolExecutor=ThreadPoolExecutor, as_completed=as_completed,
                       tqdm=lambda iterable, **kwargs: iterable)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<prepare>', 'exec'), self.ns)

    def test_head_ignore_keeps_person_label(self):
        record = {'gtboxes': [{'tag': 'person', 'vbox': [0, 0, 32, 32],
                              'hbox': [8, 0, 16, 8], 'head_attr': {'ignore': 1}}]}
        regions, counts = self.ns['ignored_regions'](record, 64, 64)
        self.assertEqual(regions, [(8, 0, 24, 8)])
        self.assertEqual(counts['masked_head_regions'], 1)
        lines, _ = self.ns['labels_for_record'](record, 64, 64)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith('0 '))

    def test_person_mask_tag_and_invalid_boxes(self):
        record = {'gtboxes': [
            {'tag': 'person', 'extra': {'ignore': 1}, 'fbox': [-2.5, 1.5, 10, 10]},
            {'tag': 'mask', 'fbox': [60, 60, 20, 20]},
            {'tag': 'person', 'extra': {'ignore': 1}, 'fbox': [0, 0, 0, 0],
             'vbox': [10, 10, 8, 8], 'hbox': [12, 8, 4, 4]},
            {'tag': 'person', 'head_attr': {'ignore': 1}, 'hbox': [float('nan'), 0, 2, 2]},
            {'tag': 'person', 'extra': {'ignore': 1}, 'fbox': [100, 100, 5, 5]},
        ]}
        regions, counts = self.ns['ignored_regions'](record, 64, 64)
        self.assertEqual(regions, [(0, 1, 8, 12), (60, 60, 64, 64), (10, 10, 18, 18), (12, 8, 16, 12)])
        self.assertEqual(counts['invalid_ignore_boxes'], 2)
        self.assertEqual(counts['masked_regions'], 4)

    def test_no_ignore_preserves_original_image_bytes(self):
        source = self.root / 'image.jpg'
        Image.new('RGB', (64, 64), (200, 200, 200)).save(source)
        before = source.read_bytes()
        width, height, counts = self.ns['prepare_image'](source, source, {'gtboxes': []})
        self.assertEqual((width, height), (64, 64))
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(counts['masked_images'], 0)

    def test_invalid_ignore_box_still_outputs_original_image(self):
        source, target = self.root / 'original.jpg', self.root / 'output.jpg'
        Image.new('RGB', (64, 64), (200, 200, 200)).save(source)
        record = {'gtboxes': [{'tag': 'person', 'extra': {'ignore': 1},
                              'fbox': [0, 0, -1, 4]}]}
        _, _, counts = self.ns['prepare_image'](source, target, record)
        self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertEqual(counts['invalid_ignore_boxes'], 1)
        self.assertEqual(counts['masked_images'], 0)

    def dataset(self):
        data = self.root / 'drive'
        annotations = data / 'raw/annotations'
        archives = data / 'raw/archives'
        annotations.mkdir(parents=True)
        archives.mkdir()
        objects = [
            {'tag': 'person', 'vbox': [48, 48, 16, 16], 'hbox': [48, 48, 8, 8]},
            {'tag': 'person', 'extra': {'ignore': 1}, 'fbox': [0, 0, 16, 16],
             'vbox': [4, 4, 8, 8], 'hbox': [4, 4, 4, 4]},
            {'tag': 'person', 'head_attr': {'ignore': 1},
             'vbox': [32, 32, 16, 32], 'hbox': [32, 32, 16, 16]},
        ]
        for split in ('train', 'val'):
            records = [{'ID': split + '_masked', 'gtboxes': objects}]
            if split == 'train':
                records.append({'ID': 'train_plain', 'gtboxes': [objects[0]]})
            (annotations / f'annotation_{split}.odgt').write_text(
                ''.join(json.dumps(record) + '\n' for record in records))
        buffer = io.BytesIO()
        Image.new('RGB', (64, 64), (200, 200, 200)).save(buffer, format='JPEG')
        original = buffer.getvalue()
        for name, ids in [('CrowdHuman_train01.zip', ['train_masked', 'train_plain']),
                          ('CrowdHuman_train02.zip', []), ('CrowdHuman_train03.zip', []),
                          ('CrowdHuman_val.zip', ['val_masked'])]:
            with zipfile.ZipFile(archives / name, 'w') as archive:
                for image_id in ids:
                    archive.writestr(f'Images/{image_id}.jpg', original)
        return data, original

    def test_pipeline_masks_both_splits_and_reuses_originals(self):
        data, original = self.dataset()
        before = {path: path.read_bytes() for path in data.rglob('*') if path.is_file()}
        output = self.root / 'prepared'
        self.ns['prepare_dataset'](data, output)
        for split in ('train', 'val'):
            with Image.open(output / 'images' / split / f'{split}_masked.jpg') as image:
                self.assertEqual(image.getpixel((8, 8)), (0, 0, 0))
                self.assertEqual(image.getpixel((40, 40)), (0, 0, 0))
                self.assertEqual(image.getpixel((56, 56)), (200, 200, 200))
            self.assertEqual((output / '.original_images' / split / f'{split}_masked.jpg').read_bytes(), original)
            lines = (output / 'labels' / split / f'{split}_masked.txt').read_text().splitlines()
            self.assertEqual([line.split()[0] for line in lines], ['0', '1', '0'])
        self.assertEqual((output / 'images/train/train_plain.jpg').read_bytes(), original)
        report = json.loads((output / 'preparation_report.json').read_text())
        self.assertEqual(report['train']['masked_images'], 1)
        self.assertEqual(report['val']['masked_regions'], 2)
        first = {path: path.read_bytes() for path in (output / 'images').rglob('*.jpg')}
        self.ns['prepare_dataset'](data, output)
        self.assertEqual(first, {path: path.read_bytes() for path in (output / 'images').rglob('*.jpg')})
        self.assertEqual(before, {path: path.read_bytes() for path in data.rglob('*') if path.is_file()})

    def test_old_preparation_config_is_rejected(self):
        data, _ = self.dataset()
        output = self.root / 'prepared'
        output.mkdir()
        (output / '.build_config.json').write_text('{"schema":1}')
        with self.assertRaisesRegex(RuntimeError, '別設定'):
            self.ns['prepare_dataset'](data, output)


if __name__ == '__main__':
    unittest.main()
