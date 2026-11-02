# この合成テストもノートブック内で直接実行します。実データは使いません。
import io
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image


class NotebookSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-notebook-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # 合成データに限り、一時ディレクトリを使って保存先を差し替えます。
        self.guard = patch.dict(globals(), BASE=self.root)
        self.guard.start()
        self.addCleanup(self.guard.stop)
        for relative in ('raw/annotations', 'raw/images', 'processed'):
            (self.root / relative).mkdir(parents=True)
        self.person = {'tag': 'person', 'vbox': [10, 20, 30, 40], 'hbox': [20, 20, 10, 10]}
        for split in ('train', 'val'):
            key = f'synthetic_{split}'
            Image.new('RGB', (100, 100)).save(self.root / 'raw/images' / f'{key}.jpg')
            record = {'ID': key, 'gtboxes': [self.person]}
            destination(f'annotation_{split}.odgt').write_text(json.dumps(record) + '\n')

    def test_vbox(self):
        rows, _ = convert_boxes([self.person], 'vbox', 100, 100)
        self.assertEqual(rows, [(0, .25, .4, .3, .4)])

    def test_hbox(self):
        rows, _ = convert_boxes([self.person], 'hbox', 100, 100)
        self.assertEqual(rows, [(0, .25, .25, .1, .1)])

    def test_independent_ignore(self):
        for body, head, expected in ((0, 0, [0, 1]), (1, 0, [1]), (0, 1, [0]), (1, 1, [])):
            annotation = dict(self.person, extra={'ignore': body}, head_attr={'ignore': head})
            rows, _ = convert_boxes([annotation], 'both', 100, 100)
            self.assertEqual([row[0] for row in rows], expected)

    def test_clipping(self):
        self.assertEqual(normalize_box([-20, -10, 150, 120], 100, 100), (.5, .5, 1, 1))

    def test_invalid_boxes(self):
        for box in (None, [], [0, 0, 0, 1], [0, 0, -1, 1], [101, 0, 5, 5], [0, 0, float('nan'), 1]):
            rows, stats = convert_boxes([dict(self.person, vbox=box)], 'vbox', 100, 100)
            self.assertEqual(rows, [])
            self.assertEqual(stats['vbox_invalid'], 1)

    def test_non_person(self):
        rows, _ = convert_boxes([dict(self.person, tag='mask')], 'both', 100, 100)
        self.assertEqual(rows, [])

    def test_class_ids(self):
        rows, _ = convert_boxes([self.person], 'both', 100, 100)
        self.assertEqual([row[0] for row in rows], [0, 1])
        self.assertEqual(NAMES['both'], {0: 'person', 1: 'head'})

    def test_split_separation(self):
        sizes, counts = inspect_dataset()
        root = convert_dataset('vbox', sizes).parent
        self.assertEqual(counts, {'train': 1, 'val': 1})
        self.assertTrue((root / 'images/train/synthetic_train.jpg').is_symlink())
        self.assertFalse((root / 'images/val/synthetic_train.jpg').exists())
        self.assertTrue((root / 'labels/val/synthetic_val.txt').exists())

    def test_yaml(self):
        import yaml
        sizes, _ = inspect_dataset()
        for mode in ('vbox', 'hbox', 'both'):
            path = convert_dataset(mode, sizes)
            data = yaml.safe_load(path.read_text())
            self.assertEqual(data['path'], str(path.parent))
            self.assertEqual(data['train'], 'images/train')
            self.assertEqual(data['val'], 'images/val')
            self.assertEqual(data['names'], NAMES[mode])
            self.assertEqual(validate_labels(path.parent, mode)['train']['images'], 1)

    def test_ignore_rules(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-git-policy-') as directory:
            root = Path(directory)
            (root / '.gitignore').write_text(GITIGNORE_CONTENT)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            for name in ('CrowdHuman/a', 'crowdhuman/a', 'datasets/a', 'data/a', 'a.odgt', 'a.zip',
                         'a.jpg', 'a.jpeg', 'a.png', 'a.webp', 'runs/a', 'weights/a', 'checkpoints/a',
                         'a.pt', 'a.pth', 'a.onnx', '.env', '.env.local', 'a.key', '__pycache__/a', '.ipynb_checkpoints/a'):
                self.assertEqual(subprocess.run(['git', '-C', str(root), 'check-ignore', '-q', '--', name]).returncode, 0)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            list(read_odgt(self.root / 'raw/annotations/missing.odgt'))
        with self.assertRaises(ValueError):
            image_size(self.root / 'raw/images/missing.jpg')

    def test_storage_boundary(self):
        with self.assertRaises(ValueError):
            runtime_path(self.root.parent / 'outside')
