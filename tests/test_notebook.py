"""Execute standalone notebook definitions without src imports or code extraction."""
import ast
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class StandaloneNotebookTests(unittest.TestCase):
    def test_readable_self_contained_cells(self):
        notebook = json.loads((ROOT / 'crowdhuman_finetuning.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            self.assertNotIn('MODULE_SOURCES', source)
            self.assertNotIn('CODE_DIR', source)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    self.assertFalse((node.module or '').startswith('src'))
        headings = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'markdown']
        self.assertEqual(len(headings), 18)

    def test_drive_cache_is_copied_to_content_before_processing(self):
        notebook = json.loads((ROOT / 'crowdhuman_finetuning.ipynb').read_text())
        sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        config = next(source for source in sources if 'DRIVE_DATASET_DIR =' in source)
        self.assertIn("'/content/drive/MyDrive/crowdhuman'", config)
        acquisition = next(source for source in sources if 'def mount_dataset_drive' in source)
        self.assertIn('drive.mount(mount_point)', acquisition)
        self.assertIn('with local_download_storage(drive_dataset):', acquisition)
        self.assertIn('copy_drive_dataset(drive_dataset', acquisition)
        self.assertLess(acquisition.rindex('download_huggingface(HF_DATASET_REPO'), acquisition.rindex('copy_drive_dataset(drive_dataset'))
        cleanup = sources[-1]
        self.assertNotIn('DRIVE_DATASET_DIR', cleanup)
        self.assertNotIn('drive_dataset', cleanup)

    def test_drive_epoch_backup_is_enabled_and_checkpoint_code_is_inline(self):
        notebook = json.loads((ROOT / 'crowdhuman_finetuning.ipynb').read_text())
        sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        self.assertTrue(any('SAVE_EVERY_EPOCH_TO_DRIVE = True' in source for source in sources))
        self.assertTrue(any('def restore_drive_checkpoint' in source for source in sources))
        training = next(source for source in sources if 'trained_runs = {}' in source)
        self.assertIn('drive_runs_dir=DRIVE_CHECKPOINT_DIR', training)
        self.assertIn('checkpoint_status=show_saved_epoch', training)

    def test_step_12_can_restore_without_steps_6_to_11(self):
        notebook = json.loads((ROOT / 'crowdhuman_finetuning.ipynb').read_text())
        heading = next(i for i, cell in enumerate(notebook['cells'])
                       if ''.join(cell['source']).startswith('## 12.'))
        source = ''.join(notebook['cells'][heading + 1]['source'])
        definitions = {node.name for node in ast.parse(source).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        self.assertTrue({'restore_prepared_data', 'validate_labels', 'read_odgt',
                         'TrainConfig', 'load_model'}.issubset(definitions))
        self.assertIn('if RESTORE_PREPARED_FROM_DRIVE:', source)
        self.assertIn('drive.mount(DRIVE_MOUNT_POINT)', source)
        self.assertIn('yaml_paths = restore_prepared_data(', source)
        self.assertNotIn('copy_drive_dataset(drive_dataset', source)
        previous = ''.join(notebook['cells'][heading - 1]['source'])
        self.assertIn('prepared_archive = save_prepared_data(', previous)

    def test_inline_synthetic_suite(self):
        notebook = json.loads((ROOT / 'crowdhuman_finetuning.ipynb').read_text())
        # Execute only imports, definitions and literal constants. No Colab setup,
        # dataset download, training, real storage initialization, or external API.
        nodes = []
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(''.join(cell['source']))
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                    nodes.append(node)
                elif isinstance(node, ast.Assign):
                    if (isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Name)
                            and node.value.func.id == 'ContextVar'):
                        nodes.append(node)
                        continue
                    try:
                        ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        continue
                    nodes.append(node)
        module = types.ModuleType('standalone_crowdhuman_notebook')
        module.BASE = Path('/content/crowdhuman')
        with patch.dict(sys.modules, {module.__name__: module}):
            exec(compile(ast.Module(body=nodes, type_ignores=[]), '<notebook-definitions>', 'exec'), module.__dict__)
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.NotebookSyntheticTests)
            result = unittest.TestResult()
            suite.run(result)
            self.assertEqual(result.testsRun, 12)
            self.assertTrue(result.wasSuccessful(), str(result.errors + result.failures))
            self.assertEqual(module.BASE, Path('/content/crowdhuman'))
