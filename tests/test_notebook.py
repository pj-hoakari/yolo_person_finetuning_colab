"""Execute standalone notebook definitions without src imports or code extraction."""
import ast
import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
        self.assertEqual(len(headings), 10)

    def test_setup_defines_workflow_before_preparing_data(self):
        notebook = json.loads((ROOT / 'crowdhuman_finetuning.ipynb').read_text())
        sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        config = next(source for source in sources if 'DRIVE_DATASET_DIR =' in source)
        self.assertIn("'/content/drive/MyDrive/crowdhuman'", config)
        definitions = next(source for source in sources if 'def prepare_training_data' in source)
        mount = next(source for source in sources if 'def mount_dataset_drive' in source)
        preparation = next(source for source in sources if source.startswith('yaml_paths = prepare_training_data('))
        self.assertLess(sources.index(definitions), sources.index(mount))
        self.assertLess(sources.index(mount), sources.index(preparation))
        self.assertIn('restore=RESTORE_PREPARED_FROM_DRIVE', preparation)
        self.assertNotIn('DRIVE_DATASET_DIR', sources[-1])

    def test_drive_epoch_backup_is_enabled_and_checkpoint_code_is_inline(self):
        notebook = json.loads((ROOT / 'crowdhuman_finetuning.ipynb').read_text())
        sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        self.assertTrue(any('SAVE_EVERY_EPOCH_TO_DRIVE = True' in source for source in sources))
        self.assertTrue(any('def restore_drive_checkpoint' in source for source in sources))
        training = next(source for source in sources if 'trained_runs = {}' in source)
        self.assertIn('drive_runs_dir=DRIVE_CHECKPOINT_DIR', training)
        self.assertIn('checkpoint_status=show_saved_epoch', training)

    def test_all_runtime_definitions_are_available_after_setup(self):
        notebook = json.loads((ROOT / 'crowdhuman_finetuning.ipynb').read_text())
        definitions = set()
        for cell in notebook['cells']:
            source = ''.join(cell['source'])
            if source.startswith('## 4.'):
                break
            if cell['cell_type'] == 'code':
                definitions.update(node.name for node in ast.parse(source).body
                                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
        self.assertTrue({'restore_prepared_data', 'prepare_training_data', 'validate_labels',
                         'read_odgt', 'TrainConfig', 'load_model', 'export_results_txt',
                         'available_exports'}.issubset(definitions))

    def test_user_cells_follow_selected_settings_and_download_without_filenames(self):
        notebook = json.loads((ROOT / 'crowdhuman_finetuning.ipynb').read_text())
        steps = {}
        for cell in notebook['cells']:
            source = ''.join(cell['source'])
            if cell['cell_type'] == 'markdown':
                step = int(source.split('. ')[0].split()[-1])
                steps[step] = []
            else:
                steps[step].append(source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SimpleNamespace(mode='vbox', run_dir=root / 'runs/vbox/experiment')
            prepare = Mock(return_value={'vbox': root / 'processed/vbox/data.yaml'})
            download = Mock()
            report = Mock(return_value='training_results_20261109_120000_000000.txt')
            env = dict(Path=Path, BASE=root, configs=[config], print=Mock(),
                       local_console=lambda _: nullcontext(), DownloadProgress=Mock(),
                       prepare_training_data=prepare, validate_labels=Mock(), load_model=Mock(),
                       train=Mock(return_value=config.run_dir),
                       training_history=Mock(return_value=({'epochs_run': 1}, [])),
                       plot_training_history=Mock(), evaluate=Mock(return_value={'map50': .5}),
                       comparison=Mock(return_value=[{'matched_conditions': True}]),
                       export_model=Mock(return_value=['vbox_experiment.pt']),
                       available_exports=Mock(side_effect=lambda *a, **k: ['vbox_experiment.pt']),
                       export_results_txt=report, download_selected=download,
                       runtime_path=lambda p: p)
            exec(steps[2][0], env)
            env.update(EXPORT_DIR=str(root), RESTORE_PREPARED_FROM_DRIVE=True)
            # Execute the user flow with external operations mocked, in notebook order.
            exec(steps[4][-1], env)
            self.assertTrue(prepare.call_args.kwargs['restore'])
            self.assertEqual(prepare.call_args.kwargs['modes'], ['vbox'])
            for step in range(5, 10):
                for source in steps[step]:
                    exec(source, env)
            env['load_model'].assert_called_once_with(config)
            env['plot_training_history'].assert_called_once_with(config)
            env['comparison'].assert_called_once_with(output=root / 'comparison.csv', configs=[config])
            report.assert_not_called()
            download.assert_not_called()
            env['DO_DOWNLOAD'] = True
            exec(steps[9][0], env)
            download.assert_called_once_with(['vbox_experiment.pt', report.return_value])
            env['RESUME'] = True
            env['load_model'].reset_mock()
            exec(steps[5][0], env)
            env['load_model'].assert_not_called()
            exec(steps[10][0], env)
            self.assertFalse(env['DELETE_DATA'])
            self.assertFalse(env['DELETE_RUNS'])

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
