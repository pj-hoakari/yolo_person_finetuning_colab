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
        self.assertEqual(len(headings), 19)

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
