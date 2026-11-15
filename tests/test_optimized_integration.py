"""检查优化版接入点和当前参数约定，不训练模型。"""
import unittest
from pathlib import Path
from src.config import parser, METHODS
from src.author_paths import ROOT, source_directory
from src.author_execution import method_module
from src.poison_budget import resolve_budget


class OptimizedIntegrationTests(unittest.TestCase):
    def test_new_methods_and_dispatch(self):
        expected = {'abp_dual': 'src.abp_dual', 'abp_assisted': 'src.abp_assisted', 'abp_learn': 'src.evaluation'}
        for name, module in expected.items():
            self.assertIn(name, METHODS)
            self.assertEqual(method_module(name, 'UGBA').__name__, module)
            self.assertTrue(callable(method_module(name, 'UGBA').evaluate))

    def test_paths_use_new_directory(self):
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])
        for method in ['abp_dual', 'abp_assisted', 'abp_learn']:
            self.assertEqual(source_directory(method, 'UGBA'), ROOT / 'upstream/RIGBD')

    def test_current_classifier_defaults_are_preserved(self):
        args = parser('检查整合参数').parse_args([])
        self.assertEqual(args.hidden, 32)
        self.assertEqual(args.epochs, 200)
        self.assertIsNone(args.learn_recover_epochs)
        self.assertEqual(args.class_sampling, 'random')

    def test_rate_uses_full_non_test_training_graph(self):
        args = parser('检查预算').parse_args(['--rate', '3'])
        self.assertEqual(resolve_budget(args, 2167, 1356), 65)


if __name__ == '__main__':
    unittest.main()
