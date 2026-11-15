"""受控最终参数路由检查，不训练模型。"""
import ast
import unittest
from src.config import ROOT, parser
from src.author_parameters import get_parameters
from src.classifier_parameters import final_classifier_args


class AttachedIndices:
    def numel(self):
        return 3


class ClassifierParameterTests(unittest.TestCase):
    def parameters(self, method, attack):
        config = parser("受控参数检查").parse_args(["--attack", attack, "--device", "cpu"])
        inputs = {"metadata": {"settings": {"target": 0}}, "idx_attach": AttachedIndices()}
        return get_parameters(method, config, inputs, ROOT / "outputs/parameter_checks")[0]

    def test_common_defaults(self):
        config = parser("受控默认值").parse_args([])
        self.assertEqual((config.hidden, config.epochs, config.lr, config.weight_decay, config.dropout),
                         (32, 200, 0.01, 0.0005, 0.5))

    def test_every_author_final_view_is_controlled(self):
        for method in ["rigbd", "losplit", "mad", "prune", "dshield"]:
            for attack in ["GTA", "UGBA", "DPGBA"]:
                author = self.parameters(method, attack)
                final = final_classifier_args(author)
                self.assertEqual((final.hidden, final.epochs, final.train_lr, final.dropout, final.weight_decay),
                                 (32, 200, 0.01, 0.5, 0.0005), (method, attack))

    def test_rigbd_scoring_unchanged(self):
        args = self.parameters("rigbd", "UGBA")
        final = final_classifier_args(args)
        self.assertEqual((args.hidden, args.epochs, args.K, args.drop_ratio), (64, 1000, 20, 0.5))
        self.assertEqual((final.hidden, final.epochs), (32, 200))
        self.assertIsNot(args, final)

    def test_losplit_dpgba_auxiliary_learning_rate_unchanged(self):
        args = self.parameters("losplit", "DPGBA")
        self.assertEqual((args.hidden, args.epochs, args.split_epoch, args.gamma, args.train_lr),
                         (32, 200, 15, 1.0, 0.015))
        self.assertEqual(final_classifier_args(args).train_lr, 0.01)

    def test_prune_dpgba_author_base_is_preserved(self):
        args = self.parameters("prune", "DPGBA")
        self.assertEqual((args.hidden, args.epochs, args.prune_thr), (512, 200, 0.8))
        self.assertEqual(final_classifier_args(args).hidden, 32)

    def test_dshield_changes_only_requested_stage(self):
        for attack in ["GTA", "DPGBA"]:
            args = self.parameters("dshield", attack)
            self.assertEqual((args.benign_hidden, args.benign_epochs, args.dshield_pretrain_epochs,
                              args.dshield_classify_epochs, args.dshield_finetune_epochs, args.dshield_classify_rounds),
                             (32, 200, 1000, 200, 500, 100))
        args = self.parameters("dshield", "UGBA")
        self.assertEqual((args.dshield_pretrain_epochs, args.dshield_classify_epochs,
                          args.dshield_finetune_epochs, args.dshield_classify_rounds), (400, 200, 400, 1))

    def test_generated_final_recovery_uses_final_epochs(self):
        for filename in ["rigbd.py", "losplit_gta.py", "losplit_ugba.py", "losplit_dpgba.py"]:
            tree = ast.parse((ROOT / "src/author_methods" / filename).read_text(encoding="utf-8"))
            recovery_calls = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                recovery = any(keyword.arg in ["finetune", "finetune2"] and isinstance(keyword.value, ast.Constant)
                               and keyword.value.value is True for keyword in node.keywords)
                if recovery:
                    recovery_calls.append(node)
                    epochs = next(keyword.value for keyword in node.keywords if keyword.arg == "train_iters")
                    self.assertEqual(ast.unparse(epochs), "final_args.epochs", filename)
            self.assertEqual(len(recovery_calls), 1, filename)

    def test_mad_keeps_five_repeats_and_final_view(self):
        for attack in ["gta", "ugba", "dpgba"]:
            text = (ROOT / "src/author_methods" / ("mad_" + attack + ".py")).read_text(encoding="utf-8")
            self.assertIn("size=5", text.replace(" ", ""))
            self.assertIn("final_args = final_classifier_args(args)", text)
            self.assertIn("train_iters=final_args.epochs", text)


if __name__ == "__main__":
    unittest.main()
