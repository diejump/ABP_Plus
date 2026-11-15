"""核对攻击训练接口的数据隔离与编号映射；不训练真实模型。"""
import unittest
from types import SimpleNamespace
import torch
from src.attack_inputs import compact_training_graph, local_positions, fit_and_poison, restore_global_graph
from src.attack import attack_labels_for_generator
from src.data import normalize_from_training_graph


def example():
    data = {"x": torch.arange(18, dtype=torch.float).reshape(6, 3),
            "edges": torch.tensor([[0, 2, 2, 3, 3, 5], [2, 0, 3, 2, 5, 3]]),
            "weights": torch.ones(6), "labels": torch.tensor([0, -1, 1, -1, -1, -1]),
            "train": torch.tensor([0, 2]), "val": torch.tensor([5]), "classes": 2}
    nodes = torch.tensor([0, 2, 3, 5])
    labels = torch.tensor([0, 1, 1, 0])
    return data, nodes, labels


class RecordingGenerator:
    def __init__(self):
        self.trojan = torch.nn.Linear(3, 3)
        self.events = []
        self.fit_input = None

    def fit(self, x, edges, weights, labels, train, attached, available):
        self.events.append('fit')
        self.fit_input = [value.clone() for value in [x, edges, weights, labels, train, attached, available]]

    def get_poisoned(self, x, edges, weights, labels, attached):
        self.events.append('get_poisoned')
        changed_labels = labels.clone()
        changed_labels[attached] = 0
        addition = torch.tensor([[int(attached[0]), x.size(0)], [x.size(0), int(attached[0])]])
        return torch.cat([x, x.new_ones(1, x.size(1))]), torch.cat([edges, addition], dim=1), torch.cat([weights, weights.new_ones(2)]), changed_labels


class AttackInputTests(unittest.TestCase):
    def test_test_extremes_cannot_change_training_normalization(self):
        data, nodes, _ = example()
        for columnwise in [False, True]:
            original = data['x'].clone()
            expected = normalize_from_training_graph(original, nodes, columnwise)
            original[[1, 4]] = -100000
            changed = normalize_from_training_graph(original, nodes, columnwise)
            self.assertTrue(torch.equal(expected[nodes], changed[nodes]))
    def test_compaction_has_no_test_feature_or_label_rows(self):
        data, nodes, labels = example()
        graph, inverse = compact_training_graph(data, nodes, labels)
        self.assertEqual(graph['x'].size(0), 4)
        self.assertTrue(torch.equal(graph['x'], data['x'][nodes]))
        self.assertEqual(graph['train'].tolist(), [0, 1])
        self.assertEqual(graph['val'].tolist(), [3])
        self.assertEqual(inverse[[1, 4]].tolist(), [-1, -1])
        self.assertTrue(bool((graph['edges'] < 4).all()))
        self.assertEqual(graph['original_labels'].tolist(), labels.tolist())

    def test_test_only_feature_changes_do_not_change_training_inputs(self):
        data, nodes, labels = example()
        before, _ = compact_training_graph(data, nodes, labels)
        data['x'][[1, 4]] = 100000
        data['labels'][[1, 4]] = 999
        after, _ = compact_training_graph(data, nodes, labels)
        for name in ['x', 'edges', 'weights', 'labels', 'original_labels', 'train', 'val']:
            self.assertTrue(torch.equal(before[name], after[name]), name)

    def test_rejects_test_indices_and_test_edges(self):
        data, nodes, labels = example()
        _, inverse = compact_training_graph(data, nodes, labels)
        with self.assertRaises(ValueError):
            local_positions(inverse, torch.tensor([1]))
        data['edges'] = torch.tensor([[0], [1]])
        with self.assertRaises(ValueError):
            compact_training_graph(data, nodes, labels)

    def test_all_three_attacks_receive_only_training_graph(self):
        data, nodes, truth = example()
        graph, inverse = compact_training_graph(data, nodes, truth)
        attached = local_positions(inverse, torch.tensor([3]))
        available = local_positions(inverse, torch.tensor([5]))
        for attack in ['GTA', 'UGBA', 'DPGBA']:
            labels = attack_labels_for_generator(graph, truth, SimpleNamespace(attack=attack, target=0))
            generator = RecordingGenerator()
            poisoned = fit_and_poison(generator, graph, labels, attached, available)
            self.assertEqual(generator.events, ['fit', 'get_poisoned'])
            self.assertEqual(generator.fit_input[0].size(0), 4)
            self.assertEqual(generator.fit_input[3].numel(), 4)
            self.assertFalse(any(p.requires_grad for p in generator.trojan.parameters()))
            restored = restore_global_graph(data, nodes, poisoned)
            x, edges, weights, restored_labels = restored
            self.assertEqual(x.size(0), 7)
            self.assertTrue(torch.equal(x[[1, 4]], data['x'][[1, 4]]))
            self.assertEqual(edges[:, -2:].tolist(), [[3, 6], [6, 3]])
            self.assertEqual(restored_labels[3].item(), 0)
            self.assertEqual(restored_labels[[1, 4]].tolist(), [-1, -1])


if __name__ == '__main__':
    unittest.main()
