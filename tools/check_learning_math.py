"""对理论中使用的有限图定义、路由上界和损失导数做数值核对。"""
import itertools
import sys
from pathlib import Path
import networkx as nx
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.abp import filter_edges
from src.abp_learning import branch_profiles
from src.common import save_json, file_hash


def graph_edges(graph):
    pairs = []
    for first, second in graph.edges:
        pairs.append((first, second))
        pairs.append((second, first))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs).t().contiguous()


def check_branch_definition():
    checked = 0
    for count in range(1, 5):
        possible = list(itertools.combinations(range(count), 2))
        for bits in tqdm(range(1 << len(possible)), desc=str(count) + '节点结构定义', unit='图', leave=False):
            graph = nx.Graph()
            graph.add_nodes_from(range(count))
            for index, pair in enumerate(possible):
                if bits & (1 << index):
                    graph.add_edge(*pair)
            edges = graph_edges(graph)
            for train_bits in range(1, 1 << count):
                train = torch.tensor([i for i in range(count) if train_bits & (1 << i)])
                values, _ = branch_profiles(edges, count, train)
                training = set(train.tolist())
                for receiver in range(count):
                    if not nx.node_connected_component(graph, receiver).intersection(training):
                        expected = count
                    else:
                        reduced = graph.copy()
                        neighbors = list(reduced.neighbors(receiver))
                        reduced.remove_node(receiver)
                        expected = 0
                        for neighbor in neighbors:
                            component = nx.node_connected_component(reduced, neighbor)
                            if not component.intersection(training):
                                expected = max(expected, len(component))
                    if int(values[receiver]) != expected:
                        raise ValueError('分支定义不一致：' + str((count, bits, train_bits, receiver, int(values[receiver]), expected)))
                checked += 1
    return checked


def check_routing_bound():
    generator = torch.Generator().manual_seed(123)
    count = 0
    for _ in range(100):
        labels = torch.randint(0, 7, (100,), generator=generator)
        original = torch.randint(0, 7, (100,), generator=generator)
        reviewer = torch.randint(0, 7, (100,), generator=generator)
        selected = torch.rand(100, generator=generator) < 0.2
        result = original.clone()
        result[selected] = reviewer[selected]
        difference = abs(int(result.eq(labels).sum()) - int(original.eq(labels).sum()))
        if difference > int(selected.sum()):
            raise ValueError('路由正确率界不成立。')
        count += 1
    return count


def check_negative_gradient():
    logits = torch.tensor([0.5, -0.2, 1.0], dtype=torch.double, requires_grad=True)
    probability = logits.softmax(-1)
    loss = -torch.log1p(-probability[0])
    loss.backward()
    expected = -probability.detach()[0] * probability.detach() / (1 - probability.detach()[0])
    expected[0] = probability.detach()[0]
    if not torch.allclose(logits.grad, expected, atol=1e-12):
        raise ValueError('负标签损失梯度公式不匹配。')
    return {'automatic_gradient': logits.grad.tolist(), 'formula_gradient': expected.tolist()}


def check_enrichment_identity():
    generator = torch.Generator().manual_seed(123)
    for _ in range(100):
        natural = torch.rand(7, generator=generator, dtype=torch.double)
        reference = torch.rand(7, generator=generator, dtype=torch.double)
        natural /= natural.sum()
        reference /= reference.sum()
        eta = float(torch.rand(1, generator=generator))
        target = int(torch.randint(0, 7, (1,), generator=generator))
        onehot = torch.zeros(7, dtype=torch.double)
        onehot[target] = 1
        delta = (1 - eta) * natural + eta * onehot - reference
        expression = eta * (onehot - reference) + (1 - eta) * (natural - reference)
        if not torch.allclose(delta, expression, atol=1e-12):
            raise ValueError('类别比例差恒等式不成立。')
        epsilon = float((natural - reference).abs().max())
        for other in range(7):
            if other == target:
                continue
            lower = eta * (1 - reference[target] + reference[other]) - 2 * (1 - eta) * epsilon
            if float(delta[target] - delta[other]) < float(lower) - 1e-12:
                raise ValueError('类别排序下界不成立。')
    return 100


def main():
    report = {'status': 'passed', 'branch_definition_cases': check_branch_definition(),
              'routing_bound_trials': check_routing_bound(), 'negative_loss_gradient': check_negative_gradient(),
              'label_enrichment_identity_and_bound_trials': check_enrichment_identity(),
              'abp_sha256': file_hash(ROOT / 'src/abp.py'), 'learning_sha256': file_hash(ROOT / 'src/abp_learning.py'),
              'scope': '有限图定义和数值等式核查；不构成ASR或CA的普遍保证；不训练数据集模型'}
    save_json(ROOT / 'outputs/mathematical_checks.json', report)
    print('有限图定义、路由界和梯度公式检查通过。')


if __name__ == '__main__':
    main()
