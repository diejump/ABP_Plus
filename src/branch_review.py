"""逐节点删除其无训练节点分支，测量辅助模型的实际预测变化。"""
import torch
from tqdm.auto import tqdm
from .abp import filter_edges
from .author_support import logits_of


def blocked_incoming(edges, weights, count, train_nodes):
    retained, _ = filter_edges(edges, weights, count, train_nodes)
    original_keys = edges[0] * count + edges[1]
    retained_keys = retained[0] * count + retained[1]
    return ~torch.isin(original_keys, retained_keys)


def js_divergence(first, second):
    mean = (first + second) / 2
    left = first * (first.clamp(min=1e-8).log() - mean.clamp(min=1e-8).log())
    right = second * (second.clamp(min=1e-8).log() - mean.clamp(min=1e-8).log())
    return ((left.sum(-1) + right.sum(-1)) / 2).clamp(min=0)


@torch.no_grad()
def local_scores(teacher, x, edges, weights, blocked, candidates):
    teacher.eval()
    raw = logits_of(teacher(x, edges, weights)).softmax(-1)
    scores = x.new_zeros(x.size(0))
    for node in tqdm(candidates.tolist(), desc="逐节点分支复核", unit="节点", leave=False):
        keep = ~(blocked & (edges[1] == node))
        changed = logits_of(teacher(x, edges[:, keep], weights[keep])).softmax(-1)
        scores[node] = js_divergence(raw[node], changed[node])
    return scores


def remove_selected(edges, weights, blocked, selected):
    keep = ~(blocked & torch.isin(edges[1], selected))
    return edges[:, keep], weights[keep]
