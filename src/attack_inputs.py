"""生成器只接收连续编号的训练子图；编号恢复只在外层保存时进行。"""
import torch


def compact_training_graph(data, node_ids, original_labels):
    if original_labels.numel() != node_ids.numel():
        raise ValueError("原始标签必须仅覆盖训练图节点。")
    if node_ids.numel() == 0 or node_ids.unique().numel() != node_ids.numel():
        raise ValueError("训练子图节点为空或存在重复。")
    inverse = torch.full((data["x"].size(0),), -1, dtype=torch.long, device=node_ids.device)
    inverse[node_ids] = torch.arange(node_ids.numel(), device=node_ids.device)
    edges = inverse[data["edges"]]
    train = inverse[data["train"]]
    val = inverse[data["val"]]
    if bool((edges < 0).any()) or bool((train < 0).any()) or bool((val < 0).any()):
        raise ValueError("训练图边或训练/验证索引包含训练图以外的节点。")
    graph = {"x": data["x"][node_ids].detach().clone(), "edges": edges,
             "weights": data["weights"].detach().clone(),
             "labels": data["labels"][node_ids].clone(), "original_labels": original_labels.clone(),
             "train": train, "val": val, "classes": data["classes"]}
    return graph, inverse


def local_positions(inverse, global_indices):
    positions = inverse[global_indices]
    if bool((positions < 0).any()):
        raise ValueError("生成器候选或附着位置包含测试节点。")
    return positions


def fit_and_poison(generator, graph, labels, attached, available):
    # 此接口没有全图特征、测试标签、测试索引或恢复测试边。
    count = graph["x"].size(0)
    if labels.numel() != count:
        raise ValueError("生成器标签长度与训练子图不一致。")
    indices = torch.cat([graph["train"], attached, available])
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= count):
        raise ValueError("生成器输入索引超出训练子图。")
    generator.fit(graph["x"], graph["edges"], graph["weights"], labels.clone(),
                  graph["train"], attached, available)
    # 冻结学习后的触发器，再仅对训练侧附着位置生成污染图。
    generator.trojan.eval()
    generator.trojan.requires_grad_(False)
    with torch.no_grad():
        poisoned = generator.get_poisoned(graph["x"], graph["edges"], graph["weights"], labels.clone(), attached)
    return poisoned


def restore_global_graph(data, node_ids, poisoned):
    # 此时生成器训练已结束；恢复旧保存格式，供既有防御和评估代码读取。
    x, edges, weights, labels = poisoned
    training_count = node_ids.numel()
    original_count = data["x"].size(0)
    if x.size(0) < training_count or labels.numel() != training_count:
        raise ValueError("作者生成器返回的训练节点或标签数量不一致。")
    new_count = x.size(0) - training_count
    new_ids = torch.arange(original_count, original_count + new_count, device=node_ids.device)
    mapping = torch.cat([node_ids, new_ids])
    restored_x = data["x"].clone()
    restored_x[node_ids] = x[:training_count]
    restored_x = torch.cat([restored_x, x[training_count:]])
    restored_labels = torch.full_like(data["labels"], -1)
    restored_labels[node_ids] = labels
    return restored_x.detach(), mapping[edges].detach(), weights.detach(), restored_labels
