"""ABP学习版：用被阻断的多节点分支定位需要复核的训练标签。"""
from copy import copy, deepcopy
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from .abp import build_tree, find_training_connections, filter_edges
from .common import setup, start_clock, elapsed, arguments
from .author_support import logits_of, capture_evaluation


def branch_profiles(edges, node_count, train_nodes):
    # 原ABP已计算分解树；计算每个末端分支含多少个原图节点。
    tree, edge_part, pairs = build_tree(edges, node_count)
    active, parent = find_training_connections(tree, train_nodes)
    children = {}
    for node, previous in parent.items():
        if previous is not None:
            children.setdefault(previous, []).append(node)
    size = {}
    for node in reversed(list(parent)):
        size[node] = int(node < node_count)
        for child in children.get(node, []):
            size[node] += size[child]
    largest = torch.zeros(node_count, dtype=torch.long, device=edges.device)
    removed = torch.zeros(node_count, dtype=torch.long, device=edges.device)
    # 先在CPU记录普通整数，避免逐边GPU同步。
    largest_values = [0] * node_count
    removed_values = [0] * node_count
    for node in range(node_count):
        if node not in parent:
            # 整个连通分量都没有训练节点，ABP会移去其全部非自连接。
            largest_values[node] = node_count
    for sender, receiver in pairs:
        if sender == receiver:
            continue
        part = edge_part[(sender, receiver)]
        if part in parent and part not in active and receiver == parent[part]:
            largest_values[receiver] = max(largest_values[receiver], size[part])
            removed_values[receiver] += 1
    largest[:] = torch.tensor(largest_values, device=edges.device)
    removed[:] = torch.tensor(removed_values, device=edges.device)
    return largest, removed


def prepare_supervision(training, args, device):
    # 本函数只接收training.pt；没有附着名单、测试真值或目标类别输入。
    if args.learn_min_branch < 1 or args.learn_epochs < 1 or args.epochs < 1:
        raise ValueError("分支阈值和训练轮数需要大于零。")
    if not 0 <= args.learn_confidence <= 1 or not 0 <= args.learn_review_quantile <= 1:
        raise ValueError("置信度和验证分位数需在0到1之间。")
    largest, removed = branch_profiles(training["edges"], training["x"].size(0), training["train"])
    candidates = training["train"][largest[training["train"]] >= args.learn_min_branch]
    trusted = training["train"][largest[training["train"]] < args.learn_min_branch]
    if trusted.numel() == 0:
        raise ValueError("排除候选标签后没有辅助训练节点。")
    labels_present = torch.unique(training["labels"][trusted])
    if labels_present.numel() != training["classes"]:
        raise ValueError("辅助训练子集没有覆盖全部类别，当前配置无法运行。")
    data = training.copy()
    data["edges"], data["weights"] = filter_edges(training["edges"], training["weights"],
                                                  training["x"].size(0), training["train"])
    return data, candidates, trusted, largest, removed


def recover_labels(model, data, positive_nodes, labels, negative_nodes, negative_weights, args):
    if args.learn_recover_epochs < 0 or args.learn_negative_weight < 0:
        raise ValueError("恢复轮数和错误标签约束权重不能为负。")
    if args.learn_recover_epochs == 0 or negative_nodes.numel() == 0:
        return {"epochs": 0, "validation_accuracy": None}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_accuracy = -1.0
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    for epoch in tqdm(range(args.learn_recover_epochs), desc="候选错误标签恢复训练", unit="轮", leave=False):
        model.train()
        optimizer.zero_grad()
        output = logits_of(model(data["x"], data["edges"], data["weights"]))
        positive_loss = F.nll_loss(output[positive_nodes], labels[positive_nodes])
        observed = data["labels"][negative_nodes]
        probabilities = output[negative_nodes].softmax(dim=-1)
        wrong_probability = probabilities.gather(1, observed[:, None]).flatten()
        # 每个候选只约束其原错误标签，保留其余类别之间的选择。
        penalties = -torch.log1p(-wrong_probability.clamp(max=1 - 1e-6))
        negative_loss = (penalties * negative_weights).sum() / negative_weights.sum().clamp(min=1e-6)
        loss = positive_loss + args.learn_negative_weight * negative_loss
        if not torch.isfinite(loss):
            raise RuntimeError("恢复训练出现非有限损失。")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = logits_of(model(data["x"], data["edges"], data["weights"]))[data["val"]]
            accuracy = validation.argmax(-1).eq(data["labels"][data["val"]]).float().mean().item()
            validation_loss = F.nll_loss(validation, data["labels"][data["val"]]).item()
        if accuracy > best_accuracy or (accuracy == best_accuracy and validation_loss < best_loss):
            best_accuracy, best_loss = accuracy, validation_loss
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return {"epochs": args.learn_recover_epochs, "validation_accuracy": best_accuracy, "validation_nll": best_loss}


def train_learning(inputs, args, device):
    from .models import make_model
    training = inputs["training"]
    begin = start_clock(device)
    stages = tqdm(total=4, desc="ABP学习版处理", unit="阶段")
    data, candidates, trusted, largest, removed = prepare_supervision(training, args, device)
    stages.update(1)
    teacher_args = copy(args)
    teacher_args.model = "GCN"
    teacher_args.hidden = args.learn_hidden
    teacher = make_model(inputs, teacher_args, device)
    teacher.fit(data["x"], data["edges"], data["weights"], data["labels"], trusted,
                data["val"], train_iters=args.learn_epochs, verbose=False)
    teacher.eval()
    with torch.no_grad():
        probability = logits_of(teacher(data["x"], data["edges"], data["weights"])).softmax(dim=-1)
    confidence, prediction = probability.max(dim=-1)
    stages.update(1)
    confident = candidates[confidence[candidates] >= args.learn_confidence]
    disagree = candidates[prediction[candidates] != data["labels"][candidates]]
    selected = confident[prediction[confident] != data["labels"][confident]]
    risk_class = None
    enrichment = None
    decision_state = None
    if args.learn_graph == "review":
        from .branch_review import blocked_incoming, local_scores, remove_selected
        blocked = blocked_incoming(training["edges"], training["weights"], training["x"].size(0), training["train"])
        calibration_nodes = data["val"][largest[data["val"]] >= args.learn_min_branch]
        if calibration_nodes.numel() < 5:
            raise ValueError("满足结构条件的验证节点少于5，当前局部复核阈值无法校准。")
        inspected = torch.cat([candidates, calibration_nodes])
        scores = local_scores(teacher, data["x"], training["edges"], training["weights"], blocked, inspected)
        threshold = float(torch.quantile(scores[calibration_nodes], args.learn_review_quantile))
        selected = candidates[scores[candidates] > threshold]
        confident = selected[(confidence[selected] >= args.learn_confidence) & (prediction[selected] != data["labels"][selected])]
        data["edges"], data["weights"] = remove_selected(training["edges"], training["weights"], blocked, selected)
        decision_state = {"threshold": threshold, "calibration_nodes": calibration_nodes.cpu().tolist(),
            "training_review_scores": scores[candidates].cpu().tolist(), "selected_nodes": selected.cpu().tolist()}
    if args.learn_rule == "risk" and candidates.numel():
        candidate_counts = torch.bincount(data["labels"][candidates], minlength=training["classes"]).float()
        trusted_counts = torch.bincount(data["labels"][trusted], minlength=training["classes"]).float()
        candidate_rate = candidate_counts / candidates.numel()
        trusted_rate = trusted_counts / trusted.numel()
        enrichment = candidate_rate - trusted_rate
        risk_class = int(enrichment.argmax())
        selected = candidates[data["labels"][candidates] == risk_class]
    labels = data["labels"].clone()
    labels[confident] = prediction[confident]
    final_nodes = torch.cat([trusted, confident]).sort().values
    if args.learn_graph == "review":
        retained_supervision = training["train"][~torch.isin(training["train"], selected)]
        final_nodes = torch.cat([retained_supervision, confident]).sort().values
    if args.learn_rule == "exclude":
        final_nodes = trusted
    elif args.learn_rule == "selective":
        # 消融：只撤回高置信冲突标签，其余保留原监督。
        final_nodes = training["train"][~torch.isin(training["train"], selected)]
        labels = data["labels"].clone()
    stages.update(1)
    # 最终模型与无防御/ABP使用相同初始种子及原fit。
    setup(args.seed, args.device)
    model = make_model(inputs, args, device)
    model.fit(data["x"], data["edges"], data["weights"], labels, final_nodes,
              data["val"], train_iters=args.epochs, verbose=False)
    negative_weights = 1 - probability[selected].gather(1, data["labels"][selected, None]).flatten()
    recovery = recover_labels(model, data, final_nodes, labels, selected, negative_weights, args)
    model.eval()
    build_seconds = elapsed(begin, device)
    frame = capture_evaluation(model, device)
    stages.update(1)
    stages.close()
    from .evaluation import evaluate
    evaluation_inputs = inputs.copy()
    evaluation_inputs["decision_state"] = decision_state
    if getattr(args, "training_only", False):
        metrics = {"prediction_seconds": 0.0}
    else:
        metrics = evaluate(evaluation_inputs, args, device, model, {"teacher": teacher})
    # 真实附着标签仅由上层评估诊断，选择过程不读取该信息。
    return {"models": [model], "evaluation_frames": [frame], "source_metrics": metrics,
        "defended_edges": inputs["poison_edge_index"], "defended_weights": inputs["poison_edge_weights"],
        "auxiliary_models": {"teacher": teacher}, "source_args": arguments(args), "internal_seeds": [args.seed],
        "author_training_nodes": final_nodes.cpu().tolist(), "author_attached_nodes": [],
        "candidate_nodes": candidates.cpu().tolist(), "trusted_nodes": trusted.cpu().tolist(),
        "selected_nodes": selected.cpu().tolist(), "teacher_disagree_nodes": disagree.cpu().tolist(),
        "corrected_nodes": confident.cpu().tolist(), "teacher_confidence": confidence[candidates].cpu().tolist(),
        "teacher_prediction": prediction[candidates].cpu().tolist(),
        "recovery": recovery,
        "inferred_risk_class": risk_class,
        "label_enrichment": enrichment.cpu().tolist() if enrichment is not None else None,
        "largest_branch": largest[candidates].cpu().tolist(),
        "decision_state": decision_state,
        "model_build_seconds": build_seconds, "reference_seconds": 0.0,
        "prediction_seconds": metrics["prediction_seconds"],
        "learning_label_scope": "辅助与最终模型只用training.pt训练/验证索引标签；未读取附着名单或测试真值",
        "training_api": "原RIGBD模型fit；ABP图处理后先结构排除辅助训练，再复核候选监督"}
