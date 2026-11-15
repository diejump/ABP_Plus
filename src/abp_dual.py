"""ABP双模型：普通节点保留原分类结果，有末端分支的节点交给复核模型。"""
import torch
from tqdm.auto import tqdm
from .abp import filter_edges
from .abp_learning import branch_profiles, recover_labels
from .common import setup, start_clock, elapsed, arguments
from .author_support import capture_evaluation, logits_of


def train_dual(inputs, args, device):
    from .models import make_model
    if args.learn_min_branch < 1 or args.epochs < 1:
        raise ValueError("分支阈值与训练轮数需要大于零。")
    data = inputs["training"]
    begin = start_clock(device)
    stages = tqdm(total=4, desc="ABP双模型训练", unit="阶段")
    largest, _ = branch_profiles(data["edges"], data["x"].size(0), data["train"])
    candidates = data["train"][largest[data["train"]] >= args.learn_min_branch]
    trusted = data["train"][largest[data["train"]] < args.learn_min_branch]
    if trusted.numel() == 0 or torch.unique(data["labels"][trusted]).numel() != data["classes"]:
        raise ValueError("复核模型的保留训练节点未覆盖全部类别。")
    protected = data.copy()
    protected["edges"], protected["weights"] = filter_edges(data["edges"], data["weights"], data["x"].size(0), data["train"])
    primary = make_model(inputs, args, device)
    primary.fit(data["x"], data["edges"], data["weights"], data["labels"], data["train"],
                data["val"], train_iters=args.epochs, verbose=False)
    primary.eval()
    stages.update(1)
    setup(args.seed, args.device)
    reviewer = make_model(inputs, args, device)
    reviewer.fit(protected["x"], protected["edges"], protected["weights"], data["labels"], trusted,
                 data["val"], train_iters=args.epochs, verbose=False)
    reviewer.eval()
    with torch.no_grad():
        probability = logits_of(reviewer(protected["x"], protected["edges"], protected["weights"])).softmax(-1)
    stages.update(1)
    candidate_counts = torch.bincount(data["labels"][candidates], minlength=data["classes"]).float()
    trusted_counts = torch.bincount(data["labels"][trusted], minlength=data["classes"]).float()
    enrichment = candidate_counts / max(candidates.numel(), 1) - trusted_counts / trusted.numel()
    risk_class = int(enrichment.argmax()) if candidates.numel() else None
    selected = candidates[data["labels"][candidates] == risk_class] if risk_class is not None else candidates
    weights = 1 - probability[selected].gather(1, data["labels"][selected, None]).flatten()
    recovery = recover_labels(reviewer, protected, trusted, data["labels"], selected, weights, args)
    stages.update(1)
    seconds = elapsed(begin, device)
    frame = capture_evaluation(primary, device)
    begin = start_clock(device)
    metrics = evaluate(inputs, args, device, primary, {"reviewer": reviewer})
    prediction_seconds = elapsed(begin, device)
    stages.update(1)
    stages.close()
    return {"models": [primary], "auxiliary_models": {"reviewer": reviewer},
        "evaluation_frames": [frame], "source_metrics": metrics,
        "defended_edges": data["edges"], "defended_weights": data["weights"],
        "source_args": arguments(args), "internal_seeds": [args.seed],
        "author_training_nodes": data["train"].cpu().tolist(), "author_attached_nodes": [],
        "candidate_nodes": candidates.cpu().tolist(), "trusted_nodes": trusted.cpu().tolist(),
        "selected_nodes": selected.cpu().tolist(), "inferred_risk_class": risk_class,
        "label_enrichment": enrichment.cpu().tolist(), "recovery": recovery,
        "reviewer_training_nodes": trusted.cpu().tolist(),
        "model_build_seconds": seconds, "reference_seconds": 0.0, "prediction_seconds": prediction_seconds,
        "learning_label_scope": "两个模型均只训练已给训练索引；验证标签只用于检查点选择；目标类别从候选标签比例推断",
        "training_api": "两个原RIGBD骨架fit；复核模型另加候选错误标签约束"}


@torch.no_grad()
def predict_dual(primary, reviewer, x, edges, weights, train_nodes, minimum):
    primary.eval()
    reviewer.eval()
    ordinary = logits_of(primary(x, edges, weights))
    protected_edges, protected_weights = filter_edges(edges, weights, x.size(0), train_nodes)
    reviewed = logits_of(reviewer(x, protected_edges, protected_weights))
    largest, _ = branch_profiles(edges, x.size(0), train_nodes)
    routing = largest >= minimum
    output = ordinary.clone()
    output[routing] = reviewed[routing]
    if not torch.isfinite(output).all():
        raise ValueError("双模型预测存在非有限数值。")
    return output, routing, ordinary


def evaluate(inputs, args, device, primary, auxiliary=None, entropy_threshold=None):
    reviewer = auxiliary["reviewer"]
    data = inputs["training"]
    heldout = inputs["mask_edge_index"]
    edges = torch.cat([inputs["poison_edge_index"], heldout], dim=1)
    weights = torch.cat([inputs["poison_edge_weights"], data["x"].new_ones(heldout.size(1))])
    graphs = [(inputs["poison_x"], edges, weights)]
    if inputs["trigger"] is not None:
        graphs.append(inputs["trigger"].inject_trigger(inputs["idx_atk"], *graphs[0]))
    outputs, masks, primary_outputs = [], [], []
    for x, edge, weight in tqdm(graphs, desc="双模型正常图与触发图评估", unit="张", leave=False):
        output, mask, ordinary = predict_dual(primary, reviewer, x, edge, weight, data["train"], args.learn_min_branch)
        outputs.append(output)
        masks.append(mask)
        primary_outputs.append(ordinary)
    from .evaluation import summarize_outputs
    metrics = summarize_outputs(inputs, outputs, "abp_dual")
    clean = inputs["idx_clean_test"]
    metrics["reviewed_clean_test_nodes"] = int(masks[0][clean].sum())
    truth = inputs["data"].y
    before = primary_outputs[0][clean].argmax(-1).eq(truth[clean])
    after = outputs[0][clean].argmax(-1).eq(truth[clean])
    metrics["primary_CA"] = before.float().mean().item()
    metrics["routing_CA_difference_bound"] = int(masks[0][clean].sum()) / clean.numel()
    metrics["route_corrected_clean_nodes"] = int((~before & after).sum())
    metrics["route_harmed_clean_nodes"] = int((before & ~after).sum())
    if len(outputs) == 2:
        attack = inputs["idx_atk"]
        metrics["reviewed_attack_nodes"] = int(masks[1][attack].sum())
        metrics["attack_test_node_count"] = attack.numel()
    return metrics
