"""无防御和ABP的评估；作者比较方法使用各自原始评估代码。"""
import torch
from tqdm.auto import tqdm
from .abp import filter_edges
from .common import start_clock, elapsed
from .author_support import logits_of


@torch.no_grad()
def predict(model, x, edges, weights, training_nodes, method, args=None, auxiliary=None, decision_state=None):
    if method == "abp_learn" and getattr(args, "learn_graph", "full") == "review":
        from .abp_learning import branch_profiles
        from .branch_review import blocked_incoming, local_scores, remove_selected
        largest, _ = branch_profiles(edges, x.size(0), training_nodes)
        candidates = (largest >= args.learn_min_branch).nonzero().flatten()
        blocked = blocked_incoming(edges, weights, x.size(0), training_nodes)
        scores = local_scores(auxiliary["teacher"], x, edges, weights, blocked, candidates)
        selected = candidates[scores[candidates] > decision_state["threshold"]]
        edges, weights = remove_selected(edges, weights, blocked, selected)
    elif method in ["abp", "abp_learn"]:
        edges, weights = filter_edges(edges, weights, x.size(0), training_nodes)
    model.eval()
    output = logits_of(model(x, edges, weights))
    if not torch.isfinite(output).all():
        raise RuntimeError("模型预测含非有限值。")
    return output


def evaluate(inputs, args, device, model, auxiliary=None, entropy_threshold=None):
    method = args.method
    edges = torch.cat([inputs["poison_edge_index"], inputs["mask_edge_index"]], dim=1)
    weights = torch.cat([inputs["poison_edge_weights"], inputs["poison_x"].new_ones(inputs["mask_edge_index"].size(1))])
    graphs = [(inputs["poison_x"], edges, weights)]
    if inputs["trigger"] is not None:
        graphs.append(inputs["trigger"].inject_trigger(inputs["idx_atk"], *graphs[0]))
    outputs = []
    begin = start_clock(device)
    for x, edge, weight in tqdm(graphs, desc="正常图与触发图评估", unit="张", leave=False):
        outputs.append(predict(model, x, edge, weight, inputs["training"]["train"], method, args, auxiliary, inputs.get("decision_state")))
    metrics = summarize_outputs(inputs, outputs, method)
    metrics["prediction_seconds"] = elapsed(begin, device)
    return metrics


def summarize_outputs(inputs, outputs, method):
    truth = inputs["data"].y
    normal = inputs["idx_clean_test"]
    ca_hits = int(outputs[0][normal].argmax(-1).eq(truth[normal]).sum())
    metrics = {"CA": ca_hits / normal.numel(), "ASR": None, "CA_hits": ca_hits,
               "CA_denominator": normal.numel(), "ASR_hits": None, "ASR_denominator": 0,
               "CA_scope": "正常完整测试图上的正常测试节点",
               "ASR_scope": "完整图整体注入；原标签非目标类攻击测试节点"}
    if len(outputs) == 2:
        target = inputs["evaluation"]["target"]
        affected = inputs["idx_atk"][truth[inputs["idx_atk"]] != target]
        hits = int(outputs[1][affected].argmax(-1).eq(target).sum())
        metrics.update(ASR=hits / affected.numel(), ASR_hits=hits, ASR_denominator=affected.numel())
        natural_hits = int(outputs[0][affected].argmax(-1).eq(target).sum())
        all_attack = inputs["idx_atk"]
        true_target = all_attack[truth[all_attack] == target]
        metrics["natural_target_error"] = natural_hits / affected.numel()
        metrics["added_target_error"] = (hits - natural_hits) / affected.numel()
        metrics["triggered_node_accuracy"] = outputs[1][all_attack].argmax(-1).eq(truth[all_attack]).float().mean().item()
        metrics["true_target_count"] = true_target.numel()
        metrics["true_target_recall_normal"] = outputs[0][true_target].argmax(-1).eq(target).float().mean().item() if true_target.numel() else None
        metrics["true_target_recall_triggered"] = outputs[1][true_target].argmax(-1).eq(target).float().mean().item() if true_target.numel() else None
        if method in ["abp", "abp_learn", "abp_dual"]:
            old_count = outputs[0].size(0)
            metrics["original_prediction_changes"] = int(outputs[0].argmax(-1).ne(outputs[1][:old_count].argmax(-1)).sum())
            metrics["original_logit_max_difference"] = float((outputs[0] - outputs[1][:old_count]).abs().max())
    return metrics
