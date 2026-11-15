"""最终分类器的独立参数视图，不改写作者辅助阶段的参数。"""
from copy import copy

PROFILE = "integrated_abp_20261115"


def add_final_parameters(values, config):
    values["final_hidden"] = config.hidden
    values["final_epochs"] = config.epochs
    values["final_lr"] = config.lr
    values["final_weight_decay"] = config.weight_decay
    values["final_dropout"] = config.dropout
    if config.hidden < 1 or config.epochs < 1:
        raise ValueError("最终分类器宽度和训练轮数必须大于零。")


def final_classifier_args(args):
    final_args = copy(args)
    for source, destination in [("final_hidden", "hidden"), ("final_epochs", "epochs"),
                                ("final_lr", "train_lr"), ("final_weight_decay", "weight_decay"),
                                ("final_dropout", "dropout")]:
        if hasattr(args, source):
            setattr(final_args, destination, getattr(args, source))
    return final_args


def final_parameter_record(config):
    return {"profile": PROFILE, "hidden": config.hidden, "epochs": config.epochs,
            "lr": config.lr, "weight_decay": config.weight_decay, "dropout": config.dropout,
            "model": config.model, "gat_heads": 8 if config.model == "GAT" else None}
