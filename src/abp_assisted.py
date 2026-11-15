"""ABP-L：普通分类模型、可靠标签教师和结构限定的复核分类模型。"""
from copy import copy
from .common import setup, start_clock, elapsed, arguments
from .author_support import capture_evaluation
from .abp_learning import train_learning
from .abp_dual import evaluate


def train_assisted(inputs, args, device):
    from .models import make_model
    begin = start_clock(device)
    data = inputs["training"]
    primary = make_model(inputs, args, device)
    primary.fit(data["x"], data["edges"], data["weights"], data["labels"], data["train"],
                data["val"], train_iters=args.epochs, verbose=False)
    primary.eval()
    setup(args.seed, args.device)
    learning = copy(args)
    learning.method = "abp_learn"
    learning.training_only = True
    learning.learn_graph = "full"
    learned = train_learning(inputs, learning, device)
    reviewer = learned["models"][0]
    teacher = learned["auxiliary_models"]["teacher"]
    build_seconds = elapsed(begin, device)
    frame = capture_evaluation(primary, device)
    begin = start_clock(device)
    metrics = evaluate(inputs, args, device, primary, {"reviewer": reviewer})
    prediction_seconds = elapsed(begin, device)
    # 教师仅用于训练监督复核，测试图推理只调用primary和reviewer。
    result = learned.copy()
    result.update(models=[primary], auxiliary_models={"reviewer": reviewer, "teacher": teacher},
                  evaluation_frames=[frame], source_metrics=metrics, source_args=arguments(args),
                  model_build_seconds=build_seconds, prediction_seconds=prediction_seconds)
    result["learning_configuration"] = {"graph": "full", "rule": learning.learn_rule,
        "teacher_arch": "GCN", "teacher_hidden": learning.learn_hidden, "teacher_epochs": learning.learn_epochs,
        "reviewer_arch": args.model, "reviewer_epochs": args.epochs,
        "recovery_epochs": args.learn_recover_epochs, "negative_weight": args.learn_negative_weight}
    result["training_api"] = "原普通模型fit；可靠子集GCN教师；原同骨架复核模型fit及候选标签约束；结构限定推理"
    return result
