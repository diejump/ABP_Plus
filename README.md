# ABP与优化版整合项目

新目录：`D:/GNNAttackReaserch/abp_integrated_20261115`。

完整复制当前abp的代码、结构、作者仓库、数据、结果、隐藏文件及Git目录，再接入abp_learn_20261112的优化方法。两个原目录保持不变。1999个源文件在接入前逐一通过SHA256核对，详细清单见`paper/完整复制核验_20261115.json`。

| 方法名 | 用途 |
|---|---|
| none | 无防御 |
| abp | 原ABP |
| abp_dual | 普通模型＋ABP复核模型，按分支结构选择预测 |
| abp_assisted | 加入GCN教师恢复候选监督，预测时仍用两个模型 |
| abp_learn | 学习版消融入口 |

RIGBD、DShield、MAD、Prune、LoSplit继续沿用当前abp的作者适配流程。

当前规则：rate按注入前约80%的非测试训练图节点数计算；默认random；先选后滤原目标类、不补选。GTA、UGBA、DPGBA的训练接口均只接收重新编号的训练子图，测试节点特征、标签和边不进入生成器训练。最终分类器默认hidden32、epochs200；优化版恢复默认跟随epochs，教师参数独立控制。

训练结束并冻结生成器后，才在测试侧调用注入接口；测试真值仅供评分。完整说明见[攻击训练隔离与预算](paper/攻击训练隔离与预算_20261115.md)。Cora训练图2167个节点，rate=3的选样预算为65个；不是仅按541个有标签节点计算的16个。

```text
uv sync --locked --extra cu118
uv run --locked --extra cu118 python check_project.py
uv run --locked --extra cu118 python run_study.py --datasets Cora --models GCN GAT GraphSAGE --attacks UGBA --seeds 10 --methods none abp abp_dual abp_assisted --rate 3 --output-root outputs/integrated_run --device cuda:0
uv run --locked --extra cu118 python verify_results.py --output-root outputs/integrated_run
uv run --locked --extra cu118 python summarize.py --output-root outputs/integrated_run
```

先进入本项目根目录。GPU方法列表显式排除了LoSplit；LoSplit可另用CPU原流程。

详细阅读：[整合运行命令](paper/整合运行命令_20261115.md)、[整合说明](paper/整合说明_20261115.md)。也可修改src/config.py和run_study.py中的parse默认值后手动运行。

复制的历史结果保留原记录及绝对路径；新实验请使用新的输出子目录。旧优化文档保存在`paper/优化版来源_20261112`，其性能数字不代表当前预算与参数下的新结果。本次只做短轮次接口和重载检查。
