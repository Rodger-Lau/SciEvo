# 消融实验说明

本项目针对 CHI 数据集支持四个实验版本：

- `full`：启用三个提出的模块，作为完整模型。
- `no_prefrontal`：消融 prefrontal decision-making。
- `no_fine_grained_editing`：消融 fine-grained editing。
- `no_learning_memory_interaction`：消融 learning-memory interaction。

每个版本都通过独立脚本运行，保证日志、TensorBoard 数据和 CSV 结果分别保存到不同目录，避免相互覆盖。

## 1. 实验设计

本次消融实验采用“单因素逐一消融”的方式，与完整模型逐项比较。

- **完整模型**：三个模块全部启用。
- **Prefrontal decision-making 消融**：去掉决定任务 / cluster 顺序的模块。
- **Fine-grained editing 消融**：去掉执行层级编辑和冻结的模块。
- **Learning-memory interaction 消融**：去掉基于 DQN 的交互策略，改为固定的回退动作。

这样设计可以保持数据划分、优化器、课程学习逻辑和评估协议不变，从而把性能差异尽量归因到被消融的模块上。

## 2. 三个模块分别如何消融

### 2.1 Prefrontal decision-making

这个模块负责在训练前决定任务顺序。在完整模型中，它会根据梯度或 cluster 排序来确定训练序列。

在消融版本中：

- 关闭该决策模块；
- 任务顺序回退到基线顺序或默认的非决策策略；
- 其他训练设置保持不变。

这里不是简单删除代码，而是通过受控的行为切换来消融，这样更容易解释结果，也更便于复现。

### 2.2 Fine-grained editing

这个模块负责编辑式训练行为，包括选择性冻结和对应的 soft-label 更新流程。

在消融版本中：

- 关闭编辑路径；
- 模型不再执行层编辑 / 选择性冻结；
- 训练改为普通的端到端更新流程。

这样可以单独观察 fine-grained editing 对性能的贡献，同时不改变网络结构本身。

### 2.3 Learning-memory interaction

这个模块是由 DQN 驱动的交互机制，用来在训练中选择和优化动作。

在消融版本中：

- 关闭 DQN 交互；
- 训练过程使用固定的回退动作，而不是学习得到的动作选择；
- 跳过交互策略的 replay / memory 更新。

这样可以保留训练流程的可运行性，同时移除自适应的决策组件。

## 3. 运行脚本

使用下面的脚本可以分别启动不同实验：

```bash
bash scripts/ablation_full.sh
bash scripts/ablation_no_prefrontal.sh
bash scripts/ablation_no_fine_grained_editing.sh
bash scripts/ablation_no_learning_memory_interaction.sh
```

每个脚本都会把结果写到自己的实验目录中。

## 4. 输出目录

主要输出按实验名隔离保存：

- `logs/BrainAI/{dataset}/{experiment_name}/`
- `Writer/BrainAI/{dataset}/{experiment_name}/`
- `csv_files/BrainAI/{dataset}/{experiment_name}/`

最终的比较表会写入对应 CSV 目录下的 `summary.csv`。

## 5. 分析方法

分析消融结果时，建议对完整模型和各个消融版本使用相同指标进行比较：

- MAE
- RMSE
- MAPE

推荐报告方式：

- 多个 seed 的均值和标准差；
- 相对完整模型的退化幅度；
- 如果 seed 数量足够，可以补充配对显著性检验。

为了更好地解释结果，还可以进一步检查：

- prefrontal decision-making 消融后的任务顺序变化；
- fine-grained editing 消融后的冻结 / 编辑行为变化；
- learning-memory interaction 消融后的动作分布和 replay 使用情况。

## 6. 结果解释

- 如果 `no_prefrontal` 的性能下降，说明任务排序策略对课程学习是有帮助的。
- 如果 `no_fine_grained_editing` 的性能下降，说明选择性编辑 / 冻结策略是有效的。
- 如果 `no_learning_memory_interaction` 的性能下降，说明学习到的交互策略优于固定回退动作。

核心思想是：每个消融只移除一个行为，其余系统保持不变，因此比较结果更偏向“归因分析”，而不是“结构性重构”。
