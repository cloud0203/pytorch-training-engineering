# 训练预算与实验可比性

调整 batch、GPU 数量、累积次数、采样次数或数据规模时，先确定比较口径，再选择训练上限、学习率横轴和评估频率。相同 epoch 不保证相同更新次数，相同更新次数也不保证相同数据量或计算量。

## 先选择比较口径

| 实验目的 | 优先控制 | 同时报告 |
| --- | --- | --- |
| 模型或损失消融 | 有效全局 batch、成功更新数、数据/划分、LR 轨迹、评估时点 | 样本/token 暴露量、耗时、显存、多个 seed 的结果 |
| 仅因显存改变每卡 batch | 用梯度累积维持有效全局 batch，再固定成功更新数 | 尾部窗口差异、吞吐、BatchNorm/随机层差异 |
| 研究 batch 本身 | 明确选择相同样本/token 预算或相同更新预算，必要时分别做两组 | 未固定的另一项、LR 调参政策、实际预算 |
| 对齐既有论文配方 | 保留其 epoch、采样、batch 和调度定义 | 实际成功更新数、采样重复数和数据量 |
| 变长序列/流式数据 | 按需要控制有效 token 数或成功更新数 | padding 排除规则、实际长度分布与超额量 |
| 吞吐与系统性能比较 | 明确固定时间、硬件或计算预算 | 验证/编译开销、处理量、模型质量；等 step 不等 FLOPs |

新建消融实验默认建议固定有效全局 batch 与成功 optimizer-step；用户指定 epoch 或其他预算时尊重其协议。不要把 step 模式宣称为唯一公平的选择。

## 计算与例子

等大小、等权重、DDP 数据并行且完整累积窗口下：

```text
nominal_effective_batch = batch_size_per_device × world_size × accumulation_steps
```

world_size 是数据并行进程数，不是可见 GPU 数；混合张量/流水线并行时使用数据并行组大小。这个值是名义 batch，尾批、mask、可变 token 和跳过更新要另外统计实际量。

假设 12,800 个训练样本、无丢弃/补齐/重复采样、窗口恰好整除、无 AMP 跳步：

| 每卡 batch | 数据并行进程 | 累积次数 | 有效 batch | 每 epoch 更新数 | 10 epoch 更新数 |
| --- | --- | --- | --- | --- | --- |
| 8 | 4 | 1 | 32 | 400 | 4,000 |
| 16 | 4 | 1 | 64 | 200 | 2,000 |
| 4 | 4 | 2 | 32 | 400 | 4,000 |

前两行都处理 128,000 次样本，但更新次数相差一倍；若都做 4,000 次更新，第二行会处理 256,000 次样本。第三行可用于显存受限时维持原协议，但 BatchNorm、对比学习的批内负样本、随机层等使梯度累积不一定等价于真实大 batch。

启动时从实际 sampler 和每 rank DataLoader 得到 `L = len(loader_train)`，不要只用 `N / effective_batch` 猜测每轮更新数。若每个数据遍历结束都提交不足窗口，则每轮尝试更新数为 `ceil(L / accumulation_steps)`；如果丢弃不足窗口则为 `floor(...)`，丢弃部分不计入 optimized，已经参与训练计算的仍计入 seen。跨遍历累积时不适用这两个逐轮公式。DDP 各 rank 更新边界必须一致。

## 计数器定义

| 字段 | 含义 |
| --- | --- |
| micro_step | 各 rank 同步的训练 microbatch 次数；不乘 world_size，不计验证 |
| attempted_updates | 完成窗口后尝试 optimizer 更新的次数 |
| optimizer_step | 未因 AMP/错误策略跳过的 optimizer 更新次数；全零梯度/LR=0 的正常 step 仍计数 |
| skipped_updates | attempted_updates − optimizer_step |
| samples_seen / tokens_seen | 所有数据并行 rank 实际用于训练计算的暴露量；重复采样仍计数，token 排除 padding；不计预取未使用数据和验证 |
| samples_optimized / tokens_optimized | 属于成功 optimizer 更新窗口的有效量；跳过窗口不增加 |
| data_epoch / batch_in_epoch | 从 0 开始的数据遍历编号及本遍历已消费批次数，不取代全局训练预算 |

所有样本计数写明是算例、patch、点还是 token；点云场景宜同时报告算例数与有效点数。samples_seen 不是去重样本数。AMP 跳步可能仍改变模型 buffer/RNG，不能声称该窗口对模型毫无影响。

## 预算、停止与调度

- 使用一个显式主预算，如 `training.budget: {unit: optimizer_step, limit: 10000}`。epoch、samples_seen、tokens_seen 等仅在项目实际实现时作为替代模式；不同时给多个含糊的主上限。
- 独立的时间/尝试次数/连续跳步上限是保护条件，命中时记录 `stop_reason` 与未完成预算，不能称为已达到目标。配置示例提供连续 AMP 跳步上限。
- step 预算达到前重新迭代有限数据集，每次正确推进 sampler/dataset 的 epoch；不要用会缓存旧 batch 的无限 cycle 代替重采样。空 loader 或零可用更新应尽早失败。
- 在成功更新后判断 step 上限，到达时停止，不为了凑整 epoch 多做更新。所有 rank 在同一边界退出。提前停止和预算结束均保存实际计数。
- 数据量预算在累积窗口边界停止；若最后窗口越过目标，记录目标、实际和超额量。精确截断需专门处理全局分母和 DDP 一致性，不悄悄丢弃数据。
- `scheduler.interval: optimizer_step` 时，warmup_steps、total_steps 都按成功更新计数。warmup 包含在 total_steps 中；变化 batch 不重新从 epochs 隐式推导固定 horizon。
- 明确第 1 次更新使用的 LR、第 warmup 结束位置、最后一次更新及结束后状态；记录更新实际使用的 LR，不能把下一步 LR 标成当前值。检查 off-by-one。
- AMP 跳步不推进成功 step 调度器和 EMA；micro_step/seen/attempted/skipped 仍记录实际变化。采用第三方 Trainer 时核对其 global_step 是否包含跳步，不仅凭变量名映射。
- 数据量预算的 LR 也应有明确横轴，例如 processed token；若仍按更新数调度，说明换算假设和变化后的影响。Plateau 依赖验证次数，不能直接宣称拥有同样的固定 LR 轨迹。
- batch 变大不默认线性放大学习率。保持有效 batch 时先保留 LR；研究 batch 时声明固定 LR 或采用何种调参规则。线性缩放、sqrt 缩放不是对 AdamW、任意模型都成立的定律。
- EMA 衰减、AdamW 衰减和动量按更新发生；相同 epoch、不同更新数会改变累计作用。对齐更新数仍不保证 batch 变化后优化轨迹或数据量一致。

## 验证、日志与保存频率

- 固定 step 预算实验默认将验证、可视化、周期保存以 optimizer_step 为单位，epoch 留作数据进度。固定数据量实验则用同一数据量轴安排评估，或明确说明差异。
- `every_n_steps` 等字段都必须有 unit/interval；本地 microbatch 日志可单独按 micro_step 计，但不能作为公平比较的主轴。
- 用“计数跨越下一阈值”判定事件并保存下个阈值；不能只用取模。数据量可一次跨过阈值，AMP 跳步则计数不变，不能在同一 step 重复验证或保存。
- 正常预算结束执行一次最终验证和保存，若终点刚验证过则去重。配置示例 `validation.at_end`/`checkpoint.at_end` 不表示异常失败后强行验证。
- 对比 best 时对齐评估机会，报告终点指标和 best 所在预算位置；更频繁验证可能改变选 best 和 early stopping 的机会。严格固定预算的基线默认关闭 early stopping，启用时报告实际提前结束位置。
- TensorBoard/JSONL 同时提供 optimizer_step、samples_seen/tokens_seen 和 elapsed 时间；主曲线使用共同预算轴。训练指标窗口按真实分母归并，不能简单平均各 microbatch loss。
- 实验摘要包含预算目标/实际、名义和实际 batch、world_size、累积次数、尾批政策、跳步、数据指纹、LR horizon、评估频率和停止原因。

## step 存档与恢复

按 step 保存或结束可能位于 epoch 中间，不能保存 `epoch + 1` 后跳过剩余数据，也不能只加载权重并重跑全轮而声称精确续训。

- 配置示例要求 `checkpoint.resume_granularity: optimizer_step`：只在完整累积窗口结束、无待提交梯度时存恢复点；记录 data_epoch、已消费游标、全部计数、下个事件阈值和 RNG/数据顺序状态。
- 恢复时明确数据顺序、worker 增强和预取策略。可使用按 epoch/样本 ID 确定性生成的增强加游标重建，或经过验证的有状态数据加载方案；仅恢复主进程 RNG 或简单跳过若干 batch 不普遍保证一致。
- epoch-only 工程可以保留较早的 epoch 恢复点，把中途保存标为评估/初始化权重并提供新的恢复路径；不能覆盖唯一可恢复的 last 后仍宣称支持续训。配置/README 必须注明限制，不静默忽略示例要求。
- 同预算恢复不重新计算 horizon，不重复触发已完成事件。延长预算保留原调度、延长还是重启，必须显式选择并记录；后两者按新阶段处理。
- batch/world_size/accumulation 或采样策略变化即使名义有效 batch 不变，也需进行兼容性判断。严格恢复默认拒绝会改变数据/更新轨迹的修改，非严格延续记录差异。

## 从旧 epoch 模板迁移

配置示例 schema_version 2 改为固定成功更新预算，采用 `training.budget`、scheduler.total_steps/warmup_steps 和带 interval 的事件频率，移除正式训练中的 epochs/total_epochs/warmup_epochs/every_n_epochs。smoke 的 max_epochs 只是试跑保护上限。

迁移已有实验时，以实际 loader、累积政策和旧训练记录估计旧预算，展示新旧更新数与样本量后再确认；不要把 100 epoch 机械替换成 100 step。既有论文/用户指定 epoch 配方可以保留，附上等效更新数和可比性说明。

## 参考与适用范围

- [timm train.py](https://github.com/huggingface/pytorch-image-models/blob/main/train.py) 提供 `--sched-on-updates`，展示按 update 调度的入口；这不自动对齐不同实验的数据量。
- [Hugging Face TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) 提供 max_steps 及按 steps 验证/保存等控制。其参数优先级和计数行为应按安装版本核对，本 skill 的成功更新定义需要实现验证。
- [PyTorch AMP examples](https://docs.pytorch.org/docs/main/notes/amp_examples.html) 说明有效 batch、累积和 scaler 更新的边界。
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) 研究特定大 batch SGD 配方中的 LR 缩放与 warmup，不能直接推广为所有任务的默认缩放规则。

上述是实现参考；本文件的预算选择、事件去重和报告要求是本项目的设计约定。
