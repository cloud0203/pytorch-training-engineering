# 训练实现规范

## 配置与启动

- 新项目默认提供 CPU/单 GPU/torchrun DDP 共用入口；硬件或模型不支持的组合明确拒绝。
- 写明配置覆盖优先级，例如代码默认值 < 配置 < CLI；尽早检测未知字段、错误类型、范围和冲突。
- batch_size_per_device 表示每进程 batch；记录 world_size、accumulation_steps 和有效全局 batch。变长任务另记录有效元素数。
- 主训练上限使用显式预算单位和数值，避免 epochs 与 step 上限含糊并存；具体选择、计数和迁移见 [训练预算与实验可比性](training-budget.md)。新消融实验优先固定有效全局 batch 与成功更新数，用户指定的 epoch 配方仍可保留。
- resume、init-from 互斥。设备不可用、数据为空、字段缺失、输出目录冲突在昂贵计算前报错。
- 启动摘要包含模型总参数/可训练参数、数据规模、精度、损失、主指标、LR、设备、输出路径，以及每卡 batch、数据并行大小、累积次数、实际 loader 长度、名义有效 batch、预算和预计更新数。

## 数据与任务接口

- 约定 batch 的输入、target、mask、样本 ID 和 shape/dtype；条件输入通过明确接口传递。
- 提供 build_model、build_datasets、compute_loss、metric update/compute 等替换点，无需强制所有任务使用相同 batch 键名。
- 验证集固定划分与采样协议，训练增强不进入验证；测试集不参与选 best 或调参。
- 检查有限值、标签范围、输入输出维度；明确空 mask 的损失、指标和分母行为。
- pin_memory、non_blocking、num_workers、prefetch_factor、persistent_workers 按设备与数据设置；num_workers=0 时不传仅多进程适用的参数。
- persistent_workers 下主进程 dataset.set_epoch 不会自动更新 worker 副本；使用共享 epoch 状态、无状态按样本种子或重建 worker 等真实方案。
- 合成数据试跑验证训练管线，真实小样本试跑验证数据契约，两者不能互相替代。

## 优化器、调度器、精度

- 无项目约定时可用 AdamW 作为起点；SGD 等按需添加。参数来自配置，记录参数组名称和 LR。
- bias/norm 是否 weight decay 由任务和模型决定，不静默改变已有配方。
- 调度器明确 interval：epoch、成功的 optimizer-step 或 validation-metric。Plateau 在对应验证后推进；step 调度器在成功更新后推进。
- warmup 与主调度使用同一明确单位，定义总周期是否含 warmup、初始 LR、终点和越界行为。step 模式不因 batch 变化重算固定 horizon；恢复时不隐式重启周期。
- FP32、FP16、BF16 显式选择并检查设备支持，不静默更换。FP16 常需 GradScaler，BF16 通常不需；按实际版本和设备实现。
- autocast 覆盖适合的算子；敏感归约、物理空间还原等必要时用 FP32 或更高精度。验证关闭梯度。
- 梯度裁剪在 unscale 后执行，记录裁剪前范数；非有限梯度按明确策略处理。
- scaler 因溢出跳过更新时，成功 optimizer-step、按 step 的 scheduler 和 EMA 不应假装发生更新，实现可测试的检测逻辑。

## 梯度累积

- 未启用时 accumulation_steps=1；启用后区分 micro_step 和 optimizer_step。
- 等大小 batch、等权 loss 可按实际累积窗口大小归一化；最后不足窗口不能仍按完整窗口机械相除。
- 不等大小 batch、mask、变长 token 按全局有效样本/元素数加权。DDP 默认平均梯度时计入 world_size，保证与目标大 batch 的均值定义一致。
- 累积期间不清零梯度；窗口结束后 unscale、裁剪、step、scaler update、清零。
- DDP no_sync 同时包住非最终 microbatch 的 forward 和 backward；各 rank 更新窗口一致。
- BatchNorm、随机层等可能使累积与大 batch 本来就不等价；验收使用适当的简单模型。

## 分布式训练与验证

- 读取 RANK/WORLD_SIZE/LOCAL_RANK；CUDA 先设置本地设备，选择适用 backend，统一初始化和 finally 清理。
- 训练 sampler 每轮 set_epoch。说明训练补齐或 drop_last 政策和实际样本暴露次数；补齐后的训练指标不等于原始数据集去重指标。
- 验证不通过重复样本补齐来改变指标。处理不等长分片及部分 rank 零样本，所有 rank 最终参与相同的指标通信。
- 不等长验证若绕过 DDP wrapper，验证前保证必要 buffer 一致，确认 forward 内无不匹配的 collectives；SyncBatchNorm、自定义通信需专门处理。
- 仅全局 rank 0 写共享 checkpoint、TensorBoard 和汇总文件；其他 rank 可写各自命名的诊断日志。
- 主指标、best 决策、early stopping 和失败退出在所有 rank 一致。
- 数据准备失败需传达其他 rank；rank 0 保存失败不能让其余 rank 永久等待。使用可终止 launcher、合理通信超时和已知同步边界的错误传播，异常处理中不盲目 barrier。
- find_unused_parameters、static_graph、SyncBatchNorm 按模型启用，不作为通用性能开关默认打开。

## 损失与指标

- 区分优化目标、各损失分量、报告指标和 best 选择指标，声明空间、单位、min/max。
- 按指标累积充分统计量和分母再跨 rank 合并，不简单平均 batch/rank 的均值、R² 或 F1。
- R² 使用数值稳定的方差统计；AUROC 等依赖排序的指标需聚合预测或采用明确的近似协议。
- MAPE 声明分母下限、mask 和覆盖率；常量目标、零分母或无有效样本产生的未定义指标写 null，不冒充零。
- 分通道/区域/类别指标明确宏平均、微平均、样本平均；总体指标不能掩盖关键分组失效。
- 无效主验证指标不能覆盖 best；测试集不驱动 scheduler、early stopping 或 best。

## 日志与时间

| 分组 | 建议内容 |
| --- | --- |
| train_step_local/ | rank 0 本地 batch 目标与损失项，以明确的 micro-step 为轴 |
| train_epoch/、val/ | 全局聚合指标、单位、分母 |
| optim/ | 各参数组 LR、梯度范数、scaler scale、跳过更新次数 |
| progress/ | micro_step、attempted_updates、optimizer_step、samples_seen/optimized、按需 tokens_seen/optimized、预算完成量 |
| time/ | 准备、训练、验证、保存、日志开销及运行总耗时 |
| perf/ | 有效 samples/s 或 tokens/s、每 rank 峰值 allocated/reserved 显存 |
| visual/ | 固定样本 ID 的预测、真值和误差图 |

- 控制台、文本、JSONL、TensorBoard 使用同源指标，记录频率可配置。TensorBoard 可关闭，关闭时不强制依赖导入成功。
- 验证、保存和正式对比曲线使用声明的预算轴；step 模式按成功更新数触发并去重，正常结束补一次最终验证。epoch 只是数据遍历进度，不能代替实际更新计数。
- 不为每个日志值重复计算完整指标。float(cuda_tensor)、item()、频繁打印也可能同步，降低热路径开销。
- 低成本计时使用 perf_counter，阶段边界必要时同步 CUDA；精细 GPU 时间用 CUDA events 或 profiler，不默认每 batch synchronize。
- DataLoader 的 next 等待是主进程阻塞，不是全部 I/O 时间；GPU 与预取可重叠，分项不能直接解释为不重叠占比。
- DDP 报告各 rank 时间及最大值；逐项最大值可能来自不同 rank，不能相加。吞吐量使用匹配阶段的总处理量和耗时。
- 显存峰值按阶段或 epoch 重置采集，首轮初始化/编译/缓存开销与稳定吞吐分开解释。
- 写明 epoch 总耗时是否含验证、保存、日志。checkpoint 无法包含自身最终写入耗时，完整耗时写侧文件。
- ETA 和累计耗时区分当前会话与此前训练，不把停机时长当成训练计算耗时。

## checkpoint 与恢复

基础 checkpoint 包含：format_version、run_id、模型 state_dict、重建配置、输入输出语义、归一化/预处理、数据身份、optimizer、scheduler、scaler、数据 epoch、micro_step、attempted/skipped_updates、成功 optimizer_step、seen/optimized 数据量、主预算与调度 horizon、best 值及其 step/epoch、各 rank RNG、world_size。启用 EMA/early stopping 时保存其状态；独立 generator、sampler 等有状态组件需保存或可确定性重建。

- last 用于恢复，best 按指定验证指标保存；周期存档和保留数量按需配置。临时文件位于目标文件同一文件系统，完成后原子替换，失败不覆盖旧文件。
- 推理可独立取得模型构建和预处理信息，不依赖训练时绝对路径、完整 Dataset 对象或 pickle 化模型对象。
- 不加载未知来源的不可信完整 pickle；完整恢复按可信本地产物处理，加载参数匹配 PyTorch 版本。
- resume 恢复训练状态；init-from 只加载指定权重并新建实验，重新建立 optimizer、scheduler、scaler、step、best 和随机状态。部分加载报告缺失/多余参数。
- 兼容性矩阵：模型、特征语义、划分、归一化、损失变化通常拒绝严格续训；日志频率等可变；batch/world_size/精度变化若允许，标为非严格延续，不声称轨迹一致。
- 恢复粒度必须与存档频率匹配。step 模板要求完整累积窗口结束后的 optimizer-step 恢复，额外保存数据游标、顺序/增强状态和事件阈值；仍只支持 epoch 恢复的工程需单独保留可恢复存档，不能把中途权重假称为完整恢复点。细节见 [预算规范](training-budget.md#step-存档与恢复)。
- RNG 在会消耗随机数的初始化完成后、下一个训练数据迭代前恢复并验证顺序；seed 本身不能代替状态恢复。
- TensorBoard purge_step 按实际事件 step 设计。混用 epoch 与 micro-step 时，使用独立 event 子目录/Writer 各自清理，或统一坐标轴；一个 purge_step 不能正确处理多种量级的横轴。
- JSONL、summary 与恢复点对齐；延长训练的 scheduler 周期政策写入文档。
- 不承诺跨版本、硬件、world_size 的逐位复现；确定性选项与性能模式可配置并记录。

## 按需扩展

- EMA：记录普通与 EMA 指标区别、best 选择对象、更新次数，保存恢复对应状态。
- early stopping：monitor、min/max、min_delta、patience；patience 按验证次数计，跨 rank 一致停止。
- compile、激活检查点、channels_last、TF32：按硬件和模型验证收益兼容性，不默认假定可用。
- FSDP、流式数据中途恢复、多优化器：单独设计恢复和评估协议，不直接套用普通 DDP 方案。

## 参考来源

按项目安装版本核对 API；以下用于结构与语义参考，不引入整套依赖或任务无关的复杂度。

- [Torchvision 官方分类训练](https://github.com/pytorch/vision/blob/main/references/classification/train.py)：显式 epoch/evaluate 和主流程。
- [timm 训练脚本](https://github.com/huggingface/pytorch-image-models/blob/main/train.py)：累积、EMA、可选能力组织。
- [PyTorch AMP 示例](https://docs.pytorch.org/docs/main/notes/amp_examples.html)：累积、unscale、裁剪、更新顺序。
- [PyTorch 可复现说明](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md)：随机数、worker、确定性边界。
