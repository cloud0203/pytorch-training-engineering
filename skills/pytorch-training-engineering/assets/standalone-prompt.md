# 可独立复制的训练工程 Prompt

复制下面代码块，填写已知项；未知项可以保留“从项目读取”。本 prompt 不依赖 skill 文件。先给出方案并等待确认，确认后生成/完善代码并做约定检查，不启动训练；如需只出设计，修改执行范围。

```text
请为我的深度学习项目构建或完善一套原生 PyTorch 训练工程。

项目信息：
- 项目路径：从当前工作目录读取。
- 任务类型、数据格式与划分方式：从项目读取；无法判断时询问关键缺失项。
- 模型入口及输入/输出 shape、dtype、单位：从项目读取。
- 损失与主验证指标：优先沿用已有定义；新增选择说明理由。
- 硬件、GPU 数量、依赖版本：读取可用信息，不假定所有 GPU 空闲。
- 执行范围：确认前仅做只读分析；先列出拟修改文件、功能和验证范围，等待我明确确认后，才生成或修改代码、配置和文档，并执行约定检查。不要启动训练，包括 dry-run、小数据训练、正式训练及间接调用训练的测试。

一、工作方式与代码风格
先只读检查项目规则、模型、数据、训练入口及环境，形成具体可审阅的实施方案并等待我确认；未回复不视为同意。当前任务已确认的方案无需重复确认，超出范围时重新确认新增部分。能从代码判断的信息不要重复询问，只询问会改变实现的关键缺失项。遵循当前会话的代理协作约定。
默认采用配置驱动、显式训练循环、轻量模块化；已有训练框架则沿用，不强行迁移。使用常规 Python 多行风格，独立语句分行，不用分号压缩代码。避免无必要的注册系统和复杂继承。

二、目录结构
新项目建议：
train.py：入口和训练编排。
engine.py：train_one_epoch、validate。
evaluate.py：从 checkpoint 独立评估；需要时另提供 infer.py。
model/：模型及构建入口。
data/：Dataset、数据变换、collate，存代码而非大规模原始数据。
config/：实际实验配置及有界试跑配置。
scripts/：单卡、多卡、续训、初始化和评估命令封装。
utils/：配置、分布式、日志、checkpoint、随机数工具；小项目可合并。
losses.py、metrics.py、optim.py：按规模拆分，不创建无用空壳。
logs/<run_id>/：配置快照、环境元信息、数据清单、train.log、metrics.jsonl、summary.json、tensorboard/、必要图像。
weight/<run_id>/：last.pt、best.pt、按需周期存档。
tests/、依赖文件、README.md、.gitignore。
已有工程保持其等价目录，不强制搬迁。原始数据和缓存用配置指向外部路径，必要时用 datasets/、cache/。logs 与 weight 使用同一 run_id；新实验禁止覆盖旧产物。

三、配置与数据契约
配置集中管理，CLI 覆盖优先级明确，启动时校验类型、范围、冲突和未知字段，保存最终生效配置。配置相对路径相对配置文件，CLI 相对路径相对调用目录。
定义输入输出 shape/dtype、mask、样本 ID、单位、损失空间及指标分母。按任务防止工况/实体/时间泄漏，拟合型预处理仅在训练集拟合，验证划分与采样固定，测试集不用于选 best。
提供合理的 DataLoader 配置，num_workers=0 可用；persistent_workers 启用时正确传播 epoch/采样状态。

四、训练能力
单卡、CPU、torchrun DDP 共用主体逻辑；不支持的设备或组合明确报错。
优化器与调度器可配置，AdamW 可作初始默认，按需加入其他实现。明确 warmup、总周期、最小 LR 和 scheduler 按 epoch/成功参数更新/验证指标推进的语义，记录每参数组实际 LR。
精度显式支持所实现的 FP32/FP16/BF16，检查设备能力；按需 GradScaler，先 unscale 再裁剪。验证关闭梯度。处理 NaN/Inf，错误包含 epoch、step、rank 和可取得的样本 ID，不静默跳过。
梯度累积可配置，区分 micro-step 与成功 optimizer-step；正确处理最后不足窗口及不同有效样本/token 数。DDP 非最终累积步的 no_sync 覆盖 forward 和 backward。AMP 跳过参数更新时，按更新计数的 scheduler/EMA 不错误推进。
EMA、early stopping、compile、激活检查点、TF32、FSDP、多优化器仅按任务需要实现；不要生成无效开关或宣称未完成能力。

五、多卡与指标正确性
train sampler 每轮 set_epoch，说明补齐/drop_last 策略及实际处理数。验证不重复补齐样本，覆盖不等长分片和某 rank 零样本；若绕过 DDP wrapper，先确保必要模型 buffer 一致，确认 forward 内无不匹配通信。
所有 rank 参与正确的指标聚合，仅全局 rank 0 写共享日志和权重。best、early stopping 与退出决策跨 rank 一致。失败不能使其他 rank 永久等待。
损失项与业务指标分开，定义按样本/点/像素/token 加权；不要平均各 batch 或 GPU 的 R²、F1 等非线性指标。近零分母、空 mask、无效通道、宏微平均有明确政策；未定义指标写 null。主指标及 min/max 可配置。

六、日志、性能与产物
提供控制台、文件、JSONL 和可关闭的 TensorBoard，使用同源指标。
记录本地 step 损失并明确本地口径，epoch 记录全局 train/val 指标、LR、损失分量；按需记录梯度范数、AMP scale、跳步、各通道/分组指标。
记录数据准备、DataLoader 阻塞、训练、验证、保存、总耗时、吞吐量和峰值显存。注明计时边界和单位，CUDA 阶段边界按需同步，不默认每 batch 同步；loader 等待不等于全部 I/O 时间，各 rank 逐项最大值不能直接相加。
可视化使用固定验证样本 ID，频率可控。metadata 保存启动命令、环境、设备、代码版本（可取得时）、数据身份、随机设置。summary 保存 best、耗时、状态及终止原因。

七、checkpoint 与复现
原子保存 last/best，按需周期存档及数量限制。checkpoint 包含模型、构建配置、预处理、数据身份、优化器、调度器、scaler、epoch、两类 step、best、各 rank RNG、world_size，以及启用的 EMA/early stopping 等状态。
resume 恢复状态，init-from 只加载权重并开新实验，两者互斥。列出兼容性规则；改变 batch/world_size/精度等若允许，不能声称严格复现。默认只承诺 epoch 边界恢复，不把加载权重称为完整续训。
初始化完成后、继续数据迭代前正确恢复 RNG；考虑 worker、sampler 和独立 generator。续训默认保留原调度周期，延长训练的行为写清楚。
恢复点之后的 JSONL/TensorBoard 记录正确处理；epoch 和 micro-step 混用时分开 event writer 清理或统一坐标轴，不能用一个 purge_step 错删不同横轴的事件。
推理可从 checkpoint 重建，不依赖训练机器的绝对路径；不承诺跨软硬件版本逐位复现。

八、验收与交付
在本次执行范围内运行格式、配置、静态及不启动训练的适当逻辑检查。不要运行训练；把需要训练才能验证的内容列为未执行。
为后续授权试跑准备有界命令和检查：合成/真实小数据闭环、连续与中断恢复一致性、单卡/DDP 固定模型评估一致性、累积尾批、AMP 跳步、best 选择、日志恢复。
提供实际文件清单、目录说明、配置字段、单卡/多卡/续训/初始化/评估命令、指标与恢复语义、已执行检查证据及未验证项。不要只给伪代码，不因缺 GPU 虚构 CUDA/DDP 通过。不要启动后台作业。
```

将执行范围替换为“只输出设计和目录方案，不修改代码、不运行训练”，可用于设计评审。需要训练验证时，应明确允许的设备、数据量与步数；正式训练另行明确启动命令和目标。
