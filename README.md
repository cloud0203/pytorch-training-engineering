# PyTorch Training Engineering

**把模型和数据接口，整理成可追溯、可恢复、经过明确验收的训练工程。**

A Chinese-language agent skill for building readable, configuration-driven PyTorch training projects, with explicit user approval before implementation.

这个 skill 为编码助手提供训练工程的工作流程、目录规范和正确性约束：先读取项目，形成方案，等待确认，再实现用户选定的能力。仓库交付的是 skill、参考规范和配置模板，不包含可直接训练任意模型的通用 Trainer。

[快速开始](#快速开始) · [能力范围](#能力范围) · [训练预算与实验比较](#训练预算与实验比较) · [确认机制](#确认机制) · [发布到 GitHub](docs/publishing.md)

## 适用场景

- 已有模型和 Dataset，需要补齐完整训练入口。
- 希望不同深度学习项目保持统一的目录、配置、日志和 checkpoint 约定。
- 现有脚本需要加入 TensorBoard、DDP、混合精度或断点恢复。
- 希望先审阅训练工程方案，再决定是否修改代码和执行验证。

默认采用原生 PyTorch、显式训练循环和轻量模块化。已有项目使用其他训练框架时，优先保留现有组织方式。回归、分类、分割、时序和物理场等任务采用各自的数据与指标契约；高级功能按需求选择。

## 快速开始

### 1. 获取项目

克隆本仓库，或在 GitHub 页面下载 ZIP 后解压。

```bash
git clone https://github.com/cloud0203/pytorch-training-engineering.git
cd pytorch-training-engineering
```

### 2. 安装到 Codex

只安装 `skills/pytorch-training-engineering/`，无需安装整个仓库的开发工具。

当前官方文档列出的用户级目录为 `~/.agents/skills/`，项目级目录为 `.agents/skills/`。以下为 Bash 下的用户级安装示例；目标已存在时先比较内容，不直接覆盖。[Codex 官方说明](https://learn.chatgpt.com/docs/build-skills)

```bash
mkdir -p "$HOME/.agents/skills"
test ! -e "$HOME/.agents/skills/pytorch-training-engineering" && \
  cp -R skills/pytorch-training-engineering "$HOME/.agents/skills/"
```

也可请求已有的 skill-installer 从你的 GitHub 仓库安装，指定子目录 `skills/pytorch-training-engineering`。如果客户端已通过其他路径（例如现有的 `~/.codex/skills/`）管理本 skill，沿用该环境的安装方式，不重复安装同名副本。安装后若列表未刷新，重新启动客户端。

### 3. 明确调用

```text
使用 $pytorch-training-engineering，为当前项目搭建训练工程。
先读取模型和数据接口，提出目录结构、拟修改文件和功能方案。
等我确认后再实施，不要启动训练。
```

助手应先只读检查并提供可审阅方案。你确认后，它才在约定范围内修改文件和执行检查。

### 不使用 skill 安装机制

复制 [独立 Prompt](skills/pytorch-training-engineering/assets/standalone-prompt.md) 中的完整提示词到支持代码工作的助手，补充项目路径和执行范围。该方式依赖助手遵循提示词；Codex 专用元数据不适用于其他客户端。

## 能力范围

以下是 skill 对生成工程的要求与可选扩展，实际实现由项目需求及确认方案决定。

| 领域 | 约定 |
| --- | --- |
| 目录与配置 | model/data/logs/weight/scripts/config；配置校验、CLI 覆盖、最终配置快照 |
| 数据 | 数据契约、固定验证协议、防泄漏、训练集拟合预处理 |
| 训练循环 | 显式 train/validate、可替换模型和损失、合理的 DataLoader 配置 |
| 优化 | 可配置优化器、warmup/调度器语义、梯度裁剪、按需累积 |
| 实验预算 | 成功更新数/数据量/epoch 的选择、有效全局 batch、统一调度与评估横轴 |
| 精度与设备 | CPU、单 GPU、DDP；FP32/FP16/BF16 按设备和模型适配 |
| 指标 | 全局统计、正确分母、分通道/区域指标、本地 step 与全局 epoch 区分 |
| 观测 | TensorBoard、文本与 JSONL、阶段耗时、吞吐量、峰值显存 |
| 恢复 | best/last、原子保存、完整训练状态、resume 与 init-from 分离 |
| 复现 | RNG、sampler/worker 状态、环境元信息和明确复现边界 |
| 验收 | 静态检查、指标测试、有界试跑、恢复与多卡一致性，按授权执行 |

EMA、early stopping、compile、激活检查点和 FSDP 等只在项目需要时实现；保留的配置项必须实际生效或明确报错。

## 训练预算与实验比较

**相同 epoch 不等于相同更新次数，相同 step 也不等于相同数据量。**

例如训练集为 12,800 个样本，没有尾批、重复采样或 AMP 跳步时：

| 有效全局 batch | 每 epoch 更新数 | 10 epoch 更新数 | 10 epoch 样本暴露量 |
| --- | --- | --- | --- |
| 32 | 400 | 4,000 | 128,000 |
| 64 | 200 | 2,000 | 128,000 |

若改成两组都训练 4,000 step，batch=64 的实验会处理两倍数据。因此先确定要控制什么：

- **模型/损失消融**：优先固定有效全局 batch 和成功更新数。显存不足可降低每卡 batch、增加梯度累积，但批内交互等模型行为仍需核对。
- **研究 batch 本身**：明确固定样本/token 总量还是固定更新数，报告另一项和耗时；必要时分别比较。
- **复现既有配方**：可继续使用 epoch，但记录实际更新次数、采样和调度政策。

配置模板 schema 2 默认采用以下协议，数值是示例，需要按任务选定：

```yaml
training:
  budget:
    unit: optimizer_step
    limit: 10000
  target_effective_batch_size: 32
  accumulation_steps: 4  # 默认单卡、每卡 batch=8
scheduler:
  interval: optimizer_step
  total_steps: 10000
  warmup_steps: 500
validation:
  interval: optimizer_step
  every_n_steps: 250
  at_end: true
checkpoint:
  interval: optimizer_step
  every_n_steps: 1000
  resume_granularity: optimizer_step
```

这段是完整模板的节选，不是独立可运行配置。4 卡、每卡 batch=8 时，将累积次数改为 1 可保持同样的名义有效 batch；无法满足目标时要求明确报错，不自动改变 LR 或预算。

调度器只随成功更新推进，验证和保存事件去重，记录更新数、样本/token 量与耗时。step checkpoint 可能位于 epoch 中间，生成工程必须实现对应的数据游标和随机状态恢复，或明确保留 epoch-only 恢复方案，不能只存 epoch+1。

完整的预算选择、计数、尾批、跳步和迁移规则见 [训练预算与实验可比性](skills/pytorch-training-engineering/references/training-budget.md)。参考了 [timm 的 update 调度](https://github.com/huggingface/pytorch-image-models/blob/main/train.py) 和 [Hugging Face 的 max_steps 配置](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)；本仓库仍提供规范和模板，并未加入实际训练器。

## 确认机制

本 skill 设置：

```yaml
policy:
  allow_implicit_invocation: false
```

因此普通训练相关请求不会自动启用它；你需要明确调用。该调用允许只读分析和提方案，**不直接授权修改工程或启动训练**。这是 Codex 的显式调用设置加上 skill 内的操作流程约束，不是额外的操作系统权限隔离。[调用策略说明](https://learn.chatgpt.com/docs/build-skills)

```text
明确调用 → 只读检查 → 提交方案 → 用户确认 → 执行约定修改与检查
```

- 未回复不视为确认。
- 已确认范围内继续完成，不重复索要同一项许可。
- 新增超出范围的操作，先更新方案并确认。
- “不要启动训练”也包含 dry-run、smoke test 和间接调用训练的测试。
- 正式训练只有在用户明确要求启动时才执行。

## 推荐生成的工程结构

```text
your-project/
├── train.py
├── engine.py
├── evaluate.py
├── model/                    # 网络与构建接口
├── data/                     # 数据处理代码
├── config/                   # 实验配置
├── scripts/                  # 启动与评估命令
├── utils/                    # 日志、分布式、恢复等
├── losses.py
├── metrics.py
├── optim.py
├── logs/<run_id>/            # 配置、日志、指标、TensorBoard
├── weight/<run_id>/          # best.pt、last.pt、周期存档
└── tests/
```

原始数据由配置指向外部路径，或放入独立的 `datasets/`；`data/` 用于代码。日志与权重通过相同 `run_id` 关联。已有工程保留等价目录，小项目可以合并工具文件。详见 [目录与产物约定](skills/pytorch-training-engineering/references/project-layout.md)。

## 使用示例

**只做设计：**

```text
使用 $pytorch-training-engineering，梳理这个时序预测项目的训练方案。
只输出目录、数据划分、功能和验收计划，不修改文件，不启动训练。
```

**补齐已有脚本：**

```text
使用 $pytorch-training-engineering，检查现有训练脚本。
我需要 TensorBoard、训练耗时、双卡 DDP 和 epoch 边界续训。
先给出拟修改文件、配置变化和验证范围，等我确认。
```

**在方案确认后授权有限试跑：**

```text
确认刚才的修改方案。允许 CPU 合成数据试跑，最多 2 个 epoch，
每轮训练和验证各最多 3 个 batch，输出到独立临时目录。
不要启动正式训练，也不要使用 GPU。
```

**比较不同显存配置：**

```text
使用 $pytorch-training-engineering，比较每卡 batch=8 和 batch=4。
固定有效全局 batch=32、成功更新 10000 次、warmup 500 次，
每 250 次更新验证，按设备数量推导并核对梯度累积次数。
先说明预算、尾批和续训方案，等我确认；不要启动训练。
```

## 仓库结构

```text
pytorch-training-engineering/
├── README.md
├── CONTRIBUTING.md
├── docs/publishing.md
├── skills/pytorch-training-engineering/
│   ├── SKILL.md
│   ├── agents/openai.yaml
│   ├── references/
│   │   ├── project-layout.md
│   │   ├── training-contract.md
│   │   ├── training-budget.md
│   │   └── acceptance.md
│   └── assets/
│       ├── config.example.yaml
│       └── standalone-prompt.md
├── scripts/validate_skill.py
├── requirements-dev.txt
└── .github/workflows/validate.yml
```

`SKILL.md` 是助手入口；`references/` 按任务读取；`assets/` 提供可适配材料。仓库级脚本用于维护 skill，不会训练模型。

## 本地校验

校验需要 Python 3.10+ 与 PyYAML；使用 skill 本身不需要安装这些依赖，也不需要 GPU 或 PyTorch。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/validate_skill.py
```

检查 frontmatter、元数据、显式调用策略、YAML、Markdown 本地文件链接、可移植性，以及 schema 2 示例中的预算/调度/事件单位一致性。它不访问外部链接、不验证 Markdown 锚点、不启动训练，也不证明生成训练工程的运行正确性。GitHub Actions 会运行同一校验。

## 贡献与发布

提交改动前阅读 [贡献说明](CONTRIBUTING.md)。首次上传、仓库设置和版本发布步骤见 [发布指南](docs/publishing.md)。

本项目采用 [MIT License](LICENSE)。

## 参考资料

- [PyTorch AMP examples](https://docs.pytorch.org/docs/main/notes/amp_examples.html)
- [Torchvision reference training](https://github.com/pytorch/vision/blob/main/references/classification/train.py)
- [timm training script](https://github.com/huggingface/pytorch-image-models/blob/main/train.py)

这些链接用于核对设计与实现语义。具体 API 应匹配目标项目安装的版本。
