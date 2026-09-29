# 目录与产物约定

## 推荐结构

以下为能力齐全时的结构，按实际职责创建文件，不生成空壳模块。已有项目保留其目录命名和导入方式。

```text
project/
├── train.py                    # 参数解析、初始化、训练编排
├── evaluate.py                 # checkpoint 独立评估
├── infer.py                    # 需要时提供推理接口
├── engine.py                   # train_one_epoch / validate
├── model/
│   ├── __init__.py             # build_model
│   └── network.py
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Dataset 和数据契约
│   ├── transforms.py          # 必要的数据变换
│   └── collate.py             # 变长数据等需要时创建
├── config/
│   ├── base.yaml              # 仅在实现合并机制时使用
│   ├── experiment.yaml
│   └── smoke.yaml             # 有界验证配置，不自动执行
├── scripts/
│   ├── train_single.sh
│   ├── train_ddp.sh
│   ├── resume.sh
│   └── evaluate.sh
├── utils/
│   ├── config.py              # 配置解析、校验、路径解析
│   ├── distributed.py
│   ├── checkpoint.py
│   ├── logging.py
│   └── reproducibility.py
├── losses.py
├── metrics.py                 # 可归并统计和任务指标
├── optim.py                   # optimizer / scheduler 构建
├── logs/
│   └── <run_id>/
│       ├── config.resolved.yaml
│       ├── metadata.json
│       ├── data_manifest.json
│       ├── train.log
│       ├── metrics.jsonl
│       ├── summary.json
│       ├── tensorboard/
│       └── figures/           # 有可视化时生成
├── weight/
│   └── <run_id>/
│       ├── last.pt
│       ├── best.pt
│       └── epoch_XXXX.pt      # 仅启用周期存档时生成
├── tests/
├── pyproject.toml             # 或沿用 requirements.txt 等依赖方案
├── README.md                  # 环境、配置、命令和指标定义
└── .gitignore
```

小项目可把少量 utils 合并为 training_utils.py；目录用于区分职责，文件数不是验收指标。已有 src/package 布局时，将代码放入现有包，不额外制造第二套导入方式。

## 数据与代码

- data/ 默认放数据处理代码，不存大规模原始数据。
- 原始数据、预处理缓存通过配置指向外部目录；需要本地目录时使用 datasets/ 和 cache/。
- 不复制、移动或覆盖原始数据来满足目录模板。
- 数据清单保存划分 ID、特征/标签语义、归一化、数据版本或合理成本的指纹，不塞入全部数据内容。
- 根据任务采用工况、实体、患者、时间段等划分单位；归一化及其他拟合型预处理只在训练集拟合。

## run_id 与路径

- 新训练由 rank 0 创建唯一 run_id 并广播；显式 ID 已存在时拒绝覆盖。可采用 experiment_name + UTC 时间 + 短随机后缀。
- logs/<run_id> 与 weight/<run_id> 一一对应；metadata 记录关联路径，checkpoint 包含重建必需的配置和预处理状态。
- 配置相对路径相对配置文件目录；继承配置时明确路径基准。CLI 相对路径相对调用目录，进入训练前统一解析。
- resume 默认继续原 run_id；若允许迁移目录，记录来源，不隐式分叉。init-from 创建新 run_id。
- 日志与权重可以位于不同磁盘。支持多节点时说明共享文件系统前提；没有共享存储时实现分发与保存策略。
- checkpoint 必须包含恢复核心状态，不能仅靠日志侧文件继续训练。

## 产物契约

| 产物 | 必要内容 |
| --- | --- |
| config.resolved.yaml | 合并及 CLI 覆盖后的配置、原始配置来源 |
| metadata.json | run_id、命令、UTC 开始时间、Python/PyTorch/CUDA/cuDNN、GPU、world size、可取得的代码版本、确定性设置 |
| data_manifest.json | 数据身份、划分、样本数、字段含义、预处理参数 |
| metrics.jsonl | epoch、micro-step、optimizer-step、指标、分母/覆盖率、LR、耗时；无效指标写 null |
| train.log | 带时间和 rank 的日志；错误和 traceback 进入文件或明确的 launcher stderr 日志 |
| summary.json | 状态、最好指标及 epoch、checkpoint 路径、耗时、终止原因 |
| checkpoint | 完整恢复状态；可独立取得推理所需结构与预处理 |

长期训练推荐追加 JSONL，避免每轮重写全部历史。恢复到旧 checkpoint 时，将之后的日志记录显式截断或归档，不混合旧尾部和新记录。

## 启动脚本与仓库

- scripts/ 仅封装 Python/torchrun 命令，不重复实现训练逻辑和超参数默认值。
- 正确引用含空格路径，使用脚本位置定位项目，不硬编码本机 Python 路径。
- 支持透传参数与选择解释器，rank 从 torchrun 环境获取。
- 提供实际实现的单卡、多卡、resume、init-from、evaluate 命令。
- 不默认选择忙碌 GPU、下载数据、安装依赖、后台启动或终止其他进程。
- .gitignore 通常排除 logs/、weight/、datasets/、cache/、临时权重和 Python 缓存；不排除 data/ 代码或 config/。
- 不为套用模板强制搬迁已有工程，不替用户提交 Git 或上传权重。
