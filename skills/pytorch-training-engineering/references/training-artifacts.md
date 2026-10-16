# 权重目录日志与 loss 曲线

生成训练工程时，`weight/<run_id>/` 同时交付权重、文本记录和静态 loss 图。用户复制该目录后无需启动 TensorBoard 即可查看训练结果。本规范描述生成工程需要实现的行为，不代表 skill 仓库自身会执行训练或生成真实实验图。

## 必需产物

```text
weight/<run_id>/
├── best.pt
├── last.pt
├── train.log
├── summary.txt
├── loss_history.csv
├── loss.png
└── loss_log.png
```

- `train.log`：UTF-8 文本，包含开始/结束时间、恢复会话、每轮 train loss、实际验证事件的 val loss、当前学习率、epoch 和 optimizer_step、各阶段耗时及累计运行耗时。按事件及时 flush，运行期间即可查看。
- `summary.txt`：正常结束或 early stopping 后记录状态、停止原因、目标/实际预算、最终指标、best 指标及位置、开始/结束时间、累计运行秒数和图像生成状态。停机间隔不计入训练运行耗时；标注耗时是否包含绘图与日志。
- `loss_history.csv`：供绘图和分析读取的数值历史，不从格式化文本日志或 TensorBoard 反解析。由与 `logs/<run_id>/metrics.jsonl` 相同的全局指标事件生成。
- `loss.png`：线性纵轴 train/val loss 图；`loss_log.png`：相同数据和横轴的对数纵轴图。图像在正常训练结束时生成，绘图完成后更新摘要。

文件应实际位于权重目录，不能只给链接、符号链接或提示用户去 logs/ 查找。`logs/` 可继续保存完整机器日志和 TensorBoard；同一个指标事件只计算一次，再分发到这些输出，避免两套独立统计漂移。

## 记录粒度与字段

每个完整数据 epoch 结束写一次全局加权 train loss，即使主要训练预算是 optimizer_step；周期验证按原预算协议执行，不因增加日志而强制每 epoch 验证。预算在 epoch 中途结束时写 partial_epoch，注明未完成。

CSV 至少包含以下字段，字段名可按已有项目等价映射：

| 字段 | 语义 |
| --- | --- |
| event_id / event_type | 唯一事件标识；epoch_end、validation、partial_epoch 等 |
| timestamp | 带时区的实际事件时间 |
| epoch / epoch_complete | 面向用户从 1 开始的 epoch 编号及是否完整，区别于 checkpoint 的零起始 data_epoch |
| micro_step / optimizer_step | 事件发生时的真实进度；保留数据量预算需要的 seen 计数 |
| train_loss / val_loss | 全局聚合后的原始损失；不适用于当前事件的列留空 |
| loss_name / loss_space | 优化目标名称、归一化/物理空间等；组合损失可另加分量列 |
| lr | 实际使用的 LR；有多组或轮内变化时记录各组及起止值，不把末值假称为整轮常量 |
| train_seconds / val_seconds / checkpoint_seconds | 明确边界的阶段耗时，不适用的列留空 |
| epoch_seconds / elapsed_seconds | 本轮及累计运行耗时（秒）；partial epoch 和恢复会话注明统计范围 |

train_loss 按任务定义的样本/元素权重汇总，不能平均不同大小 batch 的均值，也不能用 rank 0 最后一个 batch 代替整轮结果。step 中途恢复若要报告完整 epoch 均值，需恢复累计损失分子/分母与本轮已有计时；无法恢复时标为恢复后的局部窗口，不冒充完整轮次。

验证事件使用自己的真实 optimizer_step/epoch 位置。未验证的 epoch 不填 0、不前向填充、不挪用上次 val loss；若同一 epoch 验证多次，分别保留。epoch 末与验证恰好同一时点可以合并，但不能重复计入绘图序列。

## 绘图规则

- 默认横轴跟随主预算：step 预算用 optimizer_step，epoch 预算用 epoch，数据量预算用相应 seen 计数；每条曲线使用自己实际记录的位置。图例写明 Train/Validation、损失名称/空间，标题含 run_id，横纵轴有标签。
- 使用未平滑原始 loss；如额外提供平滑图，需要单独标识，不替换原始图。Train 和 Validation 的损失定义/空间不同时分面或明确标注，不能伪装成同一种目标。
- “对数曲线”默认指原始 loss 的对数纵轴，不把 CSV 值改为 log(loss)，也不先取 log 再设对数轴。
- 对数图只显示有限且严格为正的点。零、负值、NaN/Inf 在绘图副本中置为缺失、断开相关线段，并在图上注明每条曲线排除的数量与原因；禁止绝对值、任意加 epsilon 或裁剪为小正数。
- 线性图保留有限的零/负 loss，非有限值注明并跳过。原始数值历史不因绘图而改写；缺失值与非有限值的原因通过事件状态/日志保留。
- 没有任何正值时仍生成 `loss_log.png`，写明“无可用于对数轴的正 loss”，不画假数据；缺少验证数据时只画 train 并标明未验证。单点使用 marker，空历史生成明确的无数据说明。
- 使用无图形界面的绘图库后端，保存 PNG 后关闭 figure。提供独立重绘入口（例如生成工程中的 `scripts/plot_loss.py --history ... --output-dir ...`），只读取 CSV，不加载模型、数据集或启动训练。

## 生命周期与恢复

- 仅全局 rank 0 写共享产物。训练事件及时写 log/CSV；结束前先 flush 历史、保存最终 checkpoint，再绘图，最后写摘要。文件名与配置一致，不能仅写配置却缺少实际调用。
- 普通结束和 early stopping 都完成最终导出；异常时优先保留已写历史和原始错误，尽力输出失败摘要，不能为绘图进行不安全的 DDP 通信，也不能用绘图错误覆盖训练异常。
- CSV、summary 和图像使用临时文件后原子替换等方式避免半成品；追加 CSV 可逐条 flush，恢复时检测并处理不完整尾行。
- 续训从有效恢复点对齐 JSONL/CSV/log：以保存的事件游标和数据消费进度识别旧尾部，先归档或截断；不能仅按 optimizer_step 丢弃包含 AMP 跳步或同一步多个合法事件的记录。
- 重新开始运行时将旧图标为过期或归档，正常结束后根据去重的有效全历史重绘；不要把上次结束图当成本次最终图。跨会话累计耗时只累计实际运行区间。
- 绘图失败需在 train.log/summary 中明确报告并给出重绘方法，不能声称图像生成成功；已保存权重保持可用。
