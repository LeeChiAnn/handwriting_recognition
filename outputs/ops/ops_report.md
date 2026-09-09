# 数据运维巡检报告

- 生成时间：2026-09-09 08:45:44
- 巡检范围：`/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/logs` / `/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/data` / `/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/outputs`
- **健康度：70.2 / 100　等级 C** —— 及格：有明显缺项，交付前需整改
- run 数：1　告警：ERROR 3 / WARN 7 / INFO 1
- 最优 run：`None`（best_val_acc=None）

## 1. 五维健康度

| 维度 | 权重 | 得分 | 评级 |
|---|---|---|---|
| 资产完整性 | 25% | 0 | E |
| 一致性校验 | 25% | 100 | A |
| 可追溯性 | 20% | 85 | B |
| 数据源健康 | 15% | 88 | B |
| 存储治理 | 15% | 100 | A |

## 2. 资产台账（run × 产物）

| run | 体积 | 最后更新 | 权重 | 参数 | 过程 | 曲线 | 数据留档 | 日志 |
|---|---|---|---|---|---|---|---|---|
| `demo` | 1.2 KB | 1.5天前 | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |

## 3. 存储占用

### 3.1 分区域账

| 区域 | 占用 | 说明 |
|---|---|---|
| `data` | 11.1 MB | MNIST 原始数据集（实际统计自 `/Users/leechiann/.cache/paddle/dataset/mnist`，原因见告警 S10） |
| `outputs_reports` | 417.6 KB |  |
| `ops` | 368.2 KB | 巡检自身报告与看板 |
| `logs` | 1.2 KB | 训练产物（日志/权重/记录/曲线） |
| `inputs` | 286 B | 用户自提原图（新数据入口） |
| `outputs_examples` | 0 B | 测试集示例输入图 + 识别拼图 |
| `outputs_custom` | 0 B | 用户自提图的预处理结果 |
| `outputs_eval` | 0 B |  |
| **合计** | **11.8 MB** | |

### 3.2 按文件类型

| 类别 | 占用 |
|---|---|
| 日志(.log) | 904 B |
| 记录(.json) | 376 B |
| 模型权重(.pdparams) | 0 B |
| 图片(.png) | 0 B |
| 其他 | 0 B |

## 4. 训练指标（运维视角）

| run | 轮数 | train_acc | val_acc | 泛化间隙 | 最优轮 | best_val_acc |
|---|---|---|---|---|---|---|
| `demo` | - | - | - | - | - | -（无 history.json，指标不可追溯）|

## 5. 数据源体检

- 实际数据目录：`/Users/leechiann/.cache/paddle/dataset/mnist`
- 通过 IDX 校验：4 / 4
- 指纹基线：`/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/outputs/ops/data_fingerprint.json`

| 文件 | 存在 | 体积 | 样本数 | 尺寸 | IDX | md5 | 基线 |
|---|---|---|---|---|---|---|---|
| `train-images-idx3-ubyte.gz` | ✅ | 9.5 MB | 60000 | 28x28 | ✅ | `f68b3c2dcb` | ✅ |
| `train-labels-idx1-ubyte.gz` | ✅ | 28.2 KB | 60000 | - | ✅ | `d53e105ee5` | ✅ |
| `t10k-images-idx3-ubyte.gz` | ✅ | 1.6 MB | 10000 | 28x28 | ✅ | `9fb629c418` | ✅ |
| `t10k-labels-idx1-ubyte.gz` | ✅ | 4.4 KB | 10000 | - | ✅ | `ec29112dd5` | ✅ |

## 6. 用户自提输入区（新数据接入体检）

- 输入目录：`/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/inputs`（存在），现有 PNG 3 张，因格式被排除 0 个
- 成果：`/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/outputs/custom/custom_results.json`（尚未生成）

## 7. 模型效果验证与离线成果包

- ❌ 尚无全测试集验证结果，模型效果只能靠训练日志的 val_acc 间接推断

### 7.1 可下载的离线成果 HTML

| 成果 | 用途 | 状态 | 体积 | 生成时间 |
|---|---|---|---|---|
| `index.html` | 入口页：三张卡片跳转到下面两份 + 运维看板 | ✅ | 5.8 KB | 2026-09-09 08:26:05 |
| `model_report.html` | 模型效果验证：混淆矩阵 / 每类 P·R·F1 / 置信度 / 错分图集 | ✅ | 4.4 KB | 2026-09-09 08:26:05 |
| `lab_archive.html` | 实验档案袋：历次 run 参数对比、曲线、全部图像成果 | ✅ | 200.6 KB | 2026-09-09 08:26:05 |

> 成果包目录：`/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/outputs/reports`；每份都是单文件自包含（图片已 base64 内嵌），下载任意一份到本地双击即可观看。

## 8. 告警清单（按级别）

| 级别 | 编号 | 维度 | 对象 | 问题 | 建议动作 |
|---|---|---|---|---|---|
| ERROR | C1 | 资产完整性 | `demo/model.pdparams` | 缺少最终模型权重（model.pdparams） | 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录 |
| ERROR | C1 | 资产完整性 | `demo/best_model.pdparams` | 缺少最佳模型权重（best_model.pdparams） | 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录 |
| ERROR | C1 | 资产完整性 | `demo/config.json` | 缺少参数快照（config.json） | 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录 |
| WARN | C1 | 资产完整性 | `demo/history.json` | 缺少训练过程数据（history.json） | 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录 |
| WARN | C1 | 资产完整性 | `demo/curves.png` | 缺少训练曲线图（curves.png） | 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录 |
| WARN | C3 | 资产完整性 | `/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/outputs/examples/inference_results.json` | 尚无推理成果 inference_results.json | 运行：python main.py --mode inference --run_name <run名> |
| WARN | E1 | 资产完整性 | `/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/outputs/eval/eval_results.json` | 从未做过全测试集验证（缺 outputs/eval/eval_results.json） | python main.py --mode eval --run_name <run名>（需 paddle，请在 AI Studio 上跑） |
| WARN | U5 | 资产完整性 | `/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/outputs/custom/custom_results.json` | 已有 3 张输入图但没有 custom_results.json | 运行：python main.py --mode custom --run_name <run名> |
| WARN | S10 | 数据源健康 | `/Users/leechiann/.cache/paddle/dataset/mnist` | 实际数据目录为 /Users/leechiann/.cache/paddle/dataset/mnist，与 config.data_dir=./data 不一致（说明当时触发了退回 Paddle 默认缓存的降级分支） | 在报告中注明降级；如需固定路径请补齐 config.data_dir 下的文件 |
| WARN | N13 | 可追溯性 | `demo/dataset_record.json` | 留档声明数据目录 ./data，实际命中 /Users/leechiann/.cache/paddle/dataset/mnist | 留档与事实不符，追溯时以实际为准 |
| INFO | T3 | 可追溯性 | `demo` | 无 best_model.pdparams，inference 模式将回退用末轮权重 | 推理结果可能与训练报告里的 val_acc 不一致 |

## 9. 整改优先级

1. **[ERROR] 缺少最终模型权重（model.pdparams）**  
   → 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录
2. **[ERROR] 缺少最佳模型权重（best_model.pdparams）**  
   → 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录
3. **[ERROR] 缺少参数快照（config.json）**  
   → 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录
4. **[WARN] 缺少训练过程数据（history.json）**  
   → 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录
5. **[WARN] 缺少训练曲线图（curves.png）**  
   → 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录
6. **[WARN] 尚无推理成果 inference_results.json**  
   → 运行：python main.py --mode inference --run_name <run名>
7. **[WARN] 从未做过全测试集验证（缺 outputs/eval/eval_results.json）**  
   → python main.py --mode eval --run_name <run名>（需 paddle，请在 AI Studio 上跑）
8. **[WARN] 已有 3 张输入图但没有 custom_results.json**  
   → 运行：python main.py --mode custom --run_name <run名>

> 本报告由 `data_ops.py` 自动生成（只读巡检，不会修改/删除任何产物）。
