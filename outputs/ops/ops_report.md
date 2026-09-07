# 数据运维巡检报告

- 生成时间：2026-09-07 21:59:37
- 巡检范围：`/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/logs` / `/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/data` / `/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/outputs`
- **健康度：70.2 / 100　等级 C** —— 及格：有明显缺项，交付前需整改
- run 数：1　告警：ERROR 3 / WARN 6 / INFO 1
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
| `demo` | 1.2 KB | 0.0天前 | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |

## 3. 存储占用

### 3.1 分区域账

| 区域 | 占用 | 说明 |
|---|---|---|
| `data` | 11.1 MB | MNIST 原始数据集（实际统计自 `/Users/leechiann/.cache/paddle/dataset/mnist`，原因见告警 S10） |
| `ops` | 179.7 KB | 巡检自身报告与看板 |
| `logs` | 1.2 KB | 训练产物（日志/权重/记录/曲线） |
| `inputs` | 286 B | 用户自提原图（新数据入口） |
| `outputs_examples` | 0 B | 测试集示例输入图 + 识别拼图 |
| `outputs_custom` | 0 B | 用户自提图的预处理结果 |
| **合计** | **11.2 MB** | |

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

## 7. 告警清单（按级别）

| 级别 | 编号 | 维度 | 对象 | 问题 | 建议动作 |
|---|---|---|---|---|---|
| ERROR | C1 | 资产完整性 | `demo/model.pdparams` | 缺少最终模型权重（model.pdparams） | 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录 |
| ERROR | C1 | 资产完整性 | `demo/best_model.pdparams` | 缺少最佳模型权重（best_model.pdparams） | 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录 |
| ERROR | C1 | 资产完整性 | `demo/config.json` | 缺少参数快照（config.json） | 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录 |
| WARN | C1 | 资产完整性 | `demo/history.json` | 缺少训练过程数据（history.json） | 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录 |
| WARN | C1 | 资产完整性 | `demo/curves.png` | 缺少训练曲线图（curves.png） | 重跑该 run 的训练；若是中断所致请勿复用半成品 run 目录 |
| WARN | C3 | 资产完整性 | `/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/outputs/examples/inference_results.json` | 尚无推理成果 inference_results.json | 运行：python main.py --mode inference --run_name <run名> |
| WARN | U5 | 资产完整性 | `/Volumes/LCA_SSD/课程/大数据综合实训/handwriting_recognition/outputs/custom/custom_results.json` | 已有 3 张输入图但没有 custom_results.json | 运行：python main.py --mode custom --run_name <run名> |
| WARN | S10 | 数据源健康 | `/Users/leechiann/.cache/paddle/dataset/mnist` | 实际数据目录为 /Users/leechiann/.cache/paddle/dataset/mnist，与 config.data_dir=./data 不一致（说明当时触发了退回 Paddle 默认缓存的降级分支） | 在报告中注明降级；如需固定路径请补齐 config.data_dir 下的文件 |
| WARN | N13 | 可追溯性 | `demo/dataset_record.json` | 留档声明数据目录 ./data，实际命中 /Users/leechiann/.cache/paddle/dataset/mnist | 留档与事实不符，追溯时以实际为准 |
| INFO | T3 | 可追溯性 | `demo` | 无 best_model.pdparams，inference 模式将回退用末轮权重 | 推理结果可能与训练报告里的 val_acc 不一致 |

## 8. 整改优先级

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
7. **[WARN] 已有 3 张输入图但没有 custom_results.json**  
   → 运行：python main.py --mode custom --run_name <run名>
8. **[WARN] 实际数据目录为 /Users/leechiann/.cache/paddle/dataset/mnist，与 config.data_dir=./data 不一致（说明当时触发了退回 Paddle 默认缓存的降级分支）**  
   → 在报告中注明降级；如需固定路径请补齐 config.data_dir 下的文件

> 本报告由 `data_ops.py` 自动生成（只读巡检，不会修改/删除任何产物）。
