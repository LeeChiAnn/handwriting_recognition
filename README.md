# 手写数字识别实训项目（邯郸职业技术学院大数据综合实训 · 深度学习入门）

基于百度 AI Studio 经典 MNIST 手写识别项目改造，面向**邯郸职业技术学院大数据综合实训之深度学习实训课程**。
改造后满足 5 项教学要求：

| 要求 | 落地方式 |
|---|---|
| ① 可调参数（学习率 / epochs / 网络结构…） | `config.py` 集中管理，命令行一键覆盖 |
| ② 数据与日志维护机制 | `dataset_record.json` 数据集加载记录 + `train_*.log` 训练日志 + `history.json` |
| ③ 输入图片示例 + 识别结果 | `inference.py` 生成 `sample_*.png` 与 `predictions.png` 拼图 |
| ④ 训练过程可视化 | `visualize.py` 画损失/准确率曲线 `curves.png` |
| ⑤ 多 .py 模块 + 教学 .ipynb | 代码按功能拆分；`手写识别教学讲解.ipynb` 仅讲解，实际跑 .py |
| ⑥ 追加：数据运维实训 | `data_ops.py` 只读巡检全部产物 → 五维健康度评分 + 告警清单 + `dashboard.png` / `dashboard.html` 可视化看板（见 §6） |

---

## 1. 目录结构

```
handwriting_recognition/
├── config.py                  # 可调参数（dataclass + argparse）
├── logger.py                  # 日志机制（控制台 + 文件）
├── data_utils.py              # 数据加载 + 加载记录
├── model.py                   # 三种网络结构（mlp/lenet/cnn）
├── visualize.py               # 曲线 + 推理结果拼图
├── train.py                   # 训练流程
├── inference.py               # 推理 + 示例输出（MNIST 示例 / 用户自提图片两条通路）
├── custom_input.py            # 把任意尺寸 PNG 预处理成 MNIST 风格 28×28（不需 paddle）
├── data_ops.py                # 【数据运维】只读巡检 + 五维评分 + 告警（不需 paddle）
├── ops_dashboard.py           # 【数据运维】零依赖 HTML/SVG 看板渲染器
├── main.py                    # 统一入口（实际运行用这个）
├── tools/
│   └── make_preview_images.py # 纯标准库生成预览数字图（无需 paddle）
├── requirements.txt
├── README.md
├── 手写识别教学讲解.ipynb      # 教学讲解用（不负责实际训练）
├── logs/                      # 运行后生成：训练日志 + 曲线 + 模型权重 + 记录
├── data/                      # 首次运行自动下载的 MNIST 原始数据(.gz)
├── inputs/                    # 【输入】你自己手写的 PNG 放这里（--mode custom 读）
├── outputs/examples/          # 运行后生成：MNIST 输入图片示例 + 识别结果
├── outputs/custom/            # 运行后生成：你的图片 + 识别结果（与 examples 互不干扰）
└── outputs/ops/               # 运行后生成：运维巡检报告 + 可视化看板（见 §6）
```

---

## 1.1 产物保存位置总览（数据运维）

所有运行产物按 `run_name` 隔离，互不覆盖。以下为**相对项目根目录**的路径；实际运行时 `train.py` / `inference.py` 会一并打印**绝对路径**，便于在服务器/云端定位与归档。

| 类别 | 文件 | 位置（相对项目根） | 说明 |
|---|---|---|---|
| 数据集（原始） | MNIST 的 4 个 .gz | `data/`（缺失时退回 `~/.cache/paddle/dataset/mnist`） | 首次运行自动下载，之后复用 |
| 数据集记录 | `dataset_record.json` | `logs/<run_name>/dataset_record.json` | 来源/样本数/形状，追溯用 |
| 后台日志 | `train_*.log` | `logs/<run_name>/train_YYYYMMDD_HHMMSS.log` | 每轮 loss/acc（控制台+文件） |
| 训练过程数据 | `history.json` | `logs/<run_name>/history.json` | 每轮 loss/acc 数字记录 |
| 本次参数 | `config.json` | `logs/<run_name>/config.json` | 复现实验用 |
| 最佳模型权重 | `best_model.pdparams` | `logs/<run_name>/best_model.pdparams` | val_acc 最高那轮 |
| 最终模型权重 | `model.pdparams` | `logs/<run_name>/model.pdparams` | 最后一轮 |
| 训练曲线图 | `curves.png` | `logs/<run_name>/curves.png` | 损失/准确率双曲线 |
| 输入图片示例 | `sample_*.png` | `outputs/examples/sample_N_trueX_predY.png` | 每张输入图，文件名含真实/预测 |
| 结果拼图 | `predictions.png` | `outputs/examples/predictions.png` | 绿=对 红=错 |
| 结果 JSON | `inference_results.json` | `outputs/examples/inference_results.json` | 结构化识别结果 |
| 【输入】你的图 | `*.png` | `inputs/`（可用 `--image_dir` 改） | 任意尺寸/白底黑字都行，一张一个数字 |
| 你的图的输入图 | `sample_N_predX.png` | `outputs/custom/` | 预处理后模型真正看到的 28×28 |
| 你的图的结果拼图 | `predictions_custom.png` | `outputs/custom/` | 标了答案才判对错，未标只显置信度 |
| 你的图的结果 JSON | `custom_results.json` | `outputs/custom/` | 预测/top3/置信度/原图尺寸/预处理诊断 |
| 【运维】结构化报告 | `ops_report.json` | `outputs/ops/ops_report.json` | 台账/告警全量数据，供程序读 |
| 【运维】可读报告 | `ops_report.md` | `outputs/ops/ops_report.md` | 8 节表格报告，可直接贴进实验报告 |
| 【运维】四联看板 | `dashboard.png` | `outputs/ops/dashboard.png` | 存储/完整性/指标/健康度（图内英文） |
| 【运维】交互看板 | `dashboard.html` | `outputs/ops/dashboard.html` | 中文、自包含、浏览器直接打开 |
| 【运维】数据指纹基线 | `data_fingerprint.json` | `outputs/ops/data_fingerprint.json` | MNIST 四个文件的 md5/样本数，比对漂移用 |
| 【运维】巡检日志 | `ops_*.log` | `outputs/ops/audit/ops_YYYYMMDD_HHMMSS.log` | 每次巡检留痕 |

> 例：`--run_name demo` 时，所有日志与模型都在 `logs/demo/`，输入图片在 `outputs/examples/`。

---

## 2. 在百度 AI Studio（CPU 环境）运行

AI Studio 已预装 `paddlepaddle`，**无需重装**。把整个文件夹上传为项目后，在终端执行：

### 训练
```bash
python main.py --mode train --model_type lenet --lr 0.001 --epochs 5 --run_name demo
```
CPU 上 5 轮 LeNet 约几分钟即可，验证准确率通常 > 98%。

### 推理（生成输入图片示例 + 识别结果）
```bash
python main.py --mode inference --run_name demo --num_show 9
```
结果在 `outputs/examples/`：
- `sample_1_true7_pred7.png` … 每张输入图（文件名含真实/预测标签）
- `predictions.png` … 9 张拼图（绿=预测正确，红=预测错误）
- `inference_results.json` … 结构化结果

### 快速演示（训练 + 推理一条龙）
```bash
python main.py --mode demo --model_type mlp --lr 0.01 --epochs 2
```

### 识别你自己写的数字（任意尺寸 PNG）

把照片/截图直接放进 `inputs/`（**一张图只写一个数字**，尺寸不限，白底黑字或黑底白字都行），
然后：

```bash
python custom_input.py inputs                              # 可选：先自检图片可用（不需要 paddle）
python main.py --mode custom --run_name demo --image_dir inputs
```

结果在 `outputs/custom/`，**不会动 `outputs/examples/` 里的原有产物**：
- `sample_1_true7_pred7.png` / `sample_2_pred3.png` … 预处理后送进模型的那张 28×28
- `predictions_custom.png` … 识别拼图（标了答案的显 OK/BAD，没标的只显置信度）
- `custom_results.json` … 预测、top3、置信度、原图尺寸与预处理诊断

两个要点：
1. **文件名以数字开头**（`7.png`、`3_我的写法.png`）会被当作标准答案，自动算准确率；否则只出预测。
2. 预处理会自动完成：灰度化 → 白底黑字自动反色 → Otsu 自动阈值 → 裁外接框 →
   等比缩放到长边 20px → **按灰度质心居中**到 28×28 → 标准化。
   不做这步而直接拿白底黑字图去喂，准确率会跳崖式下跌——因为 MNIST 是**黑底白字且质心居中**的。

---

## 3. 可调参数一览（课堂演示用）

| 参数 | 含义 | 演示建议 |
|---|---|---|
| `--lr` | 学习率 | `0.01`（快但抖）vs `0.0005`（稳但慢） |
| `--epochs` | 训练轮数 | `2` vs `15`，看是否过拟合 |
| `--model_type` | 网络结构 | `mlp` vs `lenet` vs `cnn` |
| `--optimizer` | 优化器 | `sgd` vs `adam` |
| `--batch_size` | 批大小 | `64` vs `256` |
| `--hidden_size` | MLP 隐藏层大小 | 仅 mlp 生效 |
| `--dropout` | 丢弃比例 | 防过拟合 |
| `--run_name` | 本次运行名 | 每次换名，日志互不覆盖 |
| `--image_dir` | 自定义推理的输入图目录 | 默认 `./inputs`，仅 `--mode custom` 使用 |

示例：对比不同结构
```bash
python main.py --mode train --model_type mlp   --lr 0.01 --epochs 5 --run_name cmp_mlp
python main.py --mode train --model_type lenet --lr 0.001 --epochs 5 --run_name cmp_lenet
```
分别打开 `logs/cmp_mlp/curves.png` 与 `logs/cmp_lenet/curves.png` 即可对比准确率差异。

---

## 4. 关于教学讲解笔记本

`手写识别教学讲解.ipynb` 用于课堂讲解各模块原理，**真正的训练/推理请用终端的 `.py` 命令**。
笔记本里也用 `!python main.py ...` 直接调用 `.py`，方便边讲边跑。

---

## 5. 本地预览输入图片（无需 paddle）

在没有 paddle 的环境想先看输入图长什么样：
```bash
python tools/make_preview_images.py
```
会在 `outputs/examples/preview/` 生成 0-9 的点阵预览图（占位图，非真实 MNIST）。

---

## 6. 数据运维实训（`--mode ops`）

> 本节是**追加的扩展实训**：把 §1.1 那张「产物保存位置表」变成可自动执行的体检。
> 巡检**只读不改**，并且**不依赖 paddle**；缺 matplotlib 时只跳过 PNG，HTML 看板照样出，
> 所以在任何同学本机、教师批改机上都能跑。

### 6.1 一条命令出成果

```bash
python main.py --mode ops                       # 巡检 logs/ 下所有 run
python main.py --mode ops --retention_days 7    # 收紧保留期，观察超期归档告警
python main.py --mode ops --refresh_baseline    # 数据确认无误后重建指纹基线
```

### 6.2 巡检在查什么（五个维度）

| 维度 | 权重 | 查什么 | 典型告警 |
|---|---|---|---|
| 资产完整性 | 25% | run 目录 7 类产物是否齐全 | 缺 `config.json` / 缺 `best_model.pdparams` |
| 一致性校验 | 25% | 日志 ↔ `history.json` ↔ `config.json` ↔ 数据实测 能否互相印证 | 日志 5 轮但 history 只记 3 轮（N7）；台账写 60000 张但数据实际 10000 张（N14） |
| 可追溯性 | 20% | 参数快照、数据留档、模型可达性是否支撑复现 | 结果引用的权重已被删（U9）；`run_name` 与目录名不符（N11） |
| 数据源健康 | 15% | `data/` 的 MNIST 是否完整合法、有无基线漂移 | IDX 头声明与解压字节不符（S3）；md5 变了（S7） |
| 存储治理 | 15% | 体积、0 字节、重复日志、超期 run、孤儿产物 | 同 run 多份日志（G3）；超保留期且占空间大（G4） |

另外单独体检**用户自提图片这条新数据入口**：格式被排除的图（U3）、预处理失败的图（U7）、
报告记录的输入数与目录现有图数对不上（U8，即「图换了但报告没重跑」）。

### 6.3 可视化成果

| 文件 | 内容 | 用途 |
|---|---|---|
| `outputs/ops/dashboard.png` | 四联看板：①存储堆叠条 ②完整性矩阵 ③指标对比+泛化间隙 ④五维健康度雷达 | 贴实验报告（图内全英文，因为 AI Studio 镜像无中文字体） |
| `outputs/ops/dashboard.html` | 中文交互看板：健康度仪表盘 + 台账矩阵 + 告警表 + 趋势线 | 下载后浏览器直接打开，自包含无外链 |
| `outputs/ops/ops_report.md` | 8 节表格报告（含告警清单与整改优先级） | 可直接粘进作业文档 |
| `outputs/ops/ops_report.json` | 结构化全量结果 | 想自己再画图/做统计时用 |

### 6.4 健康度怎么算

每个维度从 100 分起扣：**ERROR -35 / WARN -12 / INFO -3**（下限 0），
再按 §6.2 的权重加权成总分。等级：≥90 A、≥75 B、≥60 C、≥40 D、其余 E。
故意分级，是为了让学生先处理阻断性问题（ERROR）而不是纠结小数点。

### 6.5 课堂练习（约 30 分钟）

1. **基线**：先跑 `--mode train` + `--mode inference`（可再跑 `--mode custom`），
   然后 `--mode ops`，记下健康度、各维得分与 ERROR/WARN 数。
2. **埋故障排查**（只动副本，不碰原始 run）：
   ```bash
   cp -r logs/demo logs/demo_broken
   rm logs/demo_broken/config.json        # 只删副本里的这个文件
   python main.py --mode ops
   ```
   预期：台账矩阵出现红格，命中 C1（缺参数快照）与 T2，完整性维度掉分。
3. **保留期治理**：`--retention_days 0` 再巡检，看 G4 给出的归档建议命令长什么样
   （巡检**不会**替你删，需你自己确认后执行）。
4. **基线漂移**：手改 `outputs/ops/data_fingerprint.json` 里任意一个 md5 的一位数字，
   再巡检 → 应报 S7 数据基线漂移；确认后 `--refresh_baseline` 重建基线。
   （这个练习不动真实数据文件，但能体会「基线本身也是一项资产」。）
5. **读懂一个真实发现**：首次训练后巡检通常会报 S10/N13 —— Paddle 实际把 MNIST 下到了
   `~/.cache/paddle/dataset/mnist`，而 `dataset_record.json` 里写的是 `./data`。
   想想：这算不算事故？运维该信台账还是信磁盘？

### 6.6 交作业时需要包含

- `dashboard.png`（或 `dashboard.html` 的截图）；
- `ops_report.md` 的 §6 告警清单 + §7 整改优先级；
- 一句话结论：**健康度 X 分（等级 Y），主要问题是…，下一步打算…**。
