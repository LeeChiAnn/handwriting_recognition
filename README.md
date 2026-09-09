# 手写数字识别实训项目
**（邯郸职业技术学院大数据综合实训 · 深度学习入门）**

基于百度 AI Studio 经典 MNIST 手写识别项目改造，面向**邯郸职业技术学院大数据综合实训之深度学习实训课程**。
改造后，一条命令链覆盖「训练 → 推理 → 数据运维 → 成果外发」全过程，满足以下 7 项教学要求：

| 要求 | 落地方式 |
|---|---|
| ① 可调参数（学习率 / epochs / 网络结构…） | `config.py` 集中管理，命令行一键覆盖（见 §4） |
| ② 数据与日志维护机制 | `dataset_record.json` 数据集加载记录 + `train_*.log` 训练日志 + `history.json` |
| ③ 输入图片示例 + 识别结果 | `inference.py` 生成 `sample_*.png` 与 `predictions.png` 拼图（见 §3.2） |
| ④ 训练过程可视化 | `visualize.py` 画损失/准确率曲线 `curves.png` |
| ⑤ 多 .py 模块 + 教学 .ipynb | 代码按功能拆分；`手写识别教学讲解.ipynb` 仅讲解，实际跑 .py（见 §8） |
| ⑥ 数据运维实训 | `data_ops.py` 只读巡检全部产物 → 五维健康度 + 告警清单 + `dashboard.png` / `dashboard.html` 看板（见 §5） |
| ⑦ 成果外发（下载到本地就能看） | `eval_model.py` 全测试集验证 + `lab_report.py` 渲染三份自包含 HTML，不在 AI Studio 里看成果（见 §6、§7） |

> 上课/做作业请直接按 [课程任务/README.md](课程任务/README.md) 的命令手册操作（面向专科生，每条命令照抄可跑通全部环节）；本文件讲原理、产物与扩展用法。

---

## 1. 先分清：哪些事在 AI Studio 做、哪些在本机就能做

| 环境 | 能做的事 | 入口 | 前提 |
|---|---|---|---|
| **AI Studio**（云端，预装 paddle） | 训练、识别 MNIST 示例、识别你自己的图、全测试集验证 | `--mode train / inference / custom / eval` | 无 |
| **任意机器**（含你自己的电脑） | 数据运维巡检、生成离线成果包 HTML、图片预处理自检、预览占位图 | `--mode ops / report`、`python custom_input.py inputs`、`python tools/make_preview_images.py` | 只需 `requirements.txt` 里的轻量库（numpy / Pillow / matplotlib），**不需要 paddle** |
| 你的本机想直接"现场识别" | ❌ 做不到 | — | 模型权重 `*.pdparams` 只存在于 AI Studio（训练产物）；本机没有权重文件，且 Apple Silicon 上 paddle CPU 版极慢（见 §9.1） |

**一句话流程（AI Studio 上跑 → 下载到本地看）：**

```bash
# AI Studio 上依次执行（顺序有讲究，见 §6.4）
python main.py --mode train --model_type mlp --lr 0.01 --epochs 10 --run_name demo
python main.py --mode inference --run_name demo
python main.py --mode custom --run_name demo
python main.py --mode eval --run_name demo
python main.py --mode ops
python main.py --mode report

# 下载 outputs/reports/ 整个目录回本地，双击 index.html 即可离线查看全部成果
```

各命令的产物与本地查看方式见 §2.2、§7。

---

## 2. 目录结构与产物总览

### 2.1 目录树

```
handwriting_recognition/
├── config.py                  # 可调参数（dataclass + argparse）
├── logger.py                  # 日志机制（控制台 + 文件）
├── data_utils.py              # 数据加载 + 加载记录
├── model.py                   # 三种网络结构（mlp/lenet/cnn）
├── visualize.py               # 曲线 + 推理结果拼图
├── train.py                   # 训练流程
├── inference.py               # 推理：MNIST 示例 / 用户自提图片两条通路
├── custom_input.py            # 任意尺寸 PNG → MNIST 风格 28×28（零 paddle，可单独自检）
├── data_ops.py                # 【运维】只读巡检 + 五维评分 + 告警（零 paddle）
├── ops_dashboard.py           # 【运维】零依赖 HTML/SVG 看板渲染器
├── eval_model.py              # 【外发】全测试集验证：混淆矩阵/每类 P·R·F1（需 paddle）
├── html_kit.py                # 【外发】零依赖 HTML 基建：内联 CSS + base64 图 + SVG 图表
├── lab_report.py              # 【外发】三份可下载 HTML 的渲染器（零 paddle，见 §6）
├── main.py                    # 统一入口（实际运行用这个）
├── tools/
│   ├── build_notebook.py      # 纯标准库生成 手写识别教学讲解.ipynb
│   └── make_preview_images.py # 纯标准库生成预览数字图（无需 paddle）
├── 课程任务/                  # 专科学生命令行任务手册：照敲命令即可跑通全流程（AI Studio 用）
├── requirements.txt
├── README.md
├── 手写识别教学讲解.ipynb      # 教学讲解用（由 tools/build_notebook.py 生成）
├── data/                      # 【运行后生成】首次运行自动下载的 MNIST 原始数据(.gz)
├── inputs/                    # 【预置】你自己手写的 PNG 放这里（--mode custom 读）
├── logs/                      # 【运行后生成】按 run_name 分目录：日志/曲线/权重/记录
└── outputs/                   # 【运行后生成】全部可视化成果，见 §2.2
    ├── examples/              #   MNIST 输入图片示例 + 识别结果
    ├── custom/                #   你的图片 + 识别结果（与 examples 互不干扰）
    ├── eval/                  #   全测试集验证结果 + 错分样本图
    ├── ops/                   #   数据运维巡检报告 + 看板
    └── reports/               #   离线成果包（下载整个目录即可离线看）
```

> **标了【运行后生成】的目录/文件，只有跑过对应命令才会出现。**
> 例如 `outputs/custom/` 在跑过 `--mode custom` 之前不存在——这是正常的，不是文件丢了。
> 各产物由哪条命令生成、长什么样，见 §2.2 总表与 §3 各小节。

### 2.2 产物保存位置总览

所有运行产物按 `run_name` 隔离，互不覆盖。以下为**相对项目根目录**的路径；实际运行时 `train.py` / `inference.py` 等会一并打印**绝对路径**，便于在服务器/云端定位与归档。

| 类别 | 文件 | 位置（相对项目根） | 说明 |
|---|---|---|---|
| 数据集（原始） | MNIST 的 4 个 .gz | `data/`（缺失时退回 `~/.cache/paddle/dataset/mnist`） | `--mode train` 首次运行自动下载，之后复用 |
| 数据集记录 | `dataset_record.json` | `logs/<run_name>/dataset_record.json` | 来源/样本数/形状，追溯用 |
| 后台日志 | `train_*.log` | `logs/<run_name>/train_YYYYMMDD_HHMMSS.log` | 每轮 loss/acc（控制台+文件） |
| 训练过程数据 | `history.json` | `logs/<run_name>/history.json` | 每轮 loss/acc 数字记录 |
| 本次参数 | `config.json` | `logs/<run_name>/config.json` | 复现实验用 |
| 最佳模型权重 | `best_model.pdparams` | `logs/<run_name>/best_model.pdparams` | val_acc 最高那轮 |
| 最终模型权重 | `model.pdparams` | `logs/<run_name>/model.pdparams` | 最后一轮 |
| 训练曲线图 | `curves.png` | `logs/<run_name>/curves.png` | 损失/准确率双曲线 |
| 输入图片示例 | `sample_*.png` | `outputs/examples/sample_N_trueX_predY.png` | 每张输入图，文件名含真实/预测标签 |
| 结果拼图 | `predictions.png` | `outputs/examples/predictions.png` | 绿=对 红=错 |
| 结果 JSON | `inference_results.json` | `outputs/examples/inference_results.json` | 结构化识别结果 |
| 你的图（原始） | `*.png` | `inputs/`（可用 `--image_dir` 改） | 任意尺寸/白底黑字都行，一张一个数字 |
| 你的图的输入图 | `sample_N[_trueX]_predY.png` | `outputs/custom/` | 预处理后模型真正看到的 28×28 |
| 你的图的结果拼图 | `predictions_custom.png` | `outputs/custom/` | 标了答案才判对错，未标只显置信度 |
| 你的图的结果 JSON | `custom_results.json` | `outputs/custom/` | 预测/top3/置信度/原图尺寸/预处理诊断 |
| 【运维】结构化报告 | `ops_report.json` | `outputs/ops/ops_report.json` | 台账/告警全量数据，供程序读 |
| 【运维】可读报告 | `ops_report.md` | `outputs/ops/ops_report.md` | 9 节表格报告，可直接粘进实验报告 |
| 【运维】四联看板 | `dashboard.png` | `outputs/ops/dashboard.png` | 存储/完整性/指标/健康度（图内英文） |
| 【运维】交互看板 | `dashboard.html` | `outputs/ops/dashboard.html` | 中文、自包含、浏览器直接打开 |
| 【运维】数据指纹基线 | `data_fingerprint.json` | `outputs/ops/data_fingerprint.json` | MNIST 四个文件的 md5/样本数，比对漂移用 |
| 【运维】巡检日志 | `ops_*.log` | `outputs/ops/audit/ops_YYYYMMDD_HHMMSS.log` | 每次巡检留痕 |
| 【验证】验证结果 | `eval_results.json` | `outputs/eval/eval_results.json` | 全测试集准确率/混淆矩阵/每类指标/置信度分布 |
| 【验证】混淆矩阵图 | `confusion.png` | `outputs/eval/confusion.png` | matplotlib 版（图内英文），截图交作业用 |
| 【验证】错分样本 | `e序号_真实_预测.png` | `outputs/eval/errors/` | 模型判错的原始 28×28 图，最多存 `num_show×6` 张 |
| 【成果包】入口页 | `index.html` | `outputs/reports/index.html` | 三张卡片跳转 + 当前体检结论 + 使用说明 |
| 【成果包】模型验证页 | `model_report.html` | `outputs/reports/` | 单文件可下载，外部看模型效果的主战场（见 §6） |
| 【成果包】实验档案袋 | `lab_archive.html` | `outputs/reports/` | 历次 run 对比 + 全部图像成果清单 |
| 【成果包】看板副本 | `dashboard.html` | `outputs/reports/` | 从 `outputs/ops/` 复制，保证只下载 reports/ 也不死链 |

> 例：`--run_name demo` 时，所有日志与模型都在 `logs/demo/`，输入图片在 `outputs/examples/`。

---

## 3. 在 AI Studio 上跑一遍（主线命令）

AI Studio 已预装 `paddlepaddle`，**无需重装**。把整个文件夹上传为项目后，在终端按顺序执行下面任意小节。

### 3.1 训练

```bash
python main.py --mode train --model_type lenet --lr 0.001 --epochs 5 --run_name demo
```

CPU 上 5 轮 LeNet 约几分钟即可，验证准确率通常 > 98%。
产物：`logs/demo/` 下日志、`curves.png`、两个权重文件等（见 §2.2）。

### 3.2 推理：生成 MNIST 输入图片示例 + 识别结果

```bash
python main.py --mode inference --run_name demo --num_show 9
```

从 MNIST **测试集**里抽前 `num_show` 张做示例。结果在 `outputs/examples/`：

- `sample_1_true7_pred7.png` … 每张输入图（文件名含真实/预测标签）
- `predictions.png` … 9 张拼图（绿=预测正确，红=预测错误）
- `inference_results.json` … 结构化结果

### 3.3 识别你自己写的数字（一张 PNG 进，识别图出）

① 把照片/截图直接放进 `inputs/`（**一张图只写一个数字**，尺寸不限，白底黑字或黑底白字都行，文件名以数字开头如 `7.png` 会当作标准答案）：

```bash
python custom_input.py inputs          # 可选：先自检图片能不能用（不需要 paddle、不需要权重）
python main.py --mode custom --run_name demo
```

② 结果在 `outputs/custom/`（**这个目录由本命令运行时创建**，跑之前不存在是正常的；它不会动 `outputs/examples/` 里的原有产物）：

- `sample_1_true3_pred3.png` … 每张图预处理后送进模型的那张 28×28（文件名含标准答案与预测）
- `predictions_custom.png` … 识别拼图（标了答案的显 OK/BAD，没标的只显置信度）
- `custom_results.json` … 预测、top3、置信度、原图尺寸与预处理诊断

③ 两个要点：

1. **文件名以数字开头**（`7.png`、`3_我的写法.png`）会被当作标准答案，自动算准确率；否则只出预测。
2. 预处理会自动完成：灰度化 → 白底黑字自动反色 → Otsu 自动阈值 → 裁外接框 →
   等比缩放到长边 20px → **按灰度质心居中**到 28×28 → 标准化。
   不做这步而直接拿白底黑字图去喂，准确率会跳崖式下跌——因为 MNIST 是**黑底白字且质心居中**的。

④ 识别结果怎么看：把 `outputs/custom/` 里的 PNG / JSON 下载到本地双击即可；想让结果跟着成果包 HTML 一起看，在跑完 §3.5 的 `eval` 后再跑 `--mode report`，`model_report.html` 里会出现"自定义输入"区（缩略图 + 原图 + 预测/置信度）。详见 §7。

### 3.4 快速演示（训练 + 推理一条龙）

```bash
python main.py --mode demo --model_type mlp --lr 0.01 --epochs 2
```

### 3.5 全测试集验证模型效果（`--mode eval`，成果可下载到本地看）

```bash
python main.py --mode eval --run_name demo        # 不给 --run_name 会自动挑唯一带权重的 run
```

跟 `--mode inference`（只抽前几张做示例图）不同，这一步把**整个测试集 10000 张**跑一遍，
产物落 `outputs/eval/`：混淆矩阵、每类 P/R/F1、置信度分布、错分样本原图。
这是"在外部验证模型效果"的**数据来源**，之后必须重跑一次 `--mode report` 才会进 HTML（顺序见 §6.4）。

### 3.6 运维巡检与成果包（不需要 paddle，任意机器都能跑）

```bash
python main.py --mode ops                         # 数据运维巡检，见 §5
python main.py --mode report                      # 生成三份离线成果 HTML，见 §6
```

---

## 4. 可调参数一览（课堂演示用）

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
| `--num_show` | 推理展示样本数 | 同时决定 `--mode eval` 最多存多少张错分图（`×6`） |
| `--retention_days` | run 产物保留天数 | 默认 14；调 0 可看超期归档告警（仅 `--mode ops`） |
| `--ops_dir` | 巡检成果输出目录 | 默认 `./outputs/ops` |
| `--refresh_baseline` | 重建数据集指纹基线 | 确认数据没坏之后重新立基线（仅 `--mode ops`） |
| `--no_embed` | 成果 HTML 不内嵌图片 | 默认内嵌（单文件可下载）；加它则文件小很多但必须整目录拷走 |

示例：对比不同结构

```bash
python main.py --mode train --model_type mlp   --lr 0.01 --epochs 5 --run_name cmp_mlp
python main.py --mode train --model_type lenet --lr 0.001 --epochs 5 --run_name cmp_lenet
```

分别打开 `logs/cmp_mlp/curves.png` 与 `logs/cmp_lenet/curves.png` 即可对比准确率差异。

---

## 5. 数据运维实训（`--mode ops`）

> 把 §2.2 那张「产物保存位置表」变成可自动执行的体检。
> 巡检**只读不改**，并且**不依赖 paddle**；缺 matplotlib 时只跳过 PNG，HTML 看板照样出，
> 所以在任何同学本机、教师批改机上都能跑。

### 5.1 一条命令出成果

```bash
python main.py --mode ops                       # 巡检 logs/ 下所有 run
python main.py --mode ops --retention_days 7    # 收紧保留期，观察超期归档告警
python main.py --mode ops --refresh_baseline    # 数据确认无误后重建指纹基线
```

### 5.2 巡检在查什么（五个维度 + 两类专项）

| 维度 | 权重 | 查什么 | 典型告警 |
|---|---|---|---|
| 资产完整性 | 25% | run 目录 7 类产物是否齐全 | 缺 `config.json` / 缺 `best_model.pdparams` |
| 一致性校验 | 25% | 日志 ↔ `history.json` ↔ `config.json` ↔ 数据实测 能否互相印证 | 日志 5 轮但 history 只记 3 轮（N7）；台账写 60000 张但数据实际 10000 张（N14） |
| 可追溯性 | 20% | 参数快照、数据留档、模型可达性是否支撑复现 | 结果引用的权重已被删（U9）；`run_name` 与目录名不符（N11） |
| 数据源健康 | 15% | `data/` 的 MNIST 是否完整合法、有无基线漂移 | IDX 头声明与解压字节不符（S3）；md5 变了（S7） |
| 存储治理 | 15% | 体积、0 字节、重复日志、超期 run、孤儿产物 | 同 run 多份日志（G3）；超保留期且占空间大（G4） |

另外单独体检**用户自提图片这条新数据入口**：格式被排除的图（U3）、预处理失败的图（U7）、
报告记录的输入数与目录现有图数对不上（U8，即「图换了但报告没重跑」）。

还体检**模型验证与离线成果包**（§6 的产物，缺了/过期了会扣分）：

| 编号 | 查什么 | 级别 |
|---|---|---|
| E1 / P1 | 没做过全测试集验证 / 三份成果 HTML 有缺失 | WARN |
| E2 / E3 | 验证结果引用的 run 已不在 `logs/` / 所用权重文件已丢失（成果不可追溯、不可复现） | ERROR |
| E4 / E6 | `eval_results.json` 解析失败 / 混淆矩阵合计与宣称的准确率、样本数对不上（防手改报告） | ERROR |
| E5 | 测试集实测准确率与训练时 `val_acc` 差 > 0.05 | WARN |
| P2 | 成果 HTML 比它引用的数据旧（训练/验证之后没重新 `--mode report`） | WARN |

### 5.3 可视化成果

| 文件 | 内容 | 用途 |
|---|---|---|
| `outputs/ops/dashboard.png` | 四联看板：①存储堆叠条 ②完整性矩阵 ③指标对比+泛化间隙 ④五维健康度雷达 | 贴实验报告（图内全英文，因为 AI Studio 镜像无中文字体） |
| `outputs/ops/dashboard.html` | 中文交互看板：健康度仪表盘 + 台账矩阵 + 告警表 + 趋势线 | 下载后浏览器直接打开，自包含无外链 |
| `outputs/ops/ops_report.md` | 9 节表格报告（含模型验证与成果包体检、告警清单、整改优先级） | 可直接粘进作业文档 |
| `outputs/ops/ops_report.json` | 结构化全量结果 | 想自己再画图/做统计时用 |

### 5.4 健康度怎么算

每个维度从 100 分起扣：**ERROR -35 / WARN -12 / INFO -3**（下限 0），
再按 §5.2 的权重加权成总分。等级：≥90 A、≥75 B、≥60 C、≥40 D、其余 E。
故意分级，是为了让学生先处理阻断性问题（ERROR）而不是纠结小数点。

### 5.5 课堂练习（约 30 分钟）

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

### 5.6 交作业时需要包含

- `dashboard.png`（或 `dashboard.html` 的截图）；
- `ops_report.md` 的**告警清单**与**整改优先级**两节（报告内的 §8、§9）；
- 一句话结论：**健康度 X 分（等级 Y），主要问题是…，下一步打算…**。

### 5.7 巡检在"缺跑/中断"环境下长什么样

本机仓库现存的 `outputs/ops/*` 是基于一次**被中断的** run 生成的
（`logs/demo/` 只有 `dataset_record.json` 和半截 `train_*.log`）：
健康度 70.2（C）、3 条 ERROR（C1 缺权重与参数快照）+ S10/N13 数据目录降级告警。
这正是巡检该有的输出——**产物不齐就该扣分**；在 AI Studio 上跑完一次完整训练后重新巡检，
完整性矩阵会转绿、指标面板与雷达图会随之填满。

---

## 6. 成果外发：下载到本地就能看的三份 HTML（`--mode eval` + `--mode report`）

> 目标很单纯：**不在 AI Studio 里看成果**。把远端跑出来的结果导成几个 `.html`，
> 学生下载后双击就能看，教师不装任何环境也能批。页面全部自包含：
> CSS 内联、图片 base64 内嵌、图表是内嵌 SVG，**无外链、无 JS、断网可用**。

### 6.1 仍坚持“图内英文 / 页内中文”

AI Studio 镜像没中文字体，PNG 里写中文会变豆腐块；而 HTML 由浏览器渲染，字体由本机提供。
所以 `confusion.png` / `dashboard.png` 里是全英文，而 HTML 里是全中文。
`lab_report.py` 甚至不依赖 matplotlib：**曲线、混淆矩阵、仪表盘都是 SVG 现画的**，
PNG 只是并排放一份供截图交作业。

### 6.2 三份成果各回答一个问题

| 文件 | 回答什么 | 上游数据 | 典型体积 |
|---|---|---|---|
| `outputs/reports/model_report.html` | **这个模型到底行不行？**总分、每类 P/R/F1、混淆矩阵、置信度分布、错分原图、与训练记录对账、你手写图的识别结果 | `outputs/eval/eval_results.json` + `outputs/custom/` | 4 KB（没验证数据时）→ ~50 KB（含 30 张错分图） |
| `outputs/reports/lab_archive.html` | **我做过哪些实验、所有图在哪？**run 参数对比、loss/acc 曲线（SVG 现画）、日志原文、数据留档、**全部图像成果清单** | `logs/*` + `outputs/*` | ~200 KB（内嵌 15 张图） |
| `outputs/reports/index.html` | **从哪儿进去？**三张卡片 + 当前体检结论 + 使用步骤 | 上面两份 + `outputs/ops/` | ~6 KB |

另有一份 `outputs/reports/dashboard.html`（从 `outputs/ops/` 复制），
目的是让 `reports/` 成为一个**整目录下载就完整**的交付包，不会因为没下 `ops/` 而留个死链；
正本仍在 `outputs/ops/dashboard.html`，两边内容一致。

### 6.3 不导出模型，那“在外部验证”验的是什么？

这是有意的设计抉择：**只导出“证据”，不导出权重**（避开 paddle2onnx / 本地前向复算的版本坑）。
`model_report.html` 里能拿到的证据是：

1. **全测试集实测**（不是 9 张示例）：准确率、macro-F1、10×10 混淆矩阵、每类 P/R/F1；
2. **错分原图**：模型判错的那几张 28×28 直接给你看，自己判断是人眼也分不清还是训练不够；
3. **置信度分布**：中位数/分位数、低于 0.90 的样本数——“没把握”比“错了”更早发现问题；
4. **分布外样本**：你自己手写的图（`--mode custom`）与模型实际看到的 28×28 并排对照；
5. **两份实现互相印证**：matplotlib 画的 `confusion.png` 与 HTML 里用 JSON 现画的 SVG 矩阵共存，
   对不上就是渲染或数据有问题。

同时 `--mode ops` 会机检这些证据能不能信（见 §5.2 的 E/P 系列告警）：
引用的 run 与权重是否还在、混淆矩阵合计是否等于宣称的样本数与准确率（防手改报告）、
实测准确率与训练 `val_acc` 是否对得上、成果包是否比上游数据旧。

> 局限说清楚：这属于**结果级验证**（看统计量与样本），不是**复算级验证**（在本地重新跑前向）。
> 需要复算级验证请自行导出权重并配合 paddle-lite / ONNX，本项目按课程设计不做这一步。

### 6.4 固定执行顺序（远端 → 本地）

```bash
# ① 在 AI Studio 上（有 paddle）
python main.py --mode train     --model_type mlp --lr 0.01 --epochs 10 --run_name mlp_e10
python main.py --mode inference --run_name mlp_e10
python main.py --mode custom    --run_name mlp_e10 --image_dir inputs
python main.py --mode eval      --run_name mlp_e10      # 全测试集，必须远端跑
python main.py --mode ops                                # 巡检，产出成果包要引用的 ops_report.json
python main.py --mode report                             # 最后一步才出 HTML，顺序不能颠倒

# ② 下载 outputs/reports/ 整个目录回本地，双击 index.html 即可
# ③ 本地改不了上游数据，但如果只改了渲染层（html_kit.py），本地重跑一次即可：
python main.py --mode report
```

为什么顺序重要：`report` 会读 `eval_results.json` 与 `ops_report.json`；而巡检的 P2 告警
专门盯“成果包比上游旧”，所以永远是 **ops → report** 收尾。

---

## 7. 下载到本地后，怎么查看这些成果

| 你想看什么 | 下载什么 | 怎么看 |
|---|---|---|
| 单张识别图 / 结果拼图 | `outputs/examples/`、`outputs/custom/`、`outputs/eval/`、`outputs/eval/errors/` 里的 PNG | 双击图片，或插进 Word/PPT |
| 模型到底行不行（验证报告） | `outputs/reports/model_report.html` | 双击，浏览器打开 |
| 历次实验 + 全部图汇总 | `outputs/reports/lab_archive.html` | 双击 |
| 数据运维看板 | `outputs/ops/dashboard.html`（或 `reports/dashboard.html` 副本） | 双击 |
| 从入口总览进 | **整个 `outputs/reports/` 目录** | 双击 `index.html`，三张卡片跳转；整目录下载保证无死链 |
| 交实验报告的素材 | `confusion.png`、`dashboard.png`、`ops_report.md` | 图片直接插入，md 可粘为表格 |

共性：所有 HTML **单文件自包含**（图片以 base64 内嵌），不联网也能打开；
只要**连同图片整目录**下载 `reports/`，发给任何人（包括没装任何软件的老师）都能看。

---

## 8. 教学讲解笔记本与本地小工具

- `手写识别教学讲解.ipynb` 用于课堂讲解各模块原理，**真正的训练/推理请用终端的 `.py` 命令**。
  笔记本里也用 `!python main.py ...` 直接调用 `.py`，方便边讲边跑。
  本仓库自带一份成品；改了代码后可重新生成：
  ```bash
  python tools/build_notebook.py          # 纯标准库，无需 nbformat
  ```
- 在没有 paddle 的环境想先看"数字长什么样"：
  ```bash
  python tools/make_preview_images.py     # 在 outputs/examples/preview/ 生成 0-9 点阵预览图（占位图，非真实 MNIST）
  ```
- 想先确认自己放进 `inputs/` 的图能不能被模型吃（不需要 paddle、不需要权重）：
  ```bash
  python custom_input.py inputs           # 逐张打印预处理诊断，不写任何文件
  ```

---

## 9. 常见问题与现象

### 9.1 为什么我的 Mac 上训练/推理特别慢？

Apple Silicon（M 系列）+ `paddlepaddle` 2.6.x 的 CPU wheel 实测 CPU 矩阵乘极慢
（本机 `512³ × 20` 次 matmul 超过 45 s，而 numpy 只要 0.1 s），一个 epoch 都跑不完。
**这不是本项目代码的问题**——训练与推理请在 AI Studio（Linux/x86，预装 paddle）上完成；
本机只用来跑巡检（`--mode ops`）、出成果包（`--mode report`）、看图和改报告。

### 9.2 现象速查表

| 现象 | 原因 / 处置 |
|---|---|
| 没有 `outputs/custom/`（或 `examples/`、`eval/`）目录 | 正常：这些目录由对应命令**运行时**创建（见 §2.1 提示）。`custom` 需要权重，本机没有就跑不了那一步，在 AI Studio 跑完 `--mode custom` 才会出现 |
| 打开页面只看到“尚未生成验证数据” | 还没跑 `--mode eval`，或跑了但没重跑 `--mode report`（顺序见 §6.4） |
| 曲线区显示“缺少 history.json” | 该 run 训练被中断，过程数据没落盘；重新训练一轮即可 |
| 图片位显示“未生成”灰框 | 对应 PNG 没产出（缺 matplotlib 或 run 不完整），这是故意留字而非留白 |
| 单文件太大 | 加 `--no_embed`（但必须整目录拷），或删掉 `outputs/eval/errors/` 里多余的错分图后重跑 report |
| 手改了 `dashboard.html` / 三份 HTML，重跑又变回原样 | 这些是**生成物**，运行时会整份覆盖；想调样式/内容请改渲染器（`ops_dashboard.py` / `html_kit.py` / `lab_report.py`）再重跑 `--mode ops` / `--mode report` |
| 在 AI Studio 的 notebook 里排版奇怪 | 本项目的 HTML 不依赖 JS，但 notebook 的内嵌预览器会注入它自己的样式；请在终端跑完后用本地浏览器打开 |
