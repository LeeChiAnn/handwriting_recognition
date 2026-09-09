# 手写数字识别实训 · 命令行任务手册

> 给**专科班同学**的操作手册。目标只有一个：**照着下面的命令，在百度 AI Studio 上一步一步复制、粘贴、回车，就能跑完整套实训并拿到全部结果**。
> 你不需要读懂代码，命令可以当成"操作咒语"。每节格式固定：
> **① 这个任务在做什么 → ② 要跑的命令（可直接复制）→ ③ 预期输出 → ④ 检查点（看到什么 = 成功）**。

---

## 0. 从零到交作业：总流程表

| 顺序 | 任务 | 你要做的 | 命令在哪一节 | 完成后你手里多了什么 |
|---|---|---|---|---|
| 1 | 环境准备 | 上传项目文件夹到 AI Studio，确认 paddle 可用 | §1 | 能跑 `python main.py` 的云端环境 |
| 2 | 训练模型 | 训一个 lenet 网络，得到模型权重 | §2 | `logs/demo/`：日志、`curves.png` 曲线、`best_model.pdparams` 权重 |
| 3 | 调参实验 | 用不同学习率 / 轮数 / 网络结构各训几次，对比准确率 | §3 | 多个 run 的 `最佳验证准确率` 数字，可做对比表 |
| 4 | 推理出图 | 让模型认 MNIST 示例图 + 认你自己手写的数字 | §4 | `outputs/examples/`、`outputs/custom/` 的识别结果 PNG |
| 5 | 全测试集验证 | 10000 张测试图全跑一遍，出混淆矩阵等 | §5 | `outputs/eval/`：混淆矩阵图、错分样本图、指标 JSON |
| 6 | 智能运维看板 | 数据巡检自动打分，出健康度报告与看板 | §6 | `outputs/ops/`：`dashboard.png` / `dashboard.html` / `ops_report.md` |
| 7 | 成果外发 | 把全部结果打包成下载到本地就能看的 HTML | §7 | `outputs/reports/` 整个文件夹（交作业最核心的成果） |

> 前 5 步在 AI Studio 上做（需要模型）；第 6、7 步**在任何电脑上都能做**（不需要深度学习环境），但你手上要已有第 2~5 步的产物。

---

## 1. 任务一：环境准备（把项目搬进 AI Studio）

### ① 这个任务在做什么

在百度 AI Studio（云端服务器，已经装好深度学习环境）上开一个"项目"，把你电脑上的项目文件夹传上去，并确认能运行 Python。

### ② 要做的步骤

**第 1 步：本地压缩项目文件夹**

找到 `handwriting_recognition` 文件夹（就是有 `main.py`、`README.md` 的那个），把它整体压缩成一个 zip 文件：
- Windows：右键 → 发送到 → 压缩(zipped)文件夹
- Mac：右键 → 压缩"handwriting_recognition"

**第 2 步：在 AI Studio 新建项目**

1. 打开 https://aistudio.baidu.com 并登录（用百度账号即可）；
2. 点右上角「创建项目」，类型选**空项目**，名称随便填（如 `mnist-实训`），创建后进入项目页面；
3. 把刚才的 `handwriting_recognition.zip` 从本地拖进项目文件列表（或点上传按钮）。**注意：AI Studio 自带 PaddlePaddle 环境，不需要安装任何东西。**

**第 3 步：打开终端，解压并进入项目目录**

在项目页面找到「终端」（有的版本叫"命令行"），打开后依次执行：

```bash
unzip handwriting_recognition.zip
cd handwriting_recognition
pwd
ls
```

> `pwd` = 显示当前在哪个目录；`ls` = 列出当前目录的文件。
> **看到 `main.py`、`README.md`、`config.py` 这些文件就说明进对目录了。**
> 如果 `ls` 看到的是 `handwriting_recognition` 文件夹（zip 多套了一层），就再执行一次 `cd handwriting_recognition`。

**第 4 步：确认环境可用**

```bash
python --version
python -c "import paddle; print('paddle 版本:', paddle.__version__)"
```

### ③ 预期输出

- `python --version` 显示 3.x 之类的 Python 版本号；
- 第二条命令打印出 `paddle 版本: 2.x.x`（具体小版本号不同没关系）。

### ④ 检查点

- [ ] `ls` 能看到 `main.py`
- [ ] 能打印出 paddle 版本号 → 环境就绪，去任务二
- [ ] 如果提示 `No module named 'paddle'` → 看 §9 排障第 2 条

---

## 2. 任务二：训练模型（跑出第一个模型）

### ① 这个任务在做什么

用 MNIST 手写数字数据集（6 万张训练图）训练一个 lenet 网络，让它学会认 0~9。
训练完成后会得到一个"模型权重文件"——**后面所有识别、验证都靠它**。

### ② 要跑的命令

在项目根目录（能看到 `main.py` 的目录）执行：

```bash
python main.py --mode train --model_type lenet --lr 0.001 --epochs 5 --run_name demo
```

参数含义（先不用记，§3 会用到）：
- `--model_type lenet`：网络结构（可选 mlp / lenet / cnn）
- `--lr 0.001`：学习率（可选 0.01 / 0.0005…）
- `--epochs 5`：训练轮数
- `--run_name demo`：给这次训练起个名字，产物都放进 `logs/demo/`

**第一次跑会自动下载 MNIST 数据**（看到 `data/` 目录慢慢出现 `.gz` 文件），要等 1~3 分钟，属正常现象。

### ③ 预期输出

终端会一屏一屏滚动，关键看这两类行：

```
Epoch 01/05 | train_loss=0.1834 train_acc=0.9487 | val_loss=0.0745 val_acc=0.9812 | 12.3s
Epoch 02/05 | train_loss=0.0766 train_acc=0.9781 | val_loss=0.0528 val_acc=0.9846 | 12.1s
...
训练完成。最佳验证准确率=0.9886。模型/日志/曲线已存至 /home/aistudio/handwriting_recognition/logs/demo
==== 本次产物保存位置（绝对路径）====
```

- `val_acc` = 验证集准确率，**越高越好**，lenet 训 5 轮通常能到 0.98 以上（98%+）；
- 最后打印的绝对路径就是产物位置。

**产物清单**（在 `logs/demo/` 目录下，可用 `ls logs/demo` 查看）：

| 文件 | 是什么 |
|---|---|
| `train_日期_时间.log` | 训练过程的完整文字记录（和终端显示的一样） |
| `config.json` | 本次训练用了什么参数 |
| `history.json` | 每轮的 loss/acc 数字 |
| `curves.png` | 损失/准确率曲线图（**交作业素材**） |
| `best_model.pdparams` | 验证准确率最高的那轮模型权重（**最值钱的文件**） |
| `model.pdparams` | 最后一轮的模型权重 |
| `dataset_record.json` | 数据集加载记录 |

### ④ 检查点

- [ ] 终端最后出现 **"训练完成。最佳验证准确率=…"**
- [ ] `ls logs/demo` 能看到 **`best_model.pdparams`**
- [ ] 有上面两个 → 训练成功；看不到 → 看 §9 排障第 3 条

---

## 3. 任务三：调参小实验（用命令对比参数）

### ① 这个任务在做什么

同一套代码，只改**一个参数**、多训几次，肉眼对比准确率变化。做 3 组小实验，每组都用**不同的 `--run_name`**（防止互相覆盖）。

### ② 要跑的命令（三条一组，依次复制执行）

**实验 A：学习率 大 vs 小**（学习率 = 每次学多少，太大容易"学飞了"，太小学得慢）

```bash
python main.py --mode train --model_type lenet --lr 0.01 --epochs 5 --run_name expA_lr_big
python main.py --mode train --model_type lenet --lr 0.0005 --epochs 5 --run_name expA_lr_small
```

**实验 B：训练轮数 少 vs 多**（轮数 = 把数据集从头学几遍）

```bash
python main.py --mode train --model_type lenet --lr 0.001 --epochs 3 --run_name expB_ep3
python main.py --mode train --model_type lenet --lr 0.001 --epochs 10 --run_name expB_ep10
```

**实验 C（选做加分）：网络结构对比**（mlp 最简 / lenet 卷积 / cnn 更深的卷积）

```bash
python main.py --mode train --model_type mlp --lr 0.001 --epochs 5 --run_name expC_mlp
python main.py --mode train --model_type lenet --lr 0.001 --epochs 5 --run_name expC_lenet
python main.py --mode train --model_type cnn --lr 0.001 --epochs 5 --run_name expC_cnn
```

### ③ 怎么对比结果（不用写代码）

每个 run 结束时终端都有 **`最佳验证准确率=0.98xx`**，把数字抄进下面的表：

| 实验 | run 名 | 参数 | 最佳验证准确率 |
|---|---|---|---|
| A | expA_lr_big | lr=0.01 | ? |
| A | expA_lr_small | lr=0.0005 | ? |
| B | expB_ep3 | epochs=3 | ? |
| B | expB_ep10 | epochs=10 | ? |
| C | expC_mlp | mlp | ? |
| C | expC_lenet | lenet | ? |
| C | expC_cnn | cnn | ? |

想看图对比的，把两个 run 的 `curves.png` 都下载下来并排看（路径：`logs/expA_lr_big/curves.png` 等）。

**常见规律**（如果你的数字方向相反，先检查命令里的参数有没有写串，再想想为什么）：
- lr=0.01：前面学得快，但曲线可能抖；lr=0.0005：稳但 5 轮不够，准确率明显偏低 → **学习率不是越大越好**；
- 10 轮通常比 3 轮高，但涨得越来越慢（数据多学几遍收益递减）；
- 结构越复杂通常准确率越高：cnn ≥ lenet ≥ mlp，但训练也更慢。

### ④ 检查点

- [ ] 7 个 run 全部出现"训练完成"
- [ ] `ls logs` 能看到 `expA_lr_big`、`expA_lr_small`、`expB_ep3`、`expB_ep10`、`expC_*` 这些目录（每个 run 一个文件夹，互不覆盖）
- [ ] 表里每个数字都填上了 → 去任务四

---

## 4. 任务四：测试与推理出图（让模型认图）

### ① 这个任务在做什么

分两步：先让模型认 **MNIST 自带的测试图**（示例），再认**你自己手写的数字图**。每张图都会输出"识别结果图"，下载到本地就能看。

> ⚠️ 下面的 `--run_name demo` 必须和任务二训练时**一模一样**，否则会报"未找到模型权重"。

### 4.1 识别 MNIST 示例图

```bash
python main.py --mode inference --run_name demo --num_show 9
```

**预期输出**（结尾类似）：

```
[推理完成] 示例数=9 示例准确率=1.0
==== 本次产物保存位置（绝对路径）====
  输入图片目录 : .../outputs/examples
  输入图片 01  : .../outputs/examples/sample_1_true7_pred7.png
  ...
  结果拼图     : .../outputs/examples/predictions.png
```

**产物**（在 `outputs/examples/`）：

- `sample_1_true7_pred7.png` 等 9 张单图：**文件名就是结果**——`true7` 表示真实是 7，`pred7` 表示模型认成 7，两者一样就是认对了；
- `predictions.png`：9 张排在一起的拼图，**绿色=认对，红色=认错**；
- `inference_results.json`：识别结果表格（程序读的，人看图片就行）。

### 4.2 识别你自己手写的数字

**第 1 步：准备图片并上传到 `inputs/` 文件夹**

- 自己用手机拍 / 画图软件写 **3~5 张数字图**，每张图**只写一个数字**，写大一点、清楚一点；
- 把图片传到 AI Studio 项目里，覆盖/放进取名为 `inputs` 的文件夹（找不到就在项目根目录新建一个）；
- **文件名尽量以数字开头**：如 `7.png`、`3_我写的.png`。这样程序会把开头数字当"标准答案"，自动告诉你认对没认对；不以数字开头也能识别，只是不算对错。

**第 2 步（推荐）：先自检图片能不能用**（不需要模型，几秒钟出结果）

```bash
python custom_input.py inputs
```

**预期输出**：每张图一行，全部 `[OK]` 表示能识别；出现 `[跳过]` 或 `[X]` 的行会写明原因（比如不是 PNG、打不开）。

**第 3 步：识别**

```bash
python main.py --mode custom --run_name demo
```

**预期输出**（结尾类似）：

```
[1/3] ✓ 3.png 原尺寸[28, 28] 反色=否 → pred=3 p=0.997
[自定义推理完成] 成功 3/3 张，标注样本准确率 1.0
==== 本次产物保存位置（绝对路径）====
```

- `✓`=认对，`✗`=认错，`?`=文件名没写答案不算对错；`p=0.997` 是置信度（越接近 1 越有把握）；
- 反色=是/否 说明程序已自动把你的图处理成模型习惯的黑底白字。

**产物**（在 `outputs/custom/`，和 4.1 的 `outputs/examples/` 互不影响）：

- `sample_1_true3_pred3.png` 等：每张图**预处理后**的样子（模型真正看到的就是这张 28×28 小图）；
- `predictions_custom.png`：你的图拼图，带答案的显示 OK/BAD，没带答案的只显示预测数字和置信度；
- `custom_results.json`：结构化结果。

> `outputs/custom/` 这个文件夹**只有跑完上面命令才会出现**，跑之前不存在是正常的。

### 4.3 如何下载到本地查看

AI Studio 网页的项目文件列表里，一层层点进 `outputs/examples`、`outputs/custom`，勾选图片后点「下载」；也可以直接下载**整个项目**后在本地文件夹里找。
下载后双击 PNG 即可查看（Windows 用"照片"、Mac 用"预览"）。

### ④ 检查点

- [ ] `outputs/examples/predictions.png` 里 9 格基本全绿
- [ ] `custom_input.py inputs` 全部 `[OK]`
- [ ] `outputs/custom/predictions_custom.png` 里能看到你手写数字的识别结果（带 ✓/✗ 或 OK/BAD）
- [ ] 图片已下载到本地电脑且能打开 → 去任务五

---

## 5. 任务五：全测试集验证（模型到底行不行）

### ① 这个任务在做什么

任务四只看了 9 张示例图，不算数。这一步把**整个测试集 10000 张**全部送进模型跑一遍，得出权威结论：总准确率、每个数字（类别）分别的精确率/召回率/F1、模型最容易把谁认成谁（混淆矩阵），还把**每一张认错的图**都存下来当"错题集"。

### ② 要跑的命令

```bash
python main.py --mode eval --run_name demo
```

> 不带 `--run_name` 也行：如果 `logs/` 下只有一个带权重的 run 它会自动用那个；有多个时会提示你加 `--run_name` 指定（排障第 8 条）。

### ③ 预期输出

终端每验证 2000 张会打一次进度，最后类似：

```
验证完成：准确率 0.9868｜macro F1 0.9866｜错分 132/10000｜置信度<0.90 共 189 张
==== 本次产物保存位置（绝对路径）====
  结果JSON     : .../outputs/eval/eval_results.json
  错分样本图集 : .../outputs/eval/errors（最多存 num_show×6 = 54 张代表）
  混淆矩阵图   : .../outputs/eval/confusion.png
```

**产物**（在 `outputs/eval/`）：

| 文件 | 是什么 | 怎么看 |
|---|---|---|
| `confusion.png` | 10×10 混淆矩阵热力图 | 每格 = 真实 X 被认成 Y 的张数；**对角线越亮越好**，越亮 = 认对的越多。常见的"亮斑"在 4↔9、7↔1、8↔9 之间——这些数字人眼也容易看混，属正常 |
| `errors/e2_true5_pred3.png` 等 | 错分样本原图（"错题集"） | 全部错分图中程序最多存 `num_show×6`（默认 54）张做代表；文件名 `e序号_true真实_pred预测`，下载下来自己看看，能认出是几，说明模型还有提升空间 |
| `eval_results.json` | 全量数字结果 | 含 0~9 每个类别的精确率/召回率/F1（文本编辑器打开可查） |

### ④ 检查点

- [ ] 终端出现 **"验证完成：准确率 0.98…"**（一般 0.97~0.99）
- [ ] `outputs/eval/confusion.png` 存在且对角线明显亮
- [ ] `outputs/eval/errors/` 里能看到几张错分图 → 去任务六

---

## 6. 任务六：生成智能运维看板（数据体检报告）

### ① 这个任务在做什么

让程序像"体检医生"一样，**只读不改**地检查你这些天产出的所有文件（日志、权重、图片、数据、报告）是否齐全、能否对得上，然后打一个**健康度分数**、列一份**告警清单**，并生成看板图片和看板网页。
这一节**不需要 paddle**——你把 `outputs/`、`logs/` 拷到任何电脑上都能跑。

### ② 要跑的命令

```bash
python main.py --mode ops
```

可选参数（看告警用）：
```bash
python main.py --mode ops --retention_days 0     # 临时把保留期设为 0，看"超期归档"告警长什么样
python main.py --mode ops --refresh_baseline     # 确认数据没坏后，重建数据指纹基线（一般不用）
```

### ③ 预期输出与产物

终端结尾类似：

```
巡检完成：健康度 87.5（B）｜ERROR=0 WARN=4 INFO=2｜run 数=7
```

> 数字会因你跑过多少实验而不同。**完整跑完任务二~五后，健康度一般在 75（B）以上；如果看到 ERROR 类告警（尤其 C1 缺权重），多半是某个 run 没训完或参数写串**（对照 §9）。

产物（在 `outputs/ops/`）：

| 文件 | 是什么 | 用途 |
|---|---|---|
| `dashboard.png` | 四联图：存储占用 / 文件完整性 / 指标对比 / 健康度雷达 | **贴实验报告用**（图内是英文，AI Studio 没中文字体，正常） |
| `dashboard.html` | 同一个看板的中文网页版 | 下载到本地双击浏览器打开，能看到健康度仪表盘、告警明细 |
| `ops_report.md` | 9 节文字报告：各项体检结果 + 告警清单 + **整改建议** | 可以整个复制进 Word |
| `ops_report.json` | 结构化原始数据 | 一般不用管 |
| `audit/ops_*.log` | 每次巡检的留痕日志 | 一般不用管 |

**怎么读告警**：`ops_report.md` 里有"告警清单"（每条有编号，如 C1、N7、G4）和"整改优先级"（每条给出**建议执行什么命令**），照着做再巡检一次，分数就会上去。

### ④ 检查点

- [ ] 终端出现 **"巡检完成：健康度 XX（等级）"**
- [ ] `outputs/ops/` 里能看到 `dashboard.png`、`dashboard.html`、`ops_report.md`
- [ ] 下载 `dashboard.html` 本地双击能打开，且能看到健康度数字 → 去任务七

---

## 7. 任务七：成果外发（生成离线 HTML 成果包）

### ① 这个任务在做什么

把前 6 个任务的成果浓缩成 **4 个 HTML 文件**，全部自包含（图片直接嵌在文件里、不联网也能开）。
把这几个文件下载到本地，发给谁都能看——**老师不装任何软件也能批你的作业**。
这一节同样**不需要 paddle**。

> ⚠️ **顺序固定：先 §6 的 `ops`，再 `report`**。程序会检查"成果包是不是比最新数据旧"，反着跑会收到 P2 过期告警。

### ② 要跑的命令

```bash
python main.py --mode report
```

（如果你刚跑完 §6 的 ops，直接执行这一条即可；没跑过 ops 就先补跑 `python main.py --mode ops`。）

### ③ 预期输出与产物

终端结尾类似：

```
==== 离线成果包生成完毕（图片内嵌=是）====
  index.html        -> .../outputs/reports/index.html（…KB）
  model_report.html -> .../outputs/reports/model_report.html
  lab_archive.html  -> .../outputs/reports/lab_archive.html
  运维看板          -> .../outputs/reports/dashboard.html
```

产物在 `outputs/reports/`（**整目录下载**）：

| 文件 | 打开后能看什么 |
|---|---|
| `index.html` | **入口页**：三张卡片 + 当前体检结论 + 使用说明，从它点进去 |
| `model_report.html` | 模型效果报告：总准确率、每类 P/R/F1、混淆矩阵、置信度分布、错分原图、**你手写图的识别结果** |
| `lab_archive.html` | 实验档案袋：你跑过的所有 run 参数对比、loss/acc 曲线、**全部图片成果汇总** |
| `dashboard.html` | 运维看板副本（和 §6 那个一样，复制过来保证整包不缺东西） |

### ④ 检查点（下载与打开）

1. 在 AI Studio 文件列表里找到 `outputs/reports` 文件夹 → 勾选 → 下载（会得到一个 zip）；
2. 本地解压后**双击 `index.html`**（用 Chrome / Edge 打开，不要用记事本）；
3. 检查：模型报告页有准确率和混淆矩阵（说明 §5 的 eval 结果进去了）；档案袋里有曲线；"自定义输入"区能看到你手写的图（说明 §4.2 的 custom 结果进去了）；首页能看到健康度（说明 §6 的 ops 结果进去了）；
4. **改动任何上游数据后（比如换了手写图重新 custom、又训了新 run），必须重跑一次 `python main.py --mode report` 才会更新**。

- [ ] 4 个 HTML 都在本地能双击打开、内容完整
- [ ] 页面里没有"尚未生成验证数据"的灰字（有 → 检查是否漏跑 §5 的 eval，或跑完没重跑 report）
- [ ] 全部通过 → 实训完成 🎉

---

## 8. 交作业素材收集（对照这个清单拿文件）

| 作业素材 | 位置 | 什么时候产出 |
|---|---|---|
| 训练曲线图 | `logs/<run名>/curves.png`（挑 2~3 个对比） | 任务二/三 |
| 调参对比表 | 你自己抄的表（§3） | 任务三 |
| MNIST 示例识别拼图 | `outputs/examples/predictions.png` | 任务四 |
| 自己手写数字识别拼图 | `outputs/custom/predictions_custom.png` | 任务四 |
| 混淆矩阵 + 错分样本 | `outputs/eval/confusion.png`、`outputs/eval/errors/` 挑几张 | 任务五 |
| 运维看板 | `outputs/ops/dashboard.png` + `ops_report.md` | 任务六 |
| 离线成果包 | **整个 `outputs/reports/` 文件夹** | 任务七 |

> 文件命名建议：截图/图片文件名改成"学号_内容.png"再交，老师好认。

---

## 9. 快速排障对照表

| # | 现象 | 原因 / 怎么办 |
|---|---|---|
| 1 | `command not found: python` | 终端没进对项目或平台环境缺 python：试 `python3`；确认在 AI Studio 的项目终端里，不是本地电脑终端 |
| 2 | `No module named 'paddle'` | 环境不是 Paddle 预装环境：项目设置里换成含 PaddlePaddle 的环境，或新建项目时选"深度学习"模板 |
| 3 | `未找到模型权重，请先训练` | `--run_name` 和训练时不一致（比如训练写 demo、推理写 demo2）。先 `ls logs` 看有哪些 run，把命令里的 run_name 改成存在的那个，或重训一次 demo |
| 4 | `logs/run_20260909_123456` 这种目录哪来的 | 你**忘了写 `--run_name`**，程序自动用时间戳当名字。产物都在那个目录里；建议重跑时显式写 `--run_name demo` 方便找 |
| 5 | 跑完找不到 `outputs/custom/` 或 `outputs/eval/` | 这些目录由对应命令**运行时才创建**：先确认命令有没有报错、有没有跑到"产物保存位置"那段；`ls outputs` 看实际有哪些目录 |
| 6 | 训练"卡住"不动 | 首次运行要下载 MNIST 数据（`data/` 里出现 .gz，等 1~3 分钟）；之后看有没有 `Epoch 01/05` 这种行在滚动——有就是正常在跑，CPU 训 5 轮 lenet 约几分钟 |
| 7 | `eval` 提示"有多个带权重的 run" | 按提示在命令末尾加 `--run_name 其中一个`，例如 `python main.py --mode eval --run_name demo` |
| 8 | 健康度很低 / 一堆 ERROR | 打开 `ops_report.md` 的告警清单，看编号（C1 缺权重=训练没成功；S10/N13 是数据目录说明，属已知提示）。按"整改优先级"给的命令处理，再跑一次 `--mode ops` |
| 9 | 下载的 HTML 双击打不开 / 显示一堆代码 | 用 **Chrome / Edge 浏览器**打开，别用记事本；也别在微信里直接点开 |
| 10 | HTML 里图片是裂的 | 说明没下全：`reports/` 要**整目录**下载解压；用了 `--no_embed` 时尤其必须整目录一起拷 |
| 11 | 自己的图识别老错 | 先跑 `python custom_input.py inputs` 看是否全 `[OK]`；拍图时一张一个数字、写大写正、光线均匀；文件名数字开头才会计对错 |
| 12 | 手改了 HTML / PNG，重跑又变回原样 | 这些文件是**程序生成的**，每次运行整份覆盖。想改效果请改代码后重跑，或直接接受生成结果 |
| 13 | 忘了上次训练用的参数 | 打开 `logs/<run名>/config.json`，里面记着那次全部参数 |
| 14 | 中文乱码 / 图片里英文 | 正常：AI Studio 没有中文字体，PNG 图内文字用英文；HTML 网页里是中文（字体由你本地浏览器提供） |

---

## 附：命令速查（一张卡片抄走）

```bash
# 环境自检
python -c "import paddle; print(paddle.__version__)"

# 训练（lenet 默认参数示例）
python main.py --mode train --model_type lenet --lr 0.001 --epochs 5 --run_name demo

# 一键演示（训练 2 轮 + 自动推理出图）
python main.py --mode demo --run_name demo

# 推理：MNIST 示例图
python main.py --mode inference --run_name demo --num_show 9

# 图片自检（不需要模型）→ 识别自己手写的图
python custom_input.py inputs
python main.py --mode custom --run_name demo

# 全测试集验证
python main.py --mode eval --run_name demo

# 运维巡检 → 成果外发（顺序固定，不可颠倒）
python main.py --mode ops
python main.py --mode report
```
