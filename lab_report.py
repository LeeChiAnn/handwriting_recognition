"""lab_report.py —— 离线成果包：三份可下载的自包含 HTML（零 paddle / 零 numpy / 零 matplotlib）

一次生成三份，全部落在 outputs/reports/：

| 文件 | 回答什么问题 | 数据来源 |
|---|---|---|
| `model_report.html` | 这个模型到底行不行？错在哪、哪些图它没把握 | `outputs/eval/eval_results.json` |
| `lab_archive.html`   | 我做过哪些实验、所有图和数据留档在哪 | `logs/` + `outputs/` 全部产物 |
| `index.html`         | 从哪儿进去看 | 上面两份 + `outputs/ops/dashboard.html` |

设计要点（对应课程要求「学生下载到本地就能看」）：
1. **图片一律 base64 内嵌**，所以单文件下载后断网也能看图；
   嫌文件大可用 `--no_embed` 改成相对路径引用（那时必须整目录一起拷）。
2. **曲线用 SVG 现画**，直接读 `history.json`，不依赖 `curves.png` 是否生成成功；
   PNG 只是并排放一份做对照。
3. **上游产物缺失时不留白**：页面会写清楚缺哪个文件、用哪条命令生成，
   这样在本地打开远端跑出来的报告，也能看懂哪些环节还没做。
"""

import glob
import json
import os
import shutil
import time

from html_kit import (AMBER, GREEN, INDIGO, RED, bar_row, details, esc, fmt_bytes,
                      fmt_int, kpis, make_logger, page, png_is_small_square, score_color,
                      shots, svg_bars, svg_donut, svg_heat, svg_line, table)

MAX_LOG_LINES = 400          # 训练日志进 HTML 的行数上限，避免单页过大
MAX_SHOTS = 60               # 单个图集最多内嵌多少张


# ---------------------------------------------------------------- 读取层

def _load_json(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:                                  # 坏文件也要能出报告
        return {"_error": "%s（%s）" % (type(e).__name__, e), "_path": path}


def _paths(cfg):
    """把所有产物路径集中算一次，后面各渲染函数只认这张表。"""
    out = os.path.abspath(cfg.output_dir)
    ops = os.path.abspath(cfg.ops_dir)
    return {
        "root": os.path.abspath(os.path.dirname(__file__)),
        "log_dir": os.path.abspath(cfg.log_dir),
        "output_dir": out,
        "report_dir": os.path.join(out, "reports"),
        "eval_dir": os.path.join(out, "eval"),
        "eval_json": os.path.join(out, "eval", "eval_results.json"),
        "eval_conf_png": os.path.join(out, "eval", "confusion.png"),
        "examples_dir": os.path.join(out, "examples"),
        "examples_json": os.path.join(out, "examples", "inference_results.json"),
        "custom_dir": os.path.join(out, "custom"),
        "custom_json": os.path.join(out, "custom", "custom_results.json"),
        "ops_dir": ops,
        "ops_json": os.path.join(ops, "ops_report.json"),
        "ops_png": os.path.join(ops, "dashboard.png"),
        "ops_html": os.path.join(ops, "dashboard.html"),
        "inputs_dir": os.path.abspath(os.path.expanduser(getattr(cfg, "image_dir", None) or "./inputs")),
    }


def _list_runs(p, focus=None):
    """扫 logs/ 下所有 run，把 4 个 json + 日志 + 曲线图一次性读齐。"""
    runs = []
    for d in sorted(glob.glob(os.path.join(p["log_dir"], "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if focus and name != focus:
            continue
        logs = sorted(glob.glob(os.path.join(d, "train_*.log")))
        runs.append({
            "name": name, "dir": d,
            "config": _load_json(os.path.join(d, "config.json")),
            "history": _load_json(os.path.join(d, "history.json")),
            "record": _load_json(os.path.join(d, "dataset_record.json")),
            "curves": os.path.join(d, "curves.png"),
            "log_path": logs[-1] if logs else None,
            "n_logs": len(logs),
            "files": sorted(glob.glob(os.path.join(d, "*"))),
        })
    return runs


def _read_text(path, limit=MAX_LOG_LINES):
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    tail = ""
    if len(lines) > limit:
        tail = "\n… 中间省略 %d 行（完整内容见 %s）" % (len(lines) - limit, os.path.basename(path))
        lines = lines[:limit // 2] + [tail + "\n"] + lines[-limit // 2:]
    return "".join(lines)


def _all_images(p, cap=MAX_SHOTS):
    """项目里生成过的每一张图（含 preview / errors），用于档案袋的「全部图像成果」清单。"""
    roots = [p["log_dir"], p["output_dir"], p["inputs_dir"]]
    found, seen = [], set()
    for r in roots:
        for path in sorted(glob.glob(os.path.join(r, "**", "*.png"), recursive=True)):
            key = os.path.abspath(path)
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    found.sort(key=lambda x: -os.path.getsize(x))
    return found, len(found)


# ---------------------------------------------------------------- 公共小构件

def _missing_box(what, how):
    return ('<div class="miss"><b>%s</b><br>%s<br><code>%s</code></div>'
            % (esc(what), "这份内容需要先生成上游产物。", esc(how)))


def _acc_color(acc):
    if acc >= 0.98:
        return GREEN
    if acc >= 0.95:
        return INDIGO
    if acc >= 0.90:
        return AMBER
    return RED


def _final_from_history(h):
    """history.json 里取末轮与最优轮，供对账用。"""
    if not isinstance(h, dict):
        return None
    va = [v for v in (h.get("val_acc") or []) if isinstance(v, (int, float))]
    tr = [v for v in (h.get("train_acc") or []) if isinstance(v, (int, float))]
    if not va:
        return None
    return {"epochs": len(va), "val_acc": va[-1], "train_acc": tr[-1] if tr else None,
            "best_val_acc": max(va)}


# ---------------------------------------------------------------- ① 模型效果验证页

def render_model_report(ev, custom, p, embed, focus_run=None):
    """model_report.html：把「模型效果」一次讲完——总分、混淆、错分图、没把握的图。"""
    base = p["report_dir"]
    body = []

    if not isinstance(ev, dict) or "confusion" not in (ev or {}):
        why = ev.get("_error") if isinstance(ev, dict) else None
        head = ('<div class="card"><h2 style="margin-top:0">尚未生成验证数据</h2>'
                + _missing_box("缺少 outputs/eval/eval_results.json"
                               + ("（读取失败：%s）" % why if why else ""),
                               "python main.py --mode eval --run_name %s" % (focus_run or "demo"))
                + '<div class="sub" style="margin-top:12px">这一步要把整个测试集跑一遍推理，'
                  '<b>依赖 paddle</b>，因此请在百度 AI Studio 上执行；生成后把 '
                  '<code>outputs/</code> 整个目录下载回本地，再跑一次 '
                  '<code>python main.py --mode report</code> 即可把结果渲染进本页。'
                  '</div></div>')
        body.append(head)
        ev = None
    else:
        o, c = ev["overall"], ev["confidence"]
        acc = o["accuracy"]
        body.append(
            '<div class="card"><h2 style="margin-top:0">整体表现</h2>'
            '<div class="grid" style="align-items:center">'
            '<div style="text-align:center">%s<div class="muted" style="font-size:11px">%s 张测试图</div></div>'
            '<div>%s</div></div>%s</div>'
            % (svg_donut(acc, "accuracy", "测试集准确率", color=_acc_color(acc)),
               fmt_int(ev["dataset"]["num_samples"]),
               svg_bars(c["hist_10"], labels=["%.1f" % (i / 10.0) for i in range(10)],
                        xlabel="模型置信度（softmax 最大值）所在区间", height=170),
               kpis([("准确率", "%.2f%%" % (acc * 100)),
                     ("macro F1", o["macro_f1"]),
                     ("错分张数", "%d / %d" % (ev["errors"]["total"], ev["dataset"]["num_samples"])),
                     ("置信度<0.90", c["num_below_090"]),
                     ("中位置信度", c["quantiles"].get("0.5", "-")),
                     ("模型体积", fmt_bytes(ev.get("model_bytes")))])))

        # ---- 每类指标：光看总分不够，要能指出「哪一类拖后腿」
        pc_rows = []
        for r in ev["per_class"]:
            pc_rows.append([
                "<b>%d</b>" % r["label"], fmt_int(r["support"]), fmt_int(r["correct"]),
                "%.4f" % r["precision"], "%.4f" % r["recall"], "%.4f" % r["f1"],
                bar_row("", r["recall"], _acc_color(r["recall"]), "", label_w=0)])
        body.append('<div class="card"><h2 style="margin-top:0">每类指标（0-9）</h2>%s'
                    '<div class="sub" style="margin-top:8px">召回率条越长说明这一类越少被认错；'
                    '哪一类明显短一截，就该去看那一类的错分图。</div></div>'
                    % table(["数字", "样本数", "判对", "精确率", "召回率", "F1", "召回率条"],
                            pc_rows, num_cols=(1, 2, 3, 4, 5)))

        # ---- 混淆矩阵：SVG 版可 hover，PNG 版方便直接截图交作业
        labels = [str(x) for x in ev.get("labels") or range(len(ev["confusion"]))]
        pairs = ev["errors"].get("top_pairs") or []
        pair_html = "".join('<span class="chip" style="background:%s">%d → %d：%d 张</span> '
                            % (AMBER, t, q, n) for t, q, n in pairs) or '<span class="muted">无</span>'
        body.append(
            '<div class="card"><h2 style="margin-top:0">混淆矩阵（谁被认成了谁）</h2>'
            '<div class="grid"><div>%s</div><div>%s</div></div>'
            '<div style="margin-top:10px">最容易混淆：%s</div></div>'
            % (svg_heat(ev["confusion"], labels, labels),
               shots([(p["eval_conf_png"], "confusion.png（matplotlib 版，可截图提交）")],
                     embed=embed, base_dir=base),
               pair_html))

        # ---- 错分样本：这是最有教学价值的一屏
        items = ev["errors"].get("items") or []
        if items:
            cards = [(os.path.join(p["eval_dir"], it["image"]),
                      "#%d 真实 %d → 预测 %d　p=%.3f" % (it["index"], it["true"], it["pred"], it["prob"]),
                      True) for it in items[:MAX_SHOTS]]
            body.append('<div class="card"><h2 style="margin-top:0">错分样本图集（%d 张，已存盘）</h2>%s'
                        '<div class="sub" style="margin-top:8px">原图在 <code>outputs/eval/errors/</code>，'
                        '文件名 <code>e序号_真实_预测.png</code>。看看这些图：是人眼也分不清，还是模型训练不够？</div></div>'
                        % (len(items), shots(cards, embed=embed, base_dir=base)))

        # ---- 与训练日志对账：训练时说 98%，测试实测多少？
        body.append(_cross_check_block(ev, p, focus_run))

    # ---- 用户自提图片：外部验证的另一半（模型遇到"没见过的分布"表现如何）
    body.append(_custom_block(custom, p, embed))

    meta_line = ("数据来源：<code>outputs/eval/eval_results.json</code>　渲染于 %s　"
                 "本页自包含（图片 base64 内嵌、无外链、无 JS），下载到本地双击即可打开"
                 % time.strftime("%Y-%m-%d %H:%M:%S"))
    if ev:
        meta_line += ("　｜　run <code>%s</code>　结构 <b>%s</b>　权重 <code>%s</code>　验证于 %s"
                      % (esc(ev["run_name"]), esc(ev["model_type"]), esc(ev["model_path"]),
                         esc(ev["generated_at"])))
    h1 = "模型效果验证 · Model Evaluation"
    return page(h1, h1, meta_line, "".join(body),
                footer="手写识别实训 · 离线成果包 ①模型效果验证", back=True)


def _cross_check_block(ev, p, focus_run):
    """把测试集实测准确率与训练时记录的 val_acc 放一起，差异过大就是问题。"""
    hist = _load_json(os.path.join(p["log_dir"], str(ev.get("run_name")), "history.json"))
    f = _final_from_history(hist)
    if not f:
        return ('<div class="card"><h2 style="margin-top:0">与训练记录对账</h2>'
                + _missing_box("缺少 logs/%s/history.json" % (ev.get("run_name") or "?"),
                               "重跑该 run 的训练") + "</div>")
    diff = round(ev["overall"]["accuracy"] - f["val_acc"], 4)
    flag = ('<span class="ok">一致（差 %.4f）</span>' % abs(diff)) if abs(diff) <= 0.02 \
        else ('<span class="bad">差 %.4f，需查：验证集是否被污染 / 权重与日志是否同一次训练</span>' % diff)
    return ('<div class="card"><h2 style="margin-top:0">与训练记录对账</h2>%s'
            '<div style="margin-top:8px">结论：%s</div></div>'
            % (table(["口径", "数值", "说明"],
                     [["训练末轮 val_acc（history.json）", "%.4f" % f["val_acc"],
                       "训练时把测试集当验证集用，%d 轮" % f["epochs"]],
                      ["训练最优 val_acc", "%.4f" % f["best_val_acc"], "best_model.pdparams 来自这一轮"],
                      ["本次全测试集实测", "%.4f" % ev["overall"]["accuracy"],
                       "eval_model.py 重新推理所得"]], num_cols=(1,)), flag))


def _custom_block(custom, p, embed):
    """用户自提 PNG 的识别结果：原图 vs 模型真正看到的 28×28 并排对照。"""
    if not isinstance(custom, dict) or not custom.get("results"):
        return ('<div class="card"><h2 style="margin-top:0">你手写的图（inputs/）</h2>'
                + _missing_box("缺少 outputs/custom/custom_results.json",
                               "python main.py --mode custom --run_name <run名>") + "</div>")
    rows, cards = [], []
    n_failed = 0
    for it in custom["results"]:
        if it.get("error"):
            n_failed += 1
            rows.append([esc(it["file"]), '<span class="bad">预处理被拒</span>', "-", "-",
                         esc(it["error"])])
            continue
        t, pr, ok = it.get("true"), it.get("pred"), it.get("correct")
        verdict = ('<span class="ok">✓ 判对</span>' if ok else '<span class="bad">✗ 判错（真实 %s）</span>' % t) \
            if ok is not None else '<span class="muted">未标答案</span>'
        top3 = " ".join("%d:%.2f" % (k, v) for k, v in (it.get("top3") or []))
        rows.append([esc(it["file"]), "<b>%s</b>" % pr, "%.3f" % it.get("prob", 0), verdict, top3])
        cards.append((os.path.join(p["custom_dir"], it["saved_image"]),
                      "%s → %s" % (it["file"], pr), True))
        cards.append((os.path.join(custom.get("source_image_dir") or p["inputs_dir"], it["file"]),
                      "你的原图 %s" % it["file"], False))
    acc = custom.get("accuracy_on_labeled")
    head = kpis([("输入张数", custom.get("num_images")),
                 ("成功送入模型", custom.get("num_succeeded")),
                 ("预处理失败", n_failed),
                 ("标注样本准确率", acc if acc is not None else "-"),
                 ("引用 run", custom.get("run_name") or "-")])
    return ('<div class="card"><h2 style="margin-top:0">你手写的图（inputs/ → 模型看到的 28×28）</h2>%s%s'
            '<h3>识别明细</h3>%s'
            '<div class="sub">左边是模型真正看到的 28×28（已居中/缩放/标准化），右边是你的原图。'
            '识别错了先看左边：如果左边已经糊了，问题在预处理而不是模型。</div></div>'
            % (head, shots(cards[:MAX_SHOTS], embed=embed, base_dir=p["report_dir"]),
               table(["文件", "预测", "置信度", "判定", "top3"], rows)))


# ---------------------------------------------------------------- ② 实验档案袋页

def render_lab_archive(runs, p, embed, cfg_meta, ops):
    """lab_archive.html：所有 run 的参数、曲线、日志、留档、以及**全部图像成果**。"""
    base = p["report_dir"]
    total = sum(os.path.getsize(f) for r in runs for f in r["files"] if os.path.isfile(f))
    imgs, n_img = _all_images(p)          # 只扫一次磁盘，下面两处复用
    body = ['<div class="card"><h2 style="margin-top:0">实验总览</h2>%s</div>' % kpis([
        ("run 数", len(runs)),
        ("训练产物体积", fmt_bytes(total)),
        ("图像成果", "%d 张" % n_img),
        ("运维健康度", (str(ops.get("score")) + " / " + str(ops.get("grade"))) if ops else "-"),
        ("最优 run", (ops or {}).get("summary", {}).get("best_run") or "-"),
    ])]

    # ---- 参数对比表：一行一个 run，横向对比"我改了什么"
    keys = ["model_type", "learning_rate", "epochs", "batch_size", "optimizer",
            "hidden_size", "dropout", "seed"]
    if runs:
        rows = []
        for r in runs:
            c = r["config"] or {}
            h = _final_from_history(r["history"])
            rows.append(["<b><code>%s</code></b>" % esc(r["name"])]
                        + [esc(c.get(k, "-")) for k in keys]
                        + ["%.4f" % h["val_acc"] if h else '<span class="bad">无记录</span>',
                           "%.4f" % h["best_val_acc"] if h else "-",
                           fmt_bytes(sum(os.path.getsize(f) for f in r["files"] if os.path.isfile(f)))])
        body.append('<div class="card"><h2 style="margin-top:0">参数 × run 对比</h2>%s'
                    '<div class="sub" style="margin-top:8px">这张表就是实验记录本：'
                    '换 <code>--run_name</code> 才不会覆盖上一次的结果。</div></div>'
                    % table(["run"] + keys + ["末轮 val_acc", "最优 val_acc", "产物体积"],
                            rows, num_cols=set(range(len(keys) + 1, len(keys) + 4))))

    # ---- 每个 run 一节
    for r in runs:
        body.append(_run_section(r, p, embed, base))

    # ---- 全部图像成果清单（用户明确要“所有图都能在一个 HTML 里看”）
    cards = [(x, "%s（%s）" % (os.path.relpath(x, p["root"]).replace(os.sep, "/"),
                               fmt_bytes(os.path.getsize(x))),
              png_is_small_square(x)) for x in imgs]
    rows = [["<code>%s</code>" % esc(os.path.relpath(x, p["root"]).replace(os.sep, "/")),
             fmt_bytes(os.path.getsize(x)),
             time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(x)))] for x in imgs]
    body.append('<div class="card"><h2 style="margin-top:0">全部图像成果（%d 张）</h2>%s'
                '<h3>清单</h3>%s'
                '<div class="sub">超过 %d 张时只内嵌体积最大的前 %d 张，其余仍在磁盘上按上表路径可查。</div></div>'
                % (n_img, shots(cards[:MAX_SHOTS], embed=embed, base_dir=base),
                   table(["相对项目根路径", "体积", "生成时间"], rows),
                   MAX_SHOTS, MAX_SHOTS))

    # ---- 推理成果（outputs/examples）
    inf = _load_json(p["examples_json"])
    if isinstance(inf, dict) and inf.get("results"):
        cards = [(os.path.join(p["examples_dir"], "sample_%d_true%d_pred%d.png"
                               % (x["index"], x["true"], x["pred"])),
                  "#%d true=%d pred=%d" % (x["index"], x["true"], x["pred"]), True)
                 for x in inf["results"]]
        body.append('<div class="card"><h2 style="margin-top:0">测试集示例推理（--mode inference）</h2>'
                    '%s%s<div class="sub">数据来源：<code>%s</code>，run <code>%s</code></div></div>'
                    % (kpis([("示例数", inf.get("num_examples")),
                             ("示例准确率", inf.get("accuracy_on_examples")),
                             ("引用 run", inf.get("run_name") or "-")]),
                       shots(cards, embed=embed, base_dir=base)
                       + shots([(os.path.join(p["examples_dir"], "predictions.png"),
                                 "predictions.png 综合拼图")], embed=embed, base_dir=base),
                       esc(p["examples_json"]), esc(inf.get("run_name"))))

    sub = ("生成于 %s　｜　run 范围：%s　｜　图片内嵌：%s　｜　本页自包含，可下载到本地打开"
           % (time.strftime("%Y-%m-%d %H:%M:%S"),
              esc(cfg_meta.get("run_scope")), "是" if embed else "否（用相对路径）"))
    h1 = "实验档案袋 · Lab Archive"
    return page(h1, h1, sub, "".join(body), footer="手写识别实训 · 离线成果包 ②实验档案袋", back=True)


def _run_section(r, p, embed, base):
    """单个 run 的完整档案：参数快照 + 曲线（SVG 现画）+ 留档 + 日志原文 + 产物清单。"""
    c, h, rec = r["config"] or {}, r["history"] or {}, r["record"] or {}
    parts = ['<div class="card"><h2>run：<code>%s</code></h2>' % esc(r["name"])]
    curves_shot = shots([(r["curves"], "logs/%s/curves.png" % r["name"])],
                        embed=embed, base_dir=base)
    if h.get("train_loss") or h.get("train_acc"):
        loss_svg = svg_line([("train_loss", h.get("train_loss"), INDIGO),
                             ("val_loss", h.get("val_loss"), RED)],
                            width=520, height=190, title="Loss")
        acc_svg = svg_line([("train_acc", h.get("train_acc"), INDIGO),
                            ("val_acc", h.get("val_acc"), GREEN)],
                           width=520, height=190, title="Accuracy")
        parts.append('<div class="grid">'
                     '<div><h3 style="margin-top:0">损失曲线（读 history.json 现画）</h3>%s</div>'
                     '<div><h3 style="margin-top:0">准确率曲线（同上）</h3>%s</div></div>'
                     '<h3>curves.png（matplotlib 版，截图交作业用）</h3>%s'
                     % (loss_svg, acc_svg, curves_shot))
    else:
        parts.append('<div class="miss">缺少 history.json，无法重绘曲线（训练可能未跑完就被中断）</div>'
                     + curves_shot)

    if c:
        parts.append("<h3>参数快照 config.json</h3>%s"
                     % table(["参数", "值"], [[k, v] for k, v in sorted(c.items())]))
    if rec:
        parts.append("<h3>数据集留档 dataset_record.json</h3>%s"
                     % table(["字段", "值"], [[k, v] for k, v in sorted(rec.items())]))
    txt = _read_text(r["log_path"])
    if txt:
        parts.append("<h3>训练日志原文</h3>%s"
                     % details("%s（%s，最多显示 %d 行）" % (
                         os.path.basename(r["log_path"]), fmt_bytes(os.path.getsize(r["log_path"])),
                         MAX_LOG_LINES),
                        "<pre>%s</pre>" % esc(txt)))
    if r["n_logs"] > 1:
        parts.append('<div class="warned" style="font-size:12px">本 run 有 %d 份 train_*.log，'
                     "只归档了最新一份，其余请确认是否为重复训练。</div>" % r["n_logs"])
    parts.append("<h3>产物清单</h3>%s</div>"
                 % table(["文件", "体积", "最后修改"],
                         [["<code>%s</code>" % esc(os.path.basename(f)),
                           fmt_bytes(os.path.getsize(f)),
                           time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f)))]
                          for f in r["files"] if os.path.isfile(f)]))
    return "".join(parts)


# ---------------------------------------------------------------- ③ 入口页

def render_index(files, p, ops):
    """index.html：三份成果的入口 + 缺件时的生成命令。"""
    desc = {
        "model_report.html": ("模型效果验证", "整体准确率、混淆矩阵、每类 P/R/F1、错分样本图集、"
                                             "低置信清单、与训练记录对账、你手写图的识别结果"),
        "lab_archive.html": ("实验档案袋", "各 run 参数对比、训练曲线（SVG 现画）、日志原文、"
                                          "数据留档、全部图像成果清单"),
        "dashboard.html": ("数据运维看板", "五维健康度、资产台账、分区域存储账、告警清单与整改优先级"),
    }
    cards = []
    for fname, (title, what) in desc.items():
        path = files.get(fname)
        if path and os.path.isfile(path):
            size = fmt_bytes(os.path.getsize(path))
            mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
            # 运维看板在 outputs/ops/ 而入口页在 outputs/reports/，必须算相对路径才能点得开
            href = os.path.relpath(path, p["report_dir"]).replace(os.sep, "/")
            cards.append('<div class="card"><h2 style="margin-top:0">%s</h2>'
                         '<div class="sub">%s</div>'
                         '<a class="btn" href="%s">打开 %s（%s · %s）</a></div>'
                         % (esc(title), esc(what), esc(href), esc(fname), size, mt))
        else:
            cmd = {"model_report.html": "python main.py --mode eval --run_name demo && python main.py --mode report",
                   "lab_archive.html": "python main.py --mode report",
                   "dashboard.html": "python main.py --mode ops"}[fname]
            cards.append('<div class="card"><h2 style="margin-top:0">%s</h2>'
                         '<div class="sub">%s</div>%s</div>'
                         % (esc(title), esc(what), _missing_box("尚未生成", cmd)))

    health = ""
    if isinstance(ops, dict):
        sc = ops.get("score")
        health = ('<div class="card"><h2 style="margin-top:0">当前体检结论</h2>%s%s</div>'
                  % (kpis([("健康度", sc), ("等级", ops.get("grade")),
                          ("ERROR", ops.get("summary", {}).get("error")),
                          ("WARN", ops.get("summary", {}).get("warn")),
                          ("run 数", ops.get("summary", {}).get("runs"))]),
                     bar_row("健康度", (sc or 0) / 100.0, score_color(sc or 0), str(sc))))
    how = ('<div class="card"><h2 style="margin-top:0">怎么用这三份 HTML</h2>'
           '<div class="sub" style="line-height:1.9">'
           '1. 在 AI Studio 上按顺序跑：<code>--mode train</code> → <code>--mode inference</code> → '
           '<code>--mode custom</code> → <code>--mode eval</code> → <code>--mode ops</code> → '
           '<code>--mode report</code>；<br>'
           '2. 把 <code>outputs/reports/</code> 里的单个 html 文件下载到本机，双击即可打开 —— '
           '图片已 base64 内嵌，<b>不需要联网、不需要拷贝其他文件、不需要装任何 Python 库</b>；<br>'
           '3. 想缩小文件体积（改走相对路径引用图片）就加 <code>--no_embed</code>，'
           '此时必须把整个 <code>outputs/</code> 目录一起拷走；<br>'
           '4. 交作业建议提交：<code>model_report.html</code> + <code>ops_report.md</code> + '
           '<code>curves.png</code>。</div></div>')
    h1 = "手写识别实训 · 离线成果包"
    sub = ("生成于 %s　｜　目录 <code>%s</code>　｜　三份 HTML 各自单文件自包含，可分别下载"
           % (time.strftime("%Y-%m-%d %H:%M:%S"), esc(p["report_dir"])))
    return page(h1, h1, sub, health + "".join(cards) + how,
                footer="课程《大数据综合实训 · 深度学习实训》配套 · 手写数字识别 MNIST", back=False)


# ---------------------------------------------------------------- 对外入口

def build_reports(cfg, logger=None):
    """生成三份 HTML（缺上游数据也会生成，页面里写清楚缺什么）。返回 {文件名: 绝对路径}。"""
    log = make_logger(logger)
    p = _paths(cfg)
    embed = bool(getattr(cfg, "embed_images", True))
    os.makedirs(p["report_dir"], exist_ok=True)

    ops = _load_json(p["ops_json"])
    if not isinstance(ops, dict):
        ops = None
    runs = _list_runs(p, focus=cfg.run_name)
    ev = _load_json(p["eval_json"])
    custom = _load_json(p["custom_json"])
    if isinstance(custom, dict) and custom.get("_error"):
        custom = None

    files = {}
    out = os.path.join(p["report_dir"], "model_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_model_report(ev if isinstance(ev, dict) else None, custom, p, embed,
                                    focus_run=cfg.run_name))
    files["model_report.html"] = out

    out = os.path.join(p["report_dir"], "lab_archive.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_lab_archive(runs, p, embed,
                                   {"run_scope": cfg.run_name or "logs/ 下全部 run"}, ops))
    files["lab_archive.html"] = out

    # 运维看板复制一份进 reports/：入口页第三张卡片就能和另两份一样“下整个 reports/ 即看”，
    # 不会因为本地没下 outputs/ops/ 而变成点不开的死链（正本仍在 outputs/ops/）。
    # 但只在内嵌模式下能这么做：--no_embed 时看板里用的是相对 ops_dir 的路径，
    # 换到 reports/ 目录就全部失效，那时直接指回正本。
    ops_src, ops_dst = p["ops_html"], os.path.join(p["report_dir"], "dashboard.html")
    if embed and os.path.exists(ops_src) and (not os.path.exists(ops_dst)
                             or os.path.getmtime(ops_src) > os.path.getmtime(ops_dst)):
        shutil.copy2(ops_src, ops_dst)
    files["dashboard.html"] = ops_dst if (embed and os.path.exists(ops_dst)) else ops_src
    out = os.path.join(p["report_dir"], "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_index(files, p, ops))
    files["index.html"] = out

    log("==== 离线成果包生成完毕（图片内嵌=%s）====", "是" if embed else "否")
    for name in ("index.html", "model_report.html", "lab_archive.html"):
        log("  %-20s -> %s（%s）", name, files[name], fmt_bytes(os.path.getsize(files[name])))
    log("  运维看板   -> %s（正本 %s）", files["dashboard.html"], p["ops_html"])
    log("把上面任一 .html 下载到本地双击即可打开；三份都无需联网。")
    return files
