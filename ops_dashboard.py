"""ops_dashboard.py —— 数据运维看板的 HTML 渲染器（纯标准库，零绘图依赖）

为什么除了 PNG 还要一份 HTML：
1. 很多本机/精简环境没装 matplotlib，PNG 出不来，但 HTML 一定能出；
2. HTML 自包含（CSS 内联、图表用内嵌 SVG、无外链、无 JS），
   在 AI Studio 里下载后双击即可打开，也能直接当作业提交物；
3. 中文问题：AI Studio 镜像没有中文字体，PNG 里写中文会变豆腐块，
   所以 **PNG 全英文、HTML 全中文**，两边各取所长。

由 data_ops.py 的 run_ops_audit() 自动调用，一般不需要单独运行。
"""


import html as _html
import math
import os

from data_ops import DIMENSIONS, ERROR, INFO, LEVEL_COLOR, MNIST_FILES, WARN, human_bytes
from html_kit import CSS as _CSS
from html_kit import esc as _esc, score_color as _score_color, shots


def svg_gauge(score):
    """半环仪表盘：用 SVG 圆弧 + stroke-dasharray 表达 0-100 健康度。"""
    r = 70.0
    arc = math.pi * r
    frac = max(0.0, min(1.0, score / 100.0))
    d = "M 30 100 A %.0f %.0f 0 0 1 170 100" % (r, r)
    return (
        '<svg viewBox="0 0 200 122" width="200" height="122">'
        '<path d="%s" fill="none" stroke="#e9edf3" stroke-width="15" stroke-linecap="round"/>'
        '<path d="%s" fill="none" stroke="%s" stroke-width="15" stroke-linecap="round" '
        'stroke-dasharray="%.1f %.1f"/>'
        '<text x="100" y="93" text-anchor="middle" font-size="30" font-weight="700" '
        'fill="#1a1d24">%.1f</text>'
        '<text x="100" y="114" text-anchor="middle" font-size="11" fill="#68707f">'
        "HEALTH SCORE / 100</text></svg>"
    ) % (d, d, _score_color(score), arc * frac, arc, score)


def svg_stacked_storage(runs, cats, colors, width=520):
    """每个 run 一条堆叠横条：直观回答「磁盘到底被什么占满了」。"""
    rows = sorted(runs, key=lambda r: -r["total_bytes"])[:10]
    if not rows:
        return '<p class="muted">无 run 产物，无法统计占用。</p>'
    left, rowh, gap = 132, 30, 6
    height = len(rows) * (rowh + gap) + 34
    bar_w = width - left - 76
    maxb = max(r["total_bytes"] for r in rows) or 1
    p = ['<svg viewBox="0 0 %d %d" width="100%%" height="%d">' % (width, height, height)]
    y = 6
    for r in rows:
        p.append('<text x="%d" y="%d" text-anchor="end" font-size="11" fill="#333">%s</text>'
                 % (left - 8, y + 17, _esc(r["name"][:16])))
        x = float(left)
        for cat in cats:
            v = r["by_cat"][cat]
            if v <= 0:
                continue
            w = bar_w * v / maxb
            p.append('<rect x="%.1f" y="%d" width="%.1f" height="22" rx="3" fill="%s">'
                     '<title>%s: %s</title></rect>'
                     % (x, y, max(w, 1), colors[cat], _esc(cat), human_bytes(v)))
            x += w
        p.append('<text x="%.1f" y="%d" font-size="11" fill="#666">%s</text>'
                 % (x + 6, y + 17, human_bytes(r["total_bytes"])))
        y += rowh + gap
    lx = 10
    step = max(70.0, (width - 20) / len(cats))
    for cat in cats:
        p.append('<rect x="%.0f" y="%d" width="11" height="11" rx="2" fill="%s"/>'
                 '<text x="%.0f" y="%d" font-size="10" fill="#666">%s</text>'
                 % (lx, height - 20, colors[cat], lx + 15, height - 11, _esc(cat)))
        lx += step
    p.append("</svg>")
    return "".join(p)


def svg_acc_bars(runs, width=520):
    """分组柱状：每 run 的 train_acc / val_acc，两柱高差就是泛化间隙。"""
    rows = [r for r in runs if r.get("final")]
    if not rows:
        return '<p class="muted">没有可用的 history.json，指标图空。</p>'
    height, base, left = 214, 166, 34
    slot = max(70, (width - left - 16) // len(rows))
    p = ['<svg viewBox="0 0 %d %d" width="100%%" height="%d">' % (width, height, height)]
    for gv in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = base - gv * 140
        p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#eef1f5"/>'
                 '<text x="4" y="%.1f" font-size="9" fill="#9aa3b2">%.2f</text>'
                 % (left, yy, width - 8, yy, yy + 3, gv))
    for i, r in enumerate(rows):
        f = r["final"]
        x0 = left + 10 + i * slot
        for j, (key, col) in enumerate((("train_acc", "#6366f1"), ("val_acc", "#12a06b"))):
            v = f.get(key) or 0
            h = v * 140
            xx = x0 + j * 26
            p.append('<rect x="%d" y="%.1f" width="20" height="%.1f" rx="3" fill="%s">'
                     '<title>%s %s=%.4f</title></rect>'
                     % (xx, base - h, h, col, _esc(r["name"]), key, v))
            p.append('<text x="%d" y="%.1f" text-anchor="middle" font-size="9" fill="#4b5568">%.3f</text>'
                     % (xx + 10, base - h - 4, v))
        p.append('<text x="%d" y="%d" text-anchor="middle" font-size="10" fill="#4b5568">%s</text>'
                 % (x0 + 26, base + 15, _esc(r["name"][:12])))
        if f.get("model_type"):
            p.append('<text x="%d" y="%d" text-anchor="middle" font-size="9" fill="#9aa3b2">%s</text>'
                     % (x0 + 26, base + 28, _esc(f["model_type"])))
    p.append('<rect x="%d" y="%d" width="11" height="11" rx="2" fill="#6366f1"/>'
             '<text x="%d" y="%d" font-size="10" fill="#666">train_acc</text>'
             '<rect x="%d" y="%d" width="11" height="11" rx="2" fill="#12a06b"/>'
             '<text x="%d" y="%d" font-size="10" fill="#666">val_acc</text>'
             % (left, height - 14, left + 15, height - 5, left + 104, height - 14, left + 119, height - 5))
    p.append("</svg>")
    return "".join(p)


def svg_spark(vals, w=104, h=26, color="#6366f1"):
    """val_acc 迷你趋势线（sparkline）：一眼看出训练后期是否退化。"""
    vals = [v for v in (vals or []) if isinstance(v, (int, float))]
    if len(vals) < 2:
        return '<span class="muted">-</span>'
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1e-6
    step = w / (len(vals) - 1)
    pts = " ".join("%.1f,%.1f" % (i * step, h - 2 - (v - lo) / rng * (h - 5))
                   for i, v in enumerate(vals))
    return ('<svg width="%d" height="%d"><polyline points="%s" fill="none" '
            'stroke="%s" stroke-width="1.6"/></svg>' % (w, h, pts, color))


def _how_to_fix(msg):
    """把预处理报因翻译成一句“学生下一步该干什么”。"""
    if "对比度" in msg:
        return "改单字、提高对比度（背景与笔迹不能同色）"
    if "没检测到笔迹" in msg:
        return "把笔画画粗画黑，或确认图里真有一个数字"
    if "几乎都判成了笔迹" in msg:
        return "只留一个数字，去掉阴影与背景杂物"
    return "见报因；可用 python custom_input.py inputs 自检"


def render_html_dashboard(rep, embed=True):
    """把巡检报告渲染成一页自包含 HTML（返回字符串）。

    embed=True 时把图像成果 base64 内嵌，学生只下载这一个文件也能看到图；
    embed=False 改走相对路径（体积小，但得整目录拷走）。"""
    s, meta, src = rep["summary"], rep["meta"], rep["source"]
    cats = ["weights", "logs", "images", "json", "other"]
    colors = {"weights": "#6366f1", "logs": "#10b981", "images": "#f59e0b",
              "json": "#38bdf8", "other": "#94a3b8"}
    dim_cn = dict((k, label) for k, label, _ in DIMENSIONS)

    chips = " ".join([
        '<span class="chip" style="background:%s">ERROR %d</span>' % (LEVEL_COLOR[ERROR], s["error"]),
        '<span class="chip" style="background:%s">WARN %d</span>' % (LEVEL_COLOR[WARN], s["warn"]),
        '<span class="chip" style="background:%s">INFO %d</span>' % (LEVEL_COLOR[INFO], s["info"]),
    ])

    kpi = "".join([
        '<div class="kpi"><span>健康度等级</span><b style="color:%s">%s</b></div>'
        % (_score_color(rep["score"]), _esc(rep["grade"])),
        '<div class="kpi"><span>巡检 run 数</span><b>%d</b></div>' % s["runs"],
        '<div class="kpi"><span>run 产物占用</span><b>%s</b></div>' % human_bytes(rep["storage"]["runs_bytes"]),
        '<div class="kpi"><span>全项目产物</span><b>%s</b></div>' % human_bytes(
            rep["storage"].get("total_bytes", rep["storage"]["runs_bytes"])),
        '<div class="kpi"><span>数据集占用</span><b>%s</b></div>' % human_bytes(rep["storage"]["source_bytes"]),
        '<div class="kpi"><span>数据文件校验</span><b>%d/%d</b></div>' % (src["ok_count"], len(MNIST_FILES)),
        '<div class="kpi"><span>最优 run</span><b style="font-size:13px">%s</b></div>'
        % _esc(s["best_run"] or "-"),
    ])

    dims_html = []
    for key, label, w in DIMENSIONS:
        sc = rep["dimensions"][key]
        dims_html.append(
            '<div class="dimrow"><span>%s</span><div class="track">'
            '<div class="fill" style="width:%.0f%%;background:%s"></div></div>'
            "<b>%.0f</b><span class=\"muted\" style=\"width:34px\">%d%%</span></div>"
            % (_esc(label), sc, _score_color(sc), sc, int(w * 100)))

    cols = [("log", "日志")] + [(a["key"], a["label"]) for a in _run_artifacts()]
    matrix = ["<table><tr><th>run</th>" + "".join("<th>%s</th>" % _esc(c[1]) for c in cols)
              + "<th>体积</th><th>最后更新</th></tr>"]
    for r in rep["runs"]:
        cells = []
        for key, _ in cols:
            got = r["artifacts"].get(key)
            cells.append('<td class="%s">%s</td>'
                         % ("cell-ok ok" if got else "cell-bad bad", "✓ 有" if got else "✗ 缺"))
        latest = max([i["mtime"] for i in r["files"].values()] or ["-"])
        matrix.append("<tr><td><code>%s</code></td>%s<td class=\"num\">%s</td>"
                      "<td class=\"muted\">%s（%.1f天前）</td></tr>"
                      % (_esc(r["name"]), "".join(cells), human_bytes(r["total_bytes"]),
                         _esc(latest), r["age_days"]))
    if not rep["runs"]:
        matrix.append('<tr><td colspan="10" class="muted">未发现任何 run 目录</td></tr>')
    matrix.append("</table>")

    metrics = ["<table><tr><th>run</th><th>结构</th><th>轮数</th><th>train_acc</th><th>val_acc</th>"
               "<th>泛化间隙</th><th>val_acc 走势</th><th>运维判定</th></tr>"]
    for r in rep["runs"]:
        f = r.get("final") or {}
        if not f:
            metrics.append('<tr><td><code>%s</code></td>'
                           '<td colspan="6" class="bad">无 history.json，训练指标不可追溯</td>'
                           '<td class="bad">不可提交</td></tr>' % _esc(r["name"]))
            continue
        gap = f["gap"]
        verdict = ('<span class="bad">过拟合</span>' if gap > 0.05
                   else '<span class="warned">待观察</span>' if gap > 0.02
                   else '<span class="ok">正常</span>')
        metrics.append(
            "<tr><td><code>%s</code></td><td>%s</td><td class=\"num\">%s</td>"
            "<td class=\"num\">%s</td><td class=\"num\">%s</td>"
            "<td class=\"num %s\">%+.4f</td><td>%s</td><td>%s</td></tr>"
            % (_esc(r["name"]), _esc(f.get("model_type", "?")), f["epochs"],
               f["train_acc"], f["val_acc"], "bad" if gap > 0.05 else "", gap,
               svg_spark((r["history"] or {}).get("val_acc")), verdict))
    metrics.append("</table>")

    src_rows = []
    for f in src["files"]:
        idx = f.get("idx") or {}
        if not f["exists"]:
            status = '<span class="bad">缺失</span>'
        elif f.get("drift"):
            status = '<span class="bad">基线漂移</span>'
        elif idx.get("ok"):
            status = '<span class="ok">正常</span>'
        else:
            status = '<span class="bad">结构异常</span>'
        src_rows.append(
            "<tr><td><code>%s</code></td><td>%s</td><td class=\"num\">%s</td>"
            '<td class="num">%s</td><td class="num">%s</td><td><code>%s</code></td><td>%s</td></tr>'
            % (_esc(f["name"]), _esc(f["split"]),
               human_bytes(f.get("size")) if f["exists"] else "-",
               idx.get("count") or "-", idx.get("actual") or "-",
               _esc((f.get("md5") or "-")[:16]), status))

    issue_rows = []
    for it in rep["issues"]:
        issue_rows.append(
            '<tr><td><span class="chip" style="background:%s">%s</span></td>'
            "<td><code>%s</code></td><td>%s</td><td><code>%s</code></td>"
            "<td>%s</td><td class=\"muted\">%s</td></tr>"
            % (LEVEL_COLOR[it["level"]], it["level"], _esc(it["code"]),
               _esc(dim_cn.get(it["dim"], it["dim"])), _esc(it["target"]),
               _esc(it["message"]), _esc(it["suggestion"] or "-")))
    if not issue_rows:
        issue_rows.append('<tr><td colspan="6" class="ok">无告警，本次巡检全部通过</td></tr>')

    todo = [i for i in rep["issues"] if i["level"] in (ERROR, WARN)][:8]
    actions = "".join(
        "<li><b>%s %s</b>：%s<br><span class=\"muted\">→ </span><code>%s</code></li>"
        % (_esc(i["level"]), _esc(i["code"]), _esc(i["message"]),
           _esc(i["suggestion"] or "见告警表")) for i in todo) or \
        '<li class="ok">无必须整改项，可直接进入参数对比实验。</li>'

    inf = rep["inference"]
    inf_line = ("推理成果：<code>%s</code>，示例 %s 张，示例准确率 <b>%s</b>"
                % (_esc(inf.get("run_name")), inf.get("num_examples"),
                   inf.get("accuracy_on_examples"))) if inf.get("exists") else \
        '推理成果：<span class="bad">尚无 inference_results.json</span>'

    # ---- 分区域存储账（回答“磁盘到底被哪一类产物吃掉了”）
    areas = rep["storage"].get("areas") or {}
    area_desc = {"logs": "训练产物 logs/", "data": "原始数据集 data/",
                 "outputs_examples": "示例输出 outputs/examples/",
                 "outputs_custom": "自提图结果 outputs/custom/",
                 "inputs": "自提原图 inputs/", "ops": "巡检报告 outputs/ops/"}
    amax = max(list(areas.values()) + [1])
    areas_html = "".join(
        '<div class="dimrow"><span style="width:170px">%s</span><div class="track">'
        '<div class="fill" style="width:%.1f%%;background:%s"></div></div>'
        '<b style="width:64px">%s</b></div>'
        % (_esc(area_desc.get(k, k)), 100.0 * v / amax,
           "#6366f1" if v == amax else "#93a0f5", human_bytes(v))
        for k, v in sorted(areas.items(), key=lambda kv: -kv[1]))

    # ---- 用户自提输入区（新数据接入体检）
    c = rep.get("custom") or {}
    ch = ['<div class="sub" style="margin-bottom:8px">输入目录 <code>%s</code>（%s）'
          '｜现有 PNG %d 张｜因格式被排除 %d 个｜成果 <code>%s</code>（%s）</div>'
          % (_esc(c.get("image_dir")), "存在" if c.get("dir_exists") else "不存在",
             len(c.get("pngs") or []), len(c.get("wrong_ext") or []),
             _esc(os.path.basename(str(c.get("results_path")))),
             "已生成" if c.get("exists") else "尚未生成")]
    if c.get("pngs"):
        ch.append("<table><tr><th>输入图</th><th class=\"num\">体积</th><th>最后修改</th>"
                  "<th>送模结果</th></tr>")
        by_file = c.get("by_file") or {}
        for f in c["pngs"][:12]:
            r = by_file.get(f["name"])
            if not c.get("exists"):
                cell = '<span class="muted">未跑推理</span>'
            elif r is None:
                cell = '<span class="bad">不在报告中</span>'
            elif r.get("error"):
                cell = '<span class="bad">✗ 预处理被拒</span>'
            elif r.get("correct") is None:
                cell = 'pred=%s <span class="muted">无答案</span>' % r.get("pred")
            else:
                cell = ('<span class="ok">✓ pred=%s</span>' % r.get("pred")) if r["correct"] \
                    else ('<span class="bad">✗ pred=%s/true=%s</span>' % (r.get("pred"), r.get("true")))
            ch.append('<tr><td><code>%s</code></td><td class="num">%s</td>'
                      '<td class="muted">%s</td><td>%s</td></tr>'
                      % (_esc(f["name"]), human_bytes(f["size"]), _esc(f["mtime"]), cell))
        if len(c["pngs"]) > 12:
            ch.append('<tr><td colspan="4" class="muted">… 共 %d 张</td></tr>' % len(c["pngs"]))
        ch.append("</table>")
    if c.get("wrong_ext"):
        ch.append('<div class="warned" style="font-size:12px;margin:6px 0">被排除的非 PNG：'
                  '<code>%s</code></div>' % _esc(", ".join(c["wrong_ext"][:8])))
    if c.get("exists"):
        ok = '<span class="ok">✅ 存在</span>' if not c.get("model_missing") else \
             '<span class="bad">❌ 已丢失（成果无法复现）</span>'
        ch.append('<div class="kpis" style="margin:10px 0">'
                  '<div class="kpi"><span>记录输入数</span><b>%s</b></div>'
                  '<div class="kpi"><span>预处理成功</span><b>%s</b></div>'
                  '<div class="kpi"><span>预处理失败</span><b class="%s">%d</b></div>'
                  '<div class="kpi"><span>无标准答案</span><b>%s</b></div>'
                  '<div class="kpi"><span>标注样本准确率</span><b>%s</b></div>'
                  '<div class="kpi"><span>引用 run</span><b style="font-size:13px">%s</b></div>'
                  "</div>"
                  % (c.get("num_images"), c.get("succeeded"),
                     "bad" if c.get("failed") else "ok", len(c.get("failed") or []),
                     c.get("unlabeled"), c.get("accuracy") if c.get("accuracy") is not None else "-",
                     _esc(c.get("run_name") or "-")))
        ch.append("<div style=\"font-size:12px\">模型权重可达性：%s　孤儿产物：%d 个</div>"
                  % (ok, len(c.get("orphans") or [])))
        if c.get("failed"):
            ch.append('<h2>数据质量工单（预处理被拒的图）</h2>'
                      "<table><tr><th>文件</th><th>拒因</th><th>怎么改</th></tr>")
            for f in c["failed"][:10]:
                ch.append("<tr><td><code>%s</code></td><td>%s</td><td>%s</td></tr>"
                          % (_esc(f["file"]), _esc(f["error"]), _esc(_how_to_fix(f["error"]))))
            ch.append("</table>")
    custom_html = "".join(ch)

    # ---- 图像成果：缺图时 shots 会显式标“未生成”，这本身就是运维信息
    img_items = [(os.path.join(meta["ops_dir"], "dashboard.png"),
                  "dashboard.png 四联看板（本页的静态图版）")]
    for r in rep["runs"]:
        img_items.append((os.path.join(r["path"], "curves.png"),
                          "logs/%s/curves.png 训练曲线" % r["name"]))
    img_items += [(os.path.join(meta["output_dir"], "examples", "predictions.png"),
                   "outputs/examples/predictions.png 示例识别拼图"),
                  (os.path.join(meta["output_dir"], "custom", "predictions_custom.png"),
                   "outputs/custom/predictions_custom.png 自提图识别拼图")]
    images_html = shots(img_items, embed=embed, base_dir=meta["ops_dir"])

    # 入口页存在才给返回按钮：单独下载本页时不能留一个点不开的死链
    nav = ('<a class="btn" href="../reports/index.html">← 返回成果包入口</a>'
           if os.path.exists(os.path.join(meta["output_dir"], "reports", "index.html")) else "")

    return (
        '<!doctype html><html lang="zh"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>手写识别项目 · 数据运维看板</title><style>%s</style></head><body><div class=\"wrap\">"
        "<h1>数据运维巡检看板 · DataOps Audit</h1>"
        '<div class="sub">生成于 %s ｜ 巡检范围 <code>%s</code> · <code>%s</code> · <code>%s</code>'
        "｜ run 范围：%s ｜ 保留期 %d 天</div>%s"
        '<div class="grid">'
        '<div class="card" style="text-align:center"><h2 style="margin-top:0">健康度总分</h2>%s'
        '<div style="margin:8px 0 2px">%s</div><div>%s</div><div class="kpis">%s</div></div>'
        '<div class="card"><h2 style="margin-top:0">五维健康度（扣分制 ERROR-35 / WARN-12 / INFO-3）</h2>%s'
        '<div class="muted" style="margin-top:10px;font-size:11px">磁盘/数据/日志均为只读扫描，'
        "本看板不会修改或删除任何产物。</div>"
        "<h2>存储占用构成（按 run 堆叠）</h2>%s<h2>分区域存储账</h2>%s</div>"
        '<div class="card"><h2 style="margin-top:0">训练指标对比（运维视角）</h2>%s'
        '<div class="muted" style="font-size:11px;margin-top:6px">两柱高度差即泛化间隙；'
        "差 &gt; 0.05 记过拟合告警。</div><h2>数据源体检</h2>"
        '<div class="muted" style="font-size:11px;margin-bottom:6px">实际命中目录：<code>%s</code></div>'
        '<table><tr><th>文件</th><th>拆分</th><th>体积</th><th>样本数</th><th>解压字节</th>'
        "<th>md5(前16)</th><th>状态</th></tr>%s</table>"
        '<div class="muted" style="font-size:11px;margin-top:6px">指纹基线：<code>%s</code></div></div>'
        "</div>"
        '<div class="card"><h2 style="margin-top:0">资产台账（run × 产物）</h2>%s</div>'
        '<div class="card"><h2 style="margin-top:0">训练指标明细</h2>%s'
        '<div class="sub" style="margin-top:10px">%s</div></div>'
        '<div class="card"><h2 style="margin-top:0">用户自提输入区 · 新数据接入体检</h2>%s</div>'
        '<div class="card"><h2 style="margin-top:0">告警清单（%d 项）</h2>'
        "<table><tr><th>级别</th><th>编号</th><th>维度</th><th>对象</th><th>问题</th>"
        "<th>建议动作</th></tr>%s</table></div>"
        '<div class="card"><h2 style="margin-top:0">整改优先级（先看这 8 条）</h2><ol>%s</ol></div>'
        '<div class="card"><h2 style="margin-top:0">图像成果（%s）</h2>%s'
        '<div class="muted" style="font-size:11px;margin-top:6px">'
        "图片已内嵌在本页里，所以只下载这一个 HTML 也能看到图；若显示“未生成”，按上方整改优先级补跑即可。</div></div>"
        "<footer>由 data_ops.py 自动生成 · 只读巡检 · "
        "配套课程《大数据综合实训 · 深度学习实训》</footer></div></body></html>"
        % (_CSS, _esc(meta["generated_at"]), _esc(meta["log_dir"]), _esc(meta["data_dir"]),
           _esc(meta["output_dir"]), _esc(meta["run_name_filter"]), meta["retention_days"], nav,
           svg_gauge(rep["score"]), _esc(rep["verdict"]), chips, kpi,
           "".join(dims_html),
           svg_stacked_storage(rep["runs"], cats, colors), areas_html,
           svg_acc_bars(rep["runs"]),
           _esc(src["root"] or "未找到"), "".join(src_rows), _esc(src["fingerprint_path"]),
           "".join(matrix), "".join(metrics), inf_line, custom_html,
           len(rep["issues"]), "".join(issue_rows), actions,
           "base64 内嵌" if embed else "相对路径引用", images_html)
    )


def _run_artifacts():
    """延迟取 RUN_ARTIFACTS，避免模块级循环导入。"""
    from data_ops import RUN_ARTIFACTS
    return RUN_ARTIFACTS
