"""html_kit.py —— 离线成果包（单文件 HTML）的共享渲染基建

为什么要有这个模块：课程要求「学生把 HTML 下载到本地电脑就能看」，
所以每一份成果 HTML 都必须满足三条硬约束——

1. **自包含**：CSS 内联、图片走 base64 data URI、图表用内嵌 SVG，
   不允许出现任何外链（CDN、字体、JS 库一概不用）。断网也能打开。
2. **零依赖**：只用 Python 标准库，不 import 项目内其他模块，
   这样在没装 numpy/matplotlib/paddle 的机器上照样能出成果。
3. **中文随便写**：AI Studio 镜像缺中文字体，PNG 里写中文会成豆腐块；
   HTML 走浏览器渲染，所以 **PNG 全英文、HTML 全中文**。

本模块提供：页面骨架、样式、图片内嵌，以及四种纯 SVG 图表
（折线 / 热力矩阵 / 直方图 / 环形占比）。表格与卡片等构件也放这里，
避免 ops_dashboard.py 与 lab_report.py 各写一份。
"""

import base64
import html as _html
import math
import os
import struct

# 调色板：与 visualize.py 的 PNG 看板保持同一色系，两份成果看起来像一套东西
INDIGO = "#6366f1"
GREEN = "#12a06b"
RED = "#e5484d"
AMBER = "#f5a623"
BLUE = "#38bdf8"
GRAY = "#94a3b8"
SERIES_COLORS = [INDIGO, GREEN, AMBER, BLUE, "#a855f7", "#10b981"]


def esc(text):
    """所有动态文本进 HTML 前都要过这里，避免图路径/日志里的尖括号破坏结构。"""
    return _html.escape(str(text), quote=True)


def fmt_bytes(n):
    """字节数转人话（本模块自带一份，保证不依赖 data_ops）。"""
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.0f %s" % (n, unit) if unit == "B" else "%.1f %s" % (n, unit)
        n /= 1024
    return "%.1f GB" % n


def score_color(score):
    if score >= 90:
        return GREEN
    if score >= 75:
        return "#3b82f6"
    if score >= 60:
        return AMBER
    return RED


CSS = """
*{box-sizing:border-box}body{margin:0;padding:24px;background:#f5f7fa;\
font-family:-apple-system,"PingFang SC","Microsoft YaHei",Helvetica,Arial,sans-serif;color:#1a1d24}\
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:24px 0 10px;color:#2c3140;\
border-left:4px solid #6366f1;padding-left:9px}h3{font-size:13px;margin:16px 0 8px;color:#3a4152}\
.sub{color:#68707f;font-size:12px;margin-bottom:18px;line-height:1.6}\
.wrap{max-width:1120px;margin:0 auto}.card{background:#fff;border:1px solid #e6eaf0;\
border-radius:12px;padding:16px 18px;margin-bottom:14px;box-shadow:0 1px 2px rgba(16,24,40,.04);\
overflow-x:auto}\
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}\
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));gap:10px;margin-top:10px}\
.kpi{background:#f8fafc;border:1px solid #e6eaf0;border-radius:10px;padding:9px 11px}\
.kpi b{display:block;font-size:20px;margin-top:2px}.kpi span{font-size:11px;color:#68707f}\
.chip{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600;color:#fff}\
table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:7px 8px;\
border-bottom:1px solid #eef1f5;text-align:left;vertical-align:middle}th{background:#f8fafc;\
color:#4b5568;font-weight:600;white-space:nowrap}td.num{text-align:right;font-variant-numeric:tabular-nums}\
code{background:#f2f4f8;padding:1px 5px;border-radius:4px;font-size:11px;word-break:break-all}\
.muted{color:#9aa3b2}.ok{color:#12805c}.bad{color:#c62f38}.warned{color:#b47109}\
.cell-ok{background:#e7f6ef}.cell-bad{background:#fdeceb}\
.dimrow{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:12px}\
.dimrow span{width:80px;color:#4b5568}.dimrow b{width:34px;text-align:right}\
.track{flex:1;height:9px;background:#eef1f5;border-radius:6px;overflow:hidden}\
.fill{height:100%;border-radius:6px}footer{color:#8d95a3;font-size:11px;text-align:center;margin:18px 0}\
.shots{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}\
.shot{border:1px solid #e6eaf0;border-radius:10px;padding:8px;background:#fcfdff;text-align:center}\
.shot img{max-width:100%;border-radius:6px;background:#fff}\
.shot .cap{font-size:11px;color:#68707f;margin-top:5px;word-break:break-all}\
.px img{image-rendering:pixelated;image-rendering:-moz-crisp-edges;width:84px;height:84px;object-fit:contain}\
.miss{border:1px dashed #d6dbe4;border-radius:10px;padding:14px;color:#9aa3b2;font-size:12px}\
details{margin:8px 0}summary{cursor:pointer;font-size:12px;color:#4b5568;\
background:#f8fafc;border:1px solid #e6eaf0;border-radius:8px;padding:7px 10px}\
details[open] summary{border-radius:8px 8px 0 0}pre{background:#0f1420;color:#d8e0ee;\
padding:12px;border-radius:0 0 8px 8px;font-size:11px;line-height:1.55;overflow:auto;margin:0}\
a.btn{display:inline-block;background:#eef0ff;color:#3f49b5;border:1px solid #dfe3ff;\
border-radius:8px;padding:7px 12px;font-size:12px;text-decoration:none;margin:4px 6px 0 0}\
a.btn:hover{background:#e2e6ff}
"""


def make_logger(logger):
    """统一“有 logger 走 logging，没 logger 走 print”两种口径。
    直接拿 print 当 logger.info 用会让 %s 占位符原样打出来，这里统一掉。"""
    if logger is not None:
        return logger.info

    def _log(fmt, *args):
        print(fmt % args if args else fmt)

    return _log


def page(title, h1, sub, body, footer="", back=True):
    """整页骨架。back=True 时顶部放一个回 index 的按钮（只在同目录时点得开）。"""
    nav = '<a class="btn" href="index.html">← 返回成果包入口</a>' if back else ""
    return (
        '<!doctype html><html lang="zh"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>%s</title><style>%s</style></head><body><div class=\"wrap\">"
        "<h1>%s</h1>%s<div class=\"sub\">%s</div>%s<footer>%s</footer></div></body></html>"
        % (esc(title), CSS, esc(h1), nav, sub, body, esc(footer))
    )


def kpis(items):
    """items: [(标签, 值, css类或颜色)]，第三项可省略。"""
    out = []
    for it in items:
        label, val = it[0], it[1]
        style = ""
        if len(it) > 2 and it[2]:
            style = ' style="color:%s"' % (score_color(val) if it[2] == "score" else it[2])
        out.append('<div class="kpi"><span>%s</span><b%s>%s</b></div>'
                   % (esc(label), style, esc(val)))
    return '<div class="kpis">%s</div>' % "".join(out)


def chip(text, color):
    return '<span class="chip" style="background:%s">%s</span>' % (color, esc(text))


def bar_row(label, frac, color=INDIGO, right="", label_w=170):
    """一条横向占比条：五维得分、分区域存储账都用它；label_w=0 可放进表格单元格里。"""
    tail = '<b style="width:70px">%s</b>' % esc(right) if right != "" else ""
    return ('<div class="dimrow"><span style="width:%dpx">%s</span><div class="track">'
            '<div class="fill" style="width:%.1f%%;background:%s"></div></div>%s</div>'
            % (label_w, esc(label), max(0.0, min(100.0, frac * 100.0)), color, tail))


def table(headers, rows, empty="无数据", num_cols=()):
    """headers 里以 '<' 开头的视为已拼好的 HTML；num_cols 是列下标集合，该列右对齐等宽数字。"""
    th = "".join("<th>%s</th>" % (h if str(h).startswith("<") else esc(h)) for h in headers)
    if not rows:
        return '<table><tr>%s</tr><tr><td colspan="%d" class="muted">%s</td></tr></table>' % (
            th, max(1, len(headers)), esc(empty))
    body = []
    for r in rows:
        tds = []
        for j, c in enumerate(r):
            cls = ' class="num"' if j in num_cols else ""
            tds.append("<td%s>%s</td>" % (cls, c if str(c).startswith("<") else esc(c)))
        body.append("<tr>%s</tr>" % "".join(tds))
    return "<table><tr>%s</tr>%s</table>" % (th, "".join(body))


def details(summary, inner, open_=False):
    return "<details%s><summary>%s</summary>%s</details>" % (
        " open" if open_ else "", esc(summary), inner)


# ---------------------------------------------------------------- 图片内嵌

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml"}


def image_uri(path):
    """把图片读成 data URI；读不到返回 None（调用方负责显示占位）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _MIME or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    if not raw:
        return None
    return "data:%s;base64,%s" % (_MIME[ext], base64.b64encode(raw).decode("ascii"))


def image_tag(path, embed=True, pixelated=False, alt=None, base_dir=None):
    """一张图 → <img>。embed=True 走 base64（下载单文件也能看）；
    embed=False 走相对路径（省体积，但必须整目录一起拷走，跨目录时用 base_dir 算相对）。"""
    if embed:
        uri = image_uri(path)
        if uri is None:
            return None
        src = uri
    else:
        if not os.path.isfile(path):
            return None
        src = os.path.relpath(path, base_dir) if base_dir else os.path.basename(path)
    return '<img src="%s" alt="%s"%s>' % (esc(src), esc(alt or os.path.basename(path)),
                                          ' loading="lazy"' if not embed else "")


def shot(path, caption, embed=True, pixelated=False, base_dir=None):
    """带说明文字的图片卡片；图缺失时明确写「未生成」，不留空位让人猜。"""
    tag = image_tag(path, embed=embed, pixelated=pixelated, base_dir=base_dir)
    if tag is None:
        return ('<div class="shot"><div class="miss">未生成<br>%s</div>'
                '<div class="cap">%s</div></div>'
                % (esc(os.path.basename(str(path))), esc(caption)))
    return '<div class="shot%s">%s<div class="cap">%s</div></div>' % (
        " px" if pixelated else "", tag, esc(caption))


def shots(items, embed=True, base_dir=None):
    """items: [(path, caption[, pixelated])] → 自适应网格。第三项缺省为 False。"""
    out = []
    for it in items:
        path, cap = it[0], it[1]
        px = bool(it[2]) if len(it) > 2 else False
        out.append(shot(path, cap, embed=embed, pixelated=px, base_dir=base_dir))
    return '<div class="shots">%s</div>' % "".join(out)


# ---------------------------------------------------------------- SVG 图表

def svg_line(series, x_labels=None, width=520, height=200, y_fmt="%.3f", title=""):
    """多序列折线。series: [(名称, [数值...], 颜色)]；None 值断开，用于缺轮次。
    训练曲线（loss / acc）直接由 history.json 重绘，不依赖 curves.png 是否存在。"""
    series = [(n, [v for v in vals], c) for n, vals, c in series if vals]
    if not series:
        return '<p class="muted">没有可绘的序列（缺少 history.json）。</p>'
    pad_l, pad_r, pad_t, pad_b = 46, 12, 14, 30
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    n_max = max(len(v) for _, v, _ in series)
    allv = [v for _, vals, _ in series for v in vals if isinstance(v, (int, float))]
    lo, hi = min(allv), max(allv)
    if hi - lo < 1e-9:
        hi = lo + 1e-3
    lo -= (hi - lo) * 0.08
    hi += (hi - lo) * 0.08

    def X(i):
        return pad_l + (plot_w * (i / (n_max - 1) if n_max > 1 else 0.5))

    def Y(v):
        return pad_t + plot_h - (float(v) - lo) / (hi - lo) * plot_h

    p = ['<svg viewBox="0 0 %d %d" width="100%%" height="%d" role="img">' % (width, height, height)]
    if title:
        p.append('<text x="%d" y="11" font-size="10" fill="#68707f">%s</text>' % (pad_l, esc(title)))
    for k in range(5):
        gv = lo + (hi - lo) * k / 4
        yy = Y(gv)
        p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#eef1f5"/>'
                 '<text x="%d" y="%.1f" font-size="9" fill="#9aa3b2" text-anchor="end">%s</text>'
                 % (pad_l, yy, width - pad_r, yy, pad_l - 5, yy + 3, y_fmt % gv))
    for si, (name, vals, col) in enumerate(series):
        pts = " ".join("%.1f,%.1f" % (X(i), Y(v)) for i, v in enumerate(vals)
                       if isinstance(v, (int, float)))
        p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2" '
                 'stroke-linejoin="round"/>' % (pts, col))
        for i, v in enumerate(vals):
            if isinstance(v, (int, float)):
                p.append('<circle cx="%.1f" cy="%.1f" r="2.4" fill="%s">'
                         '<title>%s 第%d轮 = %s</title></circle>'
                         % (X(i), Y(v), col, esc(name), i + 1, ("%s" % y_fmt) % v))
    if x_labels:
        step = max(1, n_max // 8)
        for i in range(0, n_max, step):
            lab = x_labels[i] if i < len(x_labels) else i + 1
            p.append('<text x="%.1f" y="%d" font-size="9" fill="#9aa3b2" '
                     'text-anchor="middle">%s</text>' % (X(i), height - 16, esc(lab)))
    lx = pad_l
    for name, _, col in series:
        p.append('<rect x="%d" y="%d" width="10" height="10" rx="2" fill="%s"/>'
                 '<text x="%d" y="%d" font-size="10" fill="#666">%s</text>'
                 % (lx, height - 11, col, lx + 14, height - 2, esc(name)))
        lx += 14 + 6 * len(str(name)) + 16
    p.append("</svg>")
    return "".join(p)


def svg_heat(matrix, row_labels, col_labels, width=None, color_scale="blue"):
    """热力矩阵（混淆矩阵专用）。行=真实标签，列=预测标签；数字直接写在格子里。"""
    n, m = len(matrix), len(matrix[0]) if matrix else 0
    if not n or not m:
        return '<p class="muted">无矩阵数据。</p>'
    cell = 46
    left, top = 62, 30
    width = width or (left + cell * m + 16)
    height = top + cell * n + 26
    flat = [v for row in matrix for v in row]
    vmax = max(flat) or 1
    p = ['<svg viewBox="0 0 %d %d" width="100%%" height="%d">' % (width, height, height)]
    for j in range(m):
        p.append('<text x="%.1f" y="%d" font-size="10" fill="#4b5568" '
                 'text-anchor="middle">%s</text>' % (left + cell * (j + .5), top - 8, esc(col_labels[j])))
    for i in range(n):
        p.append('<text x="%d" y="%.1f" font-size="10" fill="#4b5568" '
                 'text-anchor="end">%s</text>' % (left - 8, top + cell * (i + .5) + 4, esc(row_labels[i])))
        for j in range(m):
            v = matrix[i][j]
            t = v / vmax
            if color_scale == "blue":
                fill = "rgba(99,102,241,%.3f)" % (0.05 + 0.85 * t) if v else "#f7f8fb"
            else:                                       # 对角绿 / 非对角红用
                fill = "rgba(18,160,107,%.3f)" % (0.1 + 0.8 * t) if v else "#f7f8fb"
            fg = "#fff" if t > 0.55 else "#33383f"
            p.append('<rect x="%.1f" y="%.1f" width="%d" height="%d" rx="4" '
                     'stroke="#fff" stroke-width="1.5" fill="%s"><title>%s → %s : %s</title></rect>'
                     % (left + cell * j, top + cell * i, cell, cell, fill,
                        esc(row_labels[i]), esc(col_labels[j]), fmt_int(v)))
            p.append('<text x="%.1f" y="%.1f" font-size="11" fill="%s" '
                     'text-anchor="middle" font-weight="600">%s</text>'
                     % (left + cell * (j + .5), top + cell * (i + .5) + 4, fg, fmt_int(v)))
    p.append('<text x="%d" y="%d" font-size="10" fill="#9aa3b2">行 = 真实标签　列 = 模型预测</text>'
             % (left, height - 8))
    p.append("</svg>")
    return "".join(p)


def svg_hist(values, bins=10, lo=0.0, hi=1.0, width=520, height=150, color=INDIGO,
             xlabel="区间", ylabel="数量"):
    """直方图（置信度分布专用）。values 需已落在 [lo,hi]。"""
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    if not vals:
        return '<p class="muted">无数据。</p>'
    counts = [0] * bins
    for v in vals:
        k = int((min(max(v, lo), hi) - lo) / (hi - lo) * bins)
        counts[min(k, bins - 1)] += 1
    cmax = max(counts) or 1
    pad_l, pad_b, pad_t = 40, 26, 12
    pw, ph = width - pad_l - 12, height - pad_b - pad_t
    bw = pw / bins
    p = ['<svg viewBox="0 0 %d %d" width="100%%" height="%d">' % (width, height, height)]
    for i, c in enumerate(counts):
        h = ph * c / cmax
        x0, y0 = pad_l + bw * i, pad_t + ph - h
        p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="2" fill="%s">'
                 '<title>%s ~ %s：%d 个</title></rect>'
                 % (x0 + 1, y0, bw - 2, max(h, 0.5), color,
                    "%.2f" % (lo + (hi - lo) * i / bins), "%.2f" % (lo + (hi - lo) * (i + 1) / bins), c))
        if c:
            p.append('<text x="%.1f" y="%.1f" font-size="9" fill="#4b5568" '
                     'text-anchor="middle">%d</text>' % (x0 + bw / 2, y0 - 3, c))
    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#eef1f5"/>'
             % (pad_l, pad_t + ph, width - 12, pad_t + ph))
    for k in (0, 0.5, 1.0):
        p.append('<text x="%d" y="%.1f" font-size="9" fill="#9aa3b2" text-anchor="end">%s</text>'
                 % (pad_l - 5, pad_t + ph - ph * k + 3, fmt_int(cmax * k)))
        p.append('<text x="%.1f" y="%d" font-size="9" fill="#9aa3b2" text-anchor="middle">%s</text>'
                 % (pad_l + pw * k, height - 12, "%.2f" % (lo + (hi - lo) * k)))
    p.append('<text x="%d" y="%d" font-size="10" fill="#666">%s</text>' % (pad_l, height - 4, esc(xlabel)))
    p.append("</svg>")
    return "".join(p)


def svg_bars(counts, labels=None, width=520, height=150, color=INDIGO, xlabel="", ylabel="数量"):
    """简单柱状图，用于画**已经分好箱**的数据（如置信度直方图、每类样本数）。
    与 svg_hist 的区别：后者吃原始数值自己分箱，前者吃现成的计数列。"""
    vals = [float(v) if isinstance(v, (int, float)) else 0.0 for v in counts]
    if not vals:
        return '<p class="muted">无数据。</p>'
    cmax = max(vals) or 1
    pad_l, pad_b, pad_t = 40, 26, 12
    pw, ph = width - pad_l - 12, height - pad_b - pad_t
    bw = pw / len(vals)
    p = ['<svg viewBox="0 0 %d %d" width="100%%" height="%d">' % (width, height, height)]
    for i, c in enumerate(vals):
        h = ph * c / cmax
        x0, y0 = pad_l + bw * i, pad_t + ph - h
        lab = labels[i] if labels and i < len(labels) else i
        p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="2" fill="%s">'
                 '<title>%s：%s</title></rect>'
                 % (x0 + 1, y0, max(bw - 2, 1), max(h, 0.5), color, esc(lab), fmt_int(c)))
        if c:
            p.append('<text x="%.1f" y="%.1f" font-size="9" fill="#4b5568" '
                     'text-anchor="middle">%s</text>' % (x0 + bw / 2, y0 - 3, fmt_int(c)))
        p.append('<text x="%.1f" y="%d" font-size="9" fill="#9aa3b2" '
                 'text-anchor="middle">%s</text>' % (x0 + bw / 2, height - 12, esc(lab)))
    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#eef1f5"/>'
             % (pad_l, pad_t + ph, width - 12, pad_t + ph))
    p.append('<text x="4" y="%d" font-size="10" fill="#666">%s</text>' % (height - 4, esc(ylabel)))
    if xlabel:
        p.append('<text x="%d" y="%d" font-size="10" fill="#666">%s</text>'
                 % (pad_l, 11, esc(xlabel)))
    p.append("</svg>")
    return "".join(p)


def svg_donut(frac, label, sub="", size=150, color=INDIGO):
    """环形占比：整体准确率、每类召回率都用它。"""
    r = size / 2.0 - 14
    cx = cy = size / 2.0
    circ = 2 * math.pi * r
    frac = max(0.0, min(1.0, frac))
    return ('<svg viewBox="0 0 %d %d" width="%d" height="%d">'
            '<circle cx="%.1f" cy="%.1f" r="%.1f" fill="none" stroke="#e9edf3" stroke-width="13"/>'
            '<circle cx="%.1f" cy="%.1f" r="%.1f" fill="none" stroke="%s" stroke-width="13" '
            'stroke-linecap="round" stroke-dasharray="%.1f %.1f" '
            'transform="rotate(-90 %.1f %.1f)"/>'
            '<text x="%.1f" y="%.1f" text-anchor="middle" font-size="21" font-weight="700" '
            'fill="#1a1d24">%s</text>'
            '<text x="%.1f" y="%.1f" text-anchor="middle" font-size="10" fill="#68707f">%s</text>'
            "</svg>"
            % (size, size, size, size, cx, cy, r, cx, cy, r, color, circ * frac, circ, cx, cy,
               cx, cy + 4, "%.1f%%" % (frac * 100), cx, cy + 20, esc(sub or label)))


def fmt_int(v):
    """矩阵里的数字：整数直接显示，浮点保留一位。"""
    if isinstance(v, float) and not float(v).is_integer():
        return "%.1f" % v
    return "%d" % int(v)


def png_is_small_square(path, max_edge=64):
    """不依赖 PIL，直读 PNG 头判断是不是一张小正方形图（如 28×28 手写数字）。

    只有这种图才适合放大成固定 84×84 的像素缩略图；拿体积大小当判据会把
    all_digits.png（316×28 的长条拼图）也误判，导致长宽比被压扁。"""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
    except OSError:
        return False
    if len(head) < 24 or not head.startswith(b"\x89PNG\r\n\x1a\n") or head[12:16] != b"IHDR":
        return False
    w, h = struct.unpack(">II", head[16:24])
    return w == h and w <= max_edge
