"""visualize.py —— 训练曲线 & 推理结果可视化

依赖 matplotlib（AI Studio 已预装）。若环境缺失则跳过并给出提示，不阻塞训练。
"""


import os

import numpy as np

# 运维五维的英文名（与 data_ops.DIMENSIONS 的 key 一一对应）
_DIMS_EN = [
    ("completeness", "Completeness"),
    ("consistency", "Consistency"),
    ("traceability", "Traceability"),
    ("source_health", "Source Health"),
    ("storage_gov", "Storage Gov."),
]


def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无界面环境也能画图
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # pragma: no cover
        print(f"[warn] 未安装 matplotlib，跳过绘图：{e}")
        return None


def plot_history(history, save_path):
    """画损失曲线 + 准确率曲线，保存到 save_path。"""
    plt = _get_plt()
    if plt is None:
        return None
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], "o-", label="train loss")
    axes[0].plot(epochs, history["val_loss"], "s-", label="val loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs, history["train_acc"], "o-", label="train acc")
    axes[1].plot(epochs, history["val_acc"], "s-", label="val acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_predictions(images, truths, preds, save_path, ncols=3):
    """把若干样本画成网格：标题显示 pred / true，预测错显红。"""
    plt = _get_plt()
    if plt is None:
        return None
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, (img, t, p) in enumerate(zip(images, truths, preds)):
        arr = img[0] if (hasattr(img, "ndim") and img.ndim == 3) else img
        arr = np.clip(arr * 0.3081 + 0.1307, 0, 1)  # 反标准化还原成 [0,1]
        ax = axes[i]
        ax.imshow(arr, cmap="gray")
        ax.axis("off")
        color = "green" if t == p else "red"
        ax.set_title(f"pred={p}\ntrue={t}", color=color, fontsize=10)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def save_image(img, path):
    """把单张 [1,28,28] 归一化张量保存为灰度 PNG（优先用 PIL，缺失则退回 matplotlib）。"""
    arr = img[0] if (hasattr(img, "ndim") and img.ndim == 3) else img
    arr = np.clip(arr * 0.3081 + 0.1307, 0, 1)
    arr = (arr * 255).astype("uint8")
    try:
        from PIL import Image
        Image.fromarray(arr, "L").save(path)
    except Exception:
        plt = _get_plt()
        if plt is not None:
            plt.imsave(path, arr, cmap="gray")


def plot_custom_predictions(images, preds, probs, truths, save_path, ncols=4):
    """用户自提图片的识别拼图：格子里是模型真正看到的 28×28（预处理后）。

    truths 里可以是 None（文件名没写标准答案），此时标题只显 pred/prob 并用中性色，
    不假装“对/错”——图内文字全英文（AI Studio 镜像无中文字体）。
    """
    plt = _get_plt()
    if plt is None:
        return None
    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 3.1 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, (img, p, q, t) in enumerate(zip(images, preds, probs, truths)):
        arr = img[0] if (hasattr(img, "ndim") and img.ndim == 3) else img
        axes[i].imshow(np.clip(arr * 0.3081 + 0.1307, 0, 1), cmap="gray")
        axes[i].axis("off")
        if t is None:
            head, color = f"pred={p}  p={q:.3f}", "#4b5568"
        else:
            color = "green" if t == p else "red"
            head = f"pred={p}  p={q:.3f}\ntrue={t}  {'OK' if t == p else 'BAD'}"
        axes[i].set_title(head, color=color, fontsize=9)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("User-provided images (28x28 after preprocessing)  |  n=%d" % n, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


# =============================================================== 运维看板
# 图内文本全部用英文：AI Studio 镜像没装中文字体，中文会渲染成豆腐块。
# 需要中文成果时请看 outputs/ops/dashboard.html 与 ops_report.md。

_OPS_ARTIFACT_LABELS = [
    ("model", "model"), ("best_model", "best"), ("config", "config"),
    ("history", "history"), ("curves", "curves"), ("record", "data_rec"), ("log", "log"),
]


def plot_ops_dashboard(report, save_path):
    """画四联数据运维看板：存储占用 / 资产完整性 / 指标对比 / 五维健康度。"""
    plt = _get_plt()
    if plt is None:
        return None

    runs = report.get("runs", [])
    names = [r["name"][:16] for r in runs]
    cats = ["weights", "logs", "images", "json", "other"]
    cat_colors = {"weights": "#6366f1", "logs": "#10b981", "images": "#f59e0b",
                  "json": "#38bdf8", "other": "#94a3b8"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()
    s = report["summary"]
    fig.suptitle(
        "DataOps Audit  |  health %.0f/100 (grade %s)  |  runs=%d  ERROR=%d  WARN=%d  INFO=%d  |  %s"
        % (report["score"], report["grade"], s["runs"], s["error"], s["warn"], s["info"],
           report["meta"]["generated_at"]), fontsize=12)

    if not runs:
        for a in (ax1, ax2, ax3, ax4):
            a.axis("off")
        ax1.text(0.5, 0.5, "NO RUN FOUND under logs/\nrun `python main.py --mode train --run_name demo` first",
                 ha="center", va="center", fontsize=13, color="#c62f38")
        dims = report["dimensions"]
    else:
        # ---------- ① 存储占用（按 run 堆叠，单位自适应 KB/MB）
        total_max = max([r["total_bytes"] for r in runs] + [1])
        unit, scale = ("KB", 1024.0) if total_max < 5 << 20 else ("MB", 1e6)
        ypos = np.arange(len(runs))[::-1]
        left = np.zeros(len(runs))
        for c in cats:
            vals = np.array([r["by_cat"][c] / scale for r in runs])
            ax1.barh(ypos, vals, left=left, height=0.6, color=cat_colors[c], label=c)
            left += vals
        for i, (y, r) in enumerate(zip(ypos, runs)):
            ax1.text(left[i] + max(left.max() * 0.012, 0.2), y, r["human_size"],
                     va="center", fontsize=8)
        ax1.set_yticks(ypos)
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel("Size per run (%s)" % unit)
        ax1.set_title("1. Storage footprint by file category")
        ax1.legend(fontsize=8, ncol=5, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, -0.22))
        ax1.grid(True, axis="x", alpha=0.3)

        # ---------- ② 资产完整性矩阵
        mat = np.array([[1 if r["artifacts"].get(k) else 0 for k, _ in _OPS_ARTIFACT_LABELS]
                        for r in runs])
        ax2.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax2.set_xticks(range(len(_OPS_ARTIFACT_LABELS)))
        ax2.set_xticklabels([lab for _, lab in _OPS_ARTIFACT_LABELS], fontsize=8, rotation=30, ha="right")
        ax2.set_yticks(range(len(runs)))
        ax2.set_yticklabels(names, fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax2.text(j, i, "OK" if mat[i, j] else "MISS", ha="center", va="center",
                         fontsize=7, color="#111")
        ax2.set_title("2. Artifact completeness matrix (traceability)")

        # ---------- ③ 指标对比 + 泛化间隙
        x = np.arange(len(runs))
        tr = np.array([(r.get("final") or {}).get("train_acc", 0) for r in runs])
        va = np.array([(r.get("final") or {}).get("val_acc", 0) for r in runs])
        ax3.bar(x - 0.19, tr, width=0.38, color="#6366f1", label="train_acc")
        ax3.bar(x + 0.19, va, width=0.38, color="#12a06b", label="val_acc")
        for i, r in enumerate(runs):
            f = r.get("final") or {}
            if not f:
                ax3.text(i, 0.55, "no\nhistory", ha="center", fontsize=7, color="#c62f38")
                continue
            ax3.annotate("gap %+.3f" % f["gap"], (i, max(f["train_acc"], f["val_acc"]) + 0.015),
                         ha="center", fontsize=7,
                         color="#c62f38" if f["gap"] > 0.05 else "#4b5568")
        ax3.axhline(0.95, ls="--", lw=1, color="#f5a623")
        # 用轴坐标放文字，否则单 run 时会跑到绘图区外面
        ax3.text(0.985, 0.955, "target 0.95", transform=ax3.get_xaxis_transform(),
                 ha="right", va="bottom", fontsize=7, color="#b47109")
        ax3.set_xticks(x)
        ax3.set_xticklabels(["%s\n%s" % (n, (r.get("final") or {}).get("model_type", "?"))
                             for n, r in zip(names, runs)], fontsize=8)
        ax3.set_ylim(0, 1.12)
        ax3.set_ylabel("Accuracy")
        ax3.set_title("3. Final accuracy by run (gap = overfitting signal)")
        ax3.legend(fontsize=8, loc="upper left")
        ax3.grid(True, axis="y", alpha=0.3)

        # ---------- ④ 五维健康度雷达
        dims = report["dimensions"]

    labels = [lab for _, lab in _DIMS_EN]
    scores = [dims[k] for k, _ in _DIMS_EN]
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores_c = scores + scores[:1]
    ang_c = ang + ang[:1]
    ax4.remove()
    ax4 = fig.add_subplot(2, 2, 4, polar=True)
    ax4.plot(ang_c, scores_c, "o-", color="#6366f1", lw=2)
    ax4.fill(ang_c, scores_c, color="#6366f1", alpha=0.18)
    ax4.set_xticks(ang)
    ax4.set_xticklabels(labels, fontsize=8)
    ax4.set_ylim(0, 100)
    ax4.set_yticks([25, 50, 75, 100])
    ax4.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color="#9aa3b2")
    ax4.set_title("4. Five-dimension health radar\n(weighted score = %.0f, grade %s)"
                  % (report["score"], report["grade"]), fontsize=11, pad=18)

    # 极坐标子图与 tight_layout 不兼容（会报 Tight layout not applied），直接手动留白
    fig.subplots_adjust(left=0.09, right=0.995, top=0.90, bottom=0.13, wspace=0.42, hspace=0.52)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path
