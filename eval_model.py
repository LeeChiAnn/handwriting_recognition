"""eval_model.py —— 模型效果验证（全测试集），产出可下载的离线成果数据

与 --mode inference 的区别：inference 只挑前 num_show 张做「输入图片示例」，
本模块把**整个测试集**跑一遍，回答的是「这个模型到底行不行」：

    整体准确率 / macro F1 / 10×10 混淆矩阵 / 每类 P·R·F1 /
    置信度分布 / 错分样本图集 / 低置信样本清单

产物落在 outputs/eval/，**不读不写 outputs/examples 与 outputs/custom**，
所以不会改动原有推理成果。JSON 由本模块产出，HTML 由 lab_report.py 渲染
（渲染层不需要 paddle，因此 HTML 可以在任何机器上重新生成）。

依赖 paddle，请在 AI Studio 上跑：
    python main.py --mode eval --run_name demo
"""

import json
import os
import time

import numpy as np
import paddle

from data_utils import load_dataset
from html_kit import make_logger
from inference import _load_trained_model
from visualize import save_image


def _metrics_from_confusion(cm):
    """由混淆矩阵推每类与 macro 指标：行=真实，列=预测。"""
    cm = np.asarray(cm, dtype=np.int64)
    support = cm.sum(axis=1)
    tp = np.diag(cm).copy()
    fp = cm.sum(axis=0) - tp
    fn = support - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    per_class = [{"label": int(i), "support": int(support[i]), "correct": int(tp[i]),
                  "precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                  "f1": round(float(f1[i]), 4)} for i in range(cm.shape[0])]
    return per_class, {
        "accuracy": round(float(tp.sum() / max(cm.sum(), 1)), 4),
        "macro_precision": round(float(prec.mean()), 4),
        "macro_recall": round(float(rec.mean()), 4),
        "macro_f1": round(float(f1.mean()), 4),
    }


def _collect_predictions(model, loader, keep_errors, batch_log, log=None):
    """跑一遍测试集，只留统计量与少量错分原图（10000 张全留图会白占 30MB 内存）。"""
    log = log or (lambda fmt, *a: print(fmt % a if a else fmt, flush=True))
    true_all, pred_all, prob_all = [], [], []
    err_imgs, kept = [], 0
    seen = 0
    with paddle.no_grad():
        for x, y in loader:
            probs = np.exp(_log_softmax(model(x).numpy(), axis=1))
            pred = probs.argmax(axis=1)
            conf = probs.max(axis=1)
            truth = y.numpy().reshape([-1]).astype(np.int64)      # 标签须展平，见 train.py 同处注释
            true_all.append(truth)
            pred_all.append(pred)
            prob_all.append(conf)
            for i in range(len(truth)):
                if truth[i] != pred[i] and kept < keep_errors:
                    err_imgs.append((seen + i, x.numpy()[i], int(truth[i]), int(pred[i]),
                                     float(conf[i]), np.sort(probs[i])[::-1][:3].tolist()))
                    kept += 1
            seen += len(truth)
            if batch_log and seen % batch_log == 0:
                log("  已验证 %d 张…", seen)
    return (np.concatenate(true_all), np.concatenate(pred_all), np.concatenate(prob_all), err_imgs)


def _log_softmax(a, axis=1):
    """数值稳定的 log-softmax（用 numpy 做，避免逐张量调用 paddle kernel），取 exp 即概率。"""
    m = a.max(axis=axis, keepdims=True)
    e = np.exp(a - m)
    return np.log(e / e.sum(axis=axis, keepdims=True))


def _top_confusions(cm, k=6):
    """挑出“把 A 认成 B”最多的几对（跳过对角），这是学生最该看的错误模式。"""
    cm = np.asarray(cm)
    off = cm.astype(np.float64).copy()
    np.fill_diagonal(off, 0.0)
    flat = np.argsort(off, axis=None)[::-1]
    out = []
    for idx in flat:
        t, p = np.unravel_index(idx, off.shape)
        if off[t, p] <= 0:
            break
        out.append([int(t), int(p), int(cm[t, p])])
        if len(out) >= k:
            break
    return out


def run_evaluation(config, logger=None):
    """全测试集验证 → outputs/eval/{eval_results.json, errors/*.png, confusion.png}"""
    log = make_logger(logger)
    model, _run_dir, model_path = _load_trained_model(config)
    out_dir = os.path.join(os.path.abspath(config.output_dir), "eval")
    err_dir = os.path.join(out_dir, "errors")
    os.makedirs(err_dir, exist_ok=True)

    _, test_loader, record = load_dataset(config)
    keep = max(int(getattr(config, "num_show", 9) or 9) * 6, 24)   # 错分样本最多存几张图
    log("开始全测试集验证：模型=%s", os.path.abspath(model_path))
    truth, pred, conf, err_imgs = _collect_predictions(model, test_loader, keep, batch_log=2000, log=log)

    n_cls = int(max(truth.max(), pred.max())) + 1
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(truth, pred):
        cm[t, p] += 1
    per_class, overall = _metrics_from_confusion(cm)

    wrong = np.where(truth != pred)[0]
    low_conf = np.where(conf < 0.90)[0]
    low_right = [int(i) for i in low_conf if truth[i] == pred[i]][:12]
    quant = {q: round(float(np.quantile(conf, q)), 4) for q in (0.01, 0.10, 0.50, 0.90)}
    hist, _ = np.histogram(conf, bins=10, range=(0.0, 1.0))

    # 错分样本落盘：文件名带 true/pred，HTML 与人工排查都能直接用
    err_records = []
    for gi, img, t, p, c, top3 in err_imgs:
        fname = f"e{gi}_true{t}_pred{p}.png"
        save_image(img, os.path.join(err_dir, fname))
        err_records.append({"index": int(gi), "true": t, "pred": p, "prob": round(c, 4),
                            "top3": [[int(k), round(float(v), 4)] for k, v in top3],
                            "image": f"errors/{fname}"})

    summary = {
        "task": "模型效果验证（全测试集）",
        "run_name": config.run_name,
        "model_type": config.model_type,
        "model_path": os.path.abspath(model_path),
        "model_bytes": os.path.getsize(model_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"split": "test", "num_samples": int(len(truth)),
                    "record_source": record.get("source") if isinstance(record, dict) else None,
                    "transform": record.get("transform") if isinstance(record, dict) else None,
                    "data_dir_declared": config.data_dir},
        "overall": overall,
        "per_class": per_class,
        "confusion": [[int(v) for v in row] for row in cm],
        "labels": list(range(n_cls)),
        "confidence": {"quantiles": quant, "hist_10": [int(v) for v in hist],
                       "num_below_090": int(len(low_conf)),
                       "num_below_090_but_correct": len([i for i in low_right if truth[i] == pred[i]]),
                       "low_conf_correct_index": low_right},
        "errors": {"total": int(len(wrong)), "saved": len(err_records), "items": err_records,
                   "top_pairs": _top_confusions(cm)},
        "note": "本文件由 eval_model.py 产出；HTML 渲染见 lab_report.py（不需要 paddle）",
    }
    json_path = os.path.join(out_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    png = _plot_confusion(cm, os.path.join(out_dir, "confusion.png"))

    log("验证完成：准确率 %.4f｜macro F1 %.4f｜错分 %d/%d｜置信度<0.90 共 %d 张",
        overall["accuracy"], overall["macro_f1"], len(wrong), len(truth), len(low_conf))
    log("==== 本次产物保存位置（绝对路径）====")
    log("  结果JSON     : %s", os.path.abspath(json_path))
    log("  错分样本图集 : %s（%d 张）", os.path.abspath(err_dir), len(err_records))
    if png:
        log("  混淆矩阵图   : %s", os.path.abspath(png))
    log("  下一步（不需 paddle）：python main.py --mode report --run_name %s", config.run_name)
    return summary, out_dir


def _plot_confusion(cm, save_path):
    """混淆矩阵 PNG（图内全英文：AI Studio 镜像无中文字体）。缺 matplotlib 就跳过。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    thresh = max(cm.max(), 1) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "#33383f")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path
