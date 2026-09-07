"""inference.py —— 推理：输入图片示例 + 识别结果

两条通路，产物目录严格分开：

1. `run_inference()`（--mode inference，原有行为不变）
   从 MNIST **测试集**里取前 num_show 张 → `outputs/examples/`
2. `run_custom_inference()`（--mode custom，新增）
   读用户自己放的 **任意尺寸 PNG**（`--image_dir`，默认 ./inputs）→ `outputs/custom/`
   预处理见 custom_input.py；两边产物互不覆盖，交作业时各取所需。
"""


import json
import os
import time

import numpy as np
import paddle
import paddle.nn.functional as F

from custom_input import iter_user_images, label_from_filename, preprocess_png
from data_utils import load_dataset
from model import build_model
from visualize import plot_custom_predictions, plot_predictions, save_image


def _load_trained_model(config):
    """按 run 目录里的参数快照重建同结构模型并载入权重，返回 (model, run_dir, model_path)。"""
    paddle.set_device("cpu")
    run_dir = os.path.join(config.log_dir, config.run_name)
    model_path = os.path.join(run_dir, "best_model.pdparams")
    if not os.path.exists(model_path):
        model_path = os.path.join(run_dir, "model.pdparams")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"未找到模型权重，请先训练：python main.py --mode train --run_name {config.run_name}"
        )

    # 用训练时保存的 config 重建"相同结构"的模型，避免结构对不上
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        saved = json.load(open(cfg_path, encoding="utf-8"))
        config.model_type = saved.get("model_type", config.model_type)
        config.hidden_size = saved.get("hidden_size", config.hidden_size)
        config.dropout = saved.get("dropout", config.dropout)

    model = build_model(config)
    model.set_state_dict(paddle.load(model_path))
    model.eval()
    return model, run_dir, model_path


def run_inference(config):
    model, run_dir, model_path = _load_trained_model(config)

    _, test_loader, _ = load_dataset(config)
    examples, count = [], 0
    for x, y in test_loader:
        for i in range(x.shape[0]):
            if count >= config.num_show:
                break
            img = x[i]
            label = int(y[i].item())                                                # 0-d -> .item()
            with paddle.no_grad():
                pred = int(model(img.unsqueeze(0)).argmax(axis=1).item())            # [1] -> .item()
            examples.append((img.numpy(), label, pred))
            count += 1
        if count >= config.num_show:
            break

    # 保存每张输入图 + 结果汇总
    out_dir = os.path.join(config.output_dir, "examples")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for idx, (img, label, pred) in enumerate(examples):
        save_image(img, os.path.join(out_dir, f"sample_{idx+1}_true{label}_pred{pred}.png"))
        results.append({"index": idx + 1, "true": label, "pred": pred, "correct": label == pred})
    summary = {
        "run_name": config.run_name,
        "num_examples": len(results),
        "accuracy_on_examples": round(sum(r["correct"] for r in results) / len(results), 4),
        "results": results,
    }
    with open(os.path.join(out_dir, "inference_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 生成综合拼图（绿=对，红=错）
    plot_predictions(
        [e[0] for e in examples], [e[1] for e in examples], [e[2] for e in examples],
        os.path.join(out_dir, "predictions.png"),
    )
    print(f"[推理完成] 示例数={len(results)} 示例准确率={summary['accuracy_on_examples']}")
    print(f"==== 本次产物保存位置（绝对路径）====")
    print(f"  输入图片目录 : {os.path.abspath(out_dir)}")
    for idx in range(len(results)):
        fname = f"sample_{idx+1}_true{results[idx]['true']}_pred{results[idx]['pred']}.png"
        print(f"  输入图片 {idx+1:02d}  : {os.path.abspath(os.path.join(out_dir, fname))}")
    print(f"  结果拼图     : {os.path.abspath(os.path.join(out_dir, 'predictions.png'))}")
    print(f"  结果JSON     : {os.path.abspath(os.path.join(out_dir, 'inference_results.json'))}")
    return summary, out_dir


# =============================================================== 用户自提图片推理

def run_custom_inference(config):
    """识别 config.image_dir 里用户自己的 PNG，产物全部写 outputs/custom/。

    与 run_inference() 的 outputs/examples/ 完全隔离：不读不写对方目录，
    因此跑这个模式不会改动原有的输入图片示例与 inference_results.json。
    """
    model, _, model_path = _load_trained_model(config)

    in_dir = os.path.abspath(config.image_dir)
    files, ignored = iter_user_images(in_dir)
    print(f"[自定义推理] 模型={os.path.abspath(model_path)}  输入目录={in_dir}")
    if not files:
        print("  该目录下没有 PNG（仅支持 .png）。把“一张图一个数字”的 PNG 放进去后重试：")
        print(f"  python main.py --mode custom --run_name {config.run_name} --image_dir {config.image_dir}")
        print(f"  想先确认图能不能用（不需要 paddle）：python custom_input.py {config.image_dir}")
        return None, None
    if ignored:
        print(f"  [忽略] 是图片但不是 PNG 的文件 {len(ignored)} 个（请转成 .png）："
              f"{', '.join(ignored[:5])}{' ...' if len(ignored) > 5 else ''}")

    out_dir = os.path.join(config.output_dir, "custom")
    os.makedirs(out_dir, exist_ok=True)

    items, shown = [], []
    for i, path in enumerate(files):
        name = os.path.basename(path)
        true = label_from_filename(path)                     # 文件名开头是数字 => 当作标准答案
        try:
            prep = preprocess_png(path)                      # 任意尺寸 → [1,28,28]
        except Exception as e:
            print(f"  [{i + 1}/{len(files)}] [跳过] {name}：{e}")
            items.append({"index": i + 1, "file": name, "error": str(e)})
            continue

        x = prep["x"]
        with paddle.no_grad():
            probs = F.softmax(model(paddle.to_tensor(x)), axis=1)[0].numpy()
        pred, conf = int(probs.argmax()), float(probs.max())
        top3 = [[int(k), round(float(probs[k]), 4)] for k in np.argsort(probs)[::-1][:3]]

        tag = f"_true{true}" if true is not None else ""
        fname = f"sample_{i + 1}{tag}_pred{pred}.png"
        save_image(x, os.path.join(out_dir, fname))           # 存的就是模型真正看到的 28×28
        correct = None if true is None else bool(true == pred)
        items.append({"index": i + 1, "file": name, "pred": pred, "prob": round(conf, 4), "top3": top3,
                      "true": true, "correct": correct, "saved_image": fname,
                      "input_size": prep["meta"]["orig_size"], "preprocess": prep["meta"]})
        shown.append((x, pred, true, conf))
        flag = "?" if correct is None else ("✓" if correct else "✗")
        print(f"  [{i + 1}/{len(files)}] {flag} {name} 原尺寸{prep['meta']['orig_size']} "
              f"反色={'是' if prep['meta']['inverted'] else '否'} → pred={pred} p={conf:.3f}")

    done = [it for it in items if "error" not in it]
    labeled = [it for it in done if it["true"] is not None]
    summary = {
        "run_name": config.run_name,
        "model_type": config.model_type,
        "model_path": os.path.abspath(model_path),
        "source_image_dir": in_dir,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_images": len(files), "num_succeeded": len(done), "num_failed": len(files) - len(done),
        "ignored_non_png": ignored,
        "num_labeled": len(labeled),
        "accuracy_on_labeled": round(sum(it["correct"] for it in labeled) / len(labeled), 4) if labeled else None,
        "preprocess_spec": {
            "canvas": "28x28", "ink_long_side": 20, "centering": "灰度质心居中",
            "polarity": "白底黑字自动反色", "threshold": "Otsu 自动阈值",
            "normalize": "mean=0.1307 std=0.3081",
        },
        "note": "文件名以数字开头（如 7.png / 3_我写的.png）时会当作标准答案参与准确率统计",
        "results": items,
    }
    json_path = os.path.join(out_dir, "custom_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    png_path = None
    if shown:
        png_path = plot_custom_predictions(
            [s[0] for s in shown], [s[1] for s in shown], [s[3] for s in shown],
            [s[2] for s in shown], os.path.join(out_dir, "predictions_custom.png"))

    acc = summary["accuracy_on_labeled"]
    print(f"[自定义推理完成] 成功 {len(done)}/{len(files)} 张"
          + (f"，标注样本准确率 {acc}" if acc is not None else "（没有带标注的文件名，不算准确率）"))
    print(f"==== 本次产物保存位置（绝对路径）====")
    print(f"  输入目录     : {in_dir}")
    print(f"  输出目录     : {os.path.abspath(out_dir)}")
    print(f"  预处理后输入图: {[it['saved_image'] for it in done]}")
    print(f"  结果拼图     : {os.path.abspath(png_path) if png_path else '未生成（缺 matplotlib）'}")
    print(f"  结果JSON     : {os.path.abspath(json_path)}")
    print(f"  参考：原有 MNIST 示例产物未被改动，仍在 {os.path.abspath(os.path.join(config.output_dir, 'examples'))}")
    return summary, out_dir
