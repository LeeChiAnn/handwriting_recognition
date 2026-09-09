"""main.py —— 统一入口（实际运行用这个，不在 notebook 里跑大模型）

用法示例：
  python main.py --mode train     --model_type lenet --lr 0.001 --epochs 5 --run_name demo
  python main.py --mode inference --run_name demo --num_show 9
  python main.py --mode demo      --model_type mlp  --lr 0.01  --epochs 2
  python main.py --mode ops                       # 数据运维巡检（不需要 paddle）
  python main.py --mode custom --run_name demo --image_dir inputs
                                     # 识别自己写的数字图（任意尺寸 PNG），产物在 outputs/custom/
  python main.py --mode eval  --run_name demo      # 全测试集验证模型效果（需 paddle，在 AI Studio 跑）
  python main.py --mode report                     # 生成三份可下载的离线成果 HTML（不需 paddle）

说明：本文件是"真正干活的入口"。教学讲解看 手写识别教学讲解.ipynb，
但训练/推理/巡检请在终端用上面的命令运行 .py 文件。

注意 `--mode ops` 与 `--mode report` 刻意不导入 paddle：两者只读产物、纯标准库，
因此在没装深度学习环境的电脑上也能出看板与成果包（详见 data_ops.py / lab_report.py）。
`--mode custom` 与 `--mode inference` 的产物目录严格分开（outputs/custom/ vs
outputs/examples/），跑自定义推理不会改动原有的输入图片示例。
"""


from config import build_config_from_args


def do_train(cfg):
    """训练：建 logger → 跑 train()。"""
    import paddle
    from logger import get_logger
    from train import train

    paddle.seed(cfg.seed)
    logger, log_path = get_logger(cfg.log_dir, cfg.run_name)
    train(cfg, logger, log_path)


def do_inference(cfg):
    import paddle
    from inference import run_inference

    paddle.seed(cfg.seed)
    run_inference(cfg)


def do_ops(cfg):
    """数据运维巡检：不依赖 paddle / matplotlib（缺 matplotlib 时只跳过 PNG）。"""
    from data_ops import run_ops_audit

    run_ops_audit(cfg)


def do_custom(cfg):
    """识别用户自备的 PNG（任意尺寸/白底黑字都能吃）：预处理见 custom_input.py。"""
    import paddle
    from inference import run_custom_inference

    paddle.seed(cfg.seed)                    # 推理时 dropout 已关，设种子只为与 do_inference 口径一致
    run_custom_inference(cfg)


def do_eval(cfg):
    """全测试集验证模型效果（混淆矩阵 / 每类 P·R·F1 / 错分图集），需 paddle。"""
    import paddle
    from eval_model import run_evaluation

    if not cfg.run_name:
        cfg.run_name = _pick_run_with_weights(cfg)
    paddle.seed(cfg.seed)
    run_evaluation(cfg)


def do_report(cfg):
    """生成三份自包含 HTML 成果包（模型验证 / 实验档案袋 / 入口页），不依赖 paddle。"""
    from lab_report import build_reports

    build_reports(cfg)


def _pick_run_with_weights(cfg):
    """--mode eval 没给 --run_name 时，自动挑 logs/ 下唯一有权重的 run；多个则让人指定。"""
    import glob
    import os

    dirs = sorted(d for d in glob.glob(os.path.join(cfg.log_dir, "*")) if os.path.isdir(d))
    hits = [os.path.basename(d) for d in dirs
            if os.path.exists(os.path.join(d, "best_model.pdparams"))
            or os.path.exists(os.path.join(d, "model.pdparams"))]
    if len(hits) == 1:
        print(f"[eval] 未指定 --run_name，自动使用唯一有权重的 run：{hits[0]}")
        return hits[0]
    if not hits:
        raise SystemExit(f"[eval] {cfg.log_dir}/ 下没有任何模型权重，先训练：\n"
                         f"  python main.py --mode train --run_name demo --epochs 5")
    raise SystemExit(f"[eval] 有多个带权重的 run：{', '.join(hits)}\n"
                     f"  请用 --run_name 指定一个，例如：python main.py --mode eval --run_name {hits[-1]}")


def main():
    mode, cfg = build_config_from_args()
    if mode == "train":
        do_train(cfg)
    elif mode == "inference":
        do_inference(cfg)
    elif mode == "demo":
        # 快速演示：训几轮 + 推理
        cfg.epochs = min(cfg.epochs, 2)
        do_train(cfg)
        do_inference(cfg)
    elif mode == "ops":
        do_ops(cfg)
    elif mode == "custom":
        do_custom(cfg)
    elif mode == "eval":
        do_eval(cfg)
    elif mode == "report":
        do_report(cfg)


if __name__ == "__main__":
    main()
