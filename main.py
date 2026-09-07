"""main.py —— 统一入口（实际运行用这个，不在 notebook 里跑大模型）

用法示例：
  python main.py --mode train     --model_type lenet --lr 0.001 --epochs 5 --run_name demo
  python main.py --mode inference --run_name demo --num_show 9
  python main.py --mode demo      --model_type mlp  --lr 0.01  --epochs 2
  python main.py --mode ops                       # 数据运维巡检（不需要 paddle）
  python main.py --mode custom --run_name demo --image_dir inputs
                                     # 识别自己写的数字图（任意尺寸 PNG），产物在 outputs/custom/

说明：本文件是"真正干活的入口"。教学讲解看 手写识别教学讲解.ipynb，
但训练/推理/巡检请在终端用上面的命令运行 .py 文件。

注意 `--mode ops` 分支刻意不导入 paddle：运维巡检只读产物、纯标准库，
因此在没装深度学习环境的电脑上也能出看板（详见 data_ops.py）。
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


if __name__ == "__main__":
    main()
