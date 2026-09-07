"""config.py —— 集中管理所有可调参数

本文件把所有"可以调"的超参数集中在一个 Config 类里：
- 想演示"学习率对收敛速度的影响" → 改 learning_rate
- 想演示"网络结构对准确率的影响" → 改 model_type
- 想演示"训练轮数对过拟合的影响" → 改 epochs

课堂上用命令行覆盖即可，例如：
    python main.py --lr 0.01 --model_type mlp --epochs 3
"""


from dataclasses import dataclass, asdict
import argparse
import time


@dataclass
class Config:
    # ===== 数据 =====
    data_dir: str = "./data"          # 数据集缓存目录（MNIST 自动下载到这里）
    batch_size: int = 128             # 每批送几张图进网络
    image_dir: str = "./inputs"       # 自定义推理的输入图目录（--mode custom 使用）

    # ===== 模型 / 训练（最常被调的参数）=====
    model_type: str = "lenet"         # 网络结构：mlp / lenet / cnn
    learning_rate: float = 0.001      # 学习率：太大震荡，太小慢
    epochs: int = 5                   # 训练轮数
    optimizer: str = "adam"           # 优化器：sgd / adam
    hidden_size: int = 256            # 仅 MLP 使用：隐藏层神经元数
    dropout: float = 0.2              # 随机丢弃比例，防过拟合

    # ===== 输出 / 日志 =====
    log_dir: str = "./logs"           # 训练日志与曲线保存目录
    output_dir: str = "./outputs"     # 推理示例图片保存目录
    run_name: str = None              # 本次运行名（None 时自动用时间戳，避免覆盖）
    num_show: int = 9                 # 推理时展示几个示例
    seed: int = 42                    # 随机种子，保证可复现

    # ===== 数据运维巡检（--mode ops 使用）=====
    ops_dir: str = "./outputs/ops"    # 巡检报告 / 看板的输出目录
    retention_days: int = 14          # run 产物保留期（超期只提示归档，绝不自动删除）
    refresh_baseline: bool = False    # True 时重建数据集指纹基线（--refresh_baseline）

    def to_dict(self):
        return asdict(self)


def build_config_from_args(argv=None):
    """解析命令行参数，覆盖默认值，返回 (mode, cfg)。"""
    p = argparse.ArgumentParser(description="手写数字识别实训项目 (MNIST)")
    p.add_argument("--mode", choices=["train", "inference", "demo", "ops", "custom"], default="train",
                   help="train=训练; inference=推理示例; demo=快速演示(训练+推理); "
                        "ops=数据运维巡检(只读扫描 logs/ 与 data/ 产物并出看板); "
                        "custom=识别 --image_dir 里用户自己的 PNG(产物写 outputs/custom/)")
    p.add_argument("--model_type", default=None, help="mlp / lenet / cnn")
    p.add_argument("--lr", type=float, default=None, help="学习率")
    p.add_argument("--epochs", type=int, default=None, help="训练轮数")
    p.add_argument("--batch_size", type=int, default=None, help="批大小")
    p.add_argument("--optimizer", default=None, choices=["sgd", "adam"], help="优化器")
    p.add_argument("--hidden_size", type=int, default=None, help="MLP 隐藏层大小")
    p.add_argument("--dropout", type=float, default=None, help="dropout 比例")
    p.add_argument("--run_name", default=None, help="本次运行名")
    p.add_argument("--num_show", type=int, default=None, help="推理展示样本数")
    p.add_argument("--image_dir", default=None, help="自定义推理的输入图目录（--mode custom）")
    p.add_argument("--seed", type=int, default=None, help="随机种子")
    # ---- 数据运维巡检参数（--mode ops）----
    p.add_argument("--ops_dir", default=None, help="巡检报告输出目录")
    p.add_argument("--retention_days", type=int, default=None, help="run 产物保留天数")
    p.add_argument("--refresh_baseline", action="store_true",
                   help="重建数据集指纹基线（数据确认无损后重新立基线用）")
    args = p.parse_args(argv)

    cfg = Config()
    for key, val in vars(args).items():
        if key == "mode" or val is None:
            continue
        if key == "lr":
            setattr(cfg, "learning_rate", val)
        else:
            setattr(cfg, key, val)

    if cfg.run_name is None and args.mode != "ops":
        # ops 模式不自动编 run 名：留 None 代表“巡检 logs/ 下全部 run”
        cfg.run_name = "run_" + time.strftime("%Y%m%d_%H%M%S")
    return args.mode, cfg
