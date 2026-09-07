"""data_utils.py —— 数据集加载 + 加载记录维护

负责把 MNIST 手写数字数据集加载成"数据加载器(DataLoader)"，
并生成一份"数据集加载记录" json，方便教学追溯"这次用的什么数据、多少张"。
"""


import json
import os

import paddle
from paddle.vision import transforms
from paddle.vision.datasets import MNIST


def get_transforms():
    """图像预处理：转张量 + 标准化（MNIST 常用均值/标准差）。"""
    return transforms.Compose([
        transforms.ToTensor(),                        # [0,255] -> [0,1]
        transforms.Normalize(mean=[0.1307], std=[0.3081]),  # 标准化，均值0方差1
    ])


def _make_mnist(mode, transform, data_dir):
    """构造 MNIST 数据集，兼容不同 Paddle 版本。

    Paddle 的 MNIST 构造函数为
    ``MNIST(image_path, label_path, mode, transform, download, backend)``，
    **没有 cache_dir 参数**。这里优先把数据下载/读取到 data_dir，
    若当前版本不支持自定义路径或本地文件缺失，则退回 Paddle 默认缓存目录
    （~/.cache/paddle/dataset/mnist），保证在 AI Studio 一定能跑起来。
    """
    os.makedirs(data_dir, exist_ok=True)
    if mode == "train":
        img, lab = "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
    else:
        img, lab = "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    image_path = os.path.join(data_dir, img)
    label_path = os.path.join(data_dir, lab)
    try:
        return MNIST(image_path=image_path, label_path=label_path,
                     mode=mode, transform=transform, download=True)
    except (TypeError, FileNotFoundError, ValueError, OSError):
        # 退回默认缓存目录，保证一定可运行
        return MNIST(mode=mode, transform=transform, download=True)


def load_dataset(config):
    """返回 (train_loader, test_loader, load_record)。

    - train_loader：训练用，打乱顺序
    - test_loader ：测试用；训练时兼当验证集，方便观察泛化效果
    - load_record：数据集元信息，会被保存成 json 留档
    """
    tf = get_transforms()
    # Paddle 的 MNIST 没有 cache_dir 参数；用 _make_mnist 兼容处理，
    # 优先下载到 config.data_dir，失败则退回 Paddle 默认缓存目录
    train_set = _make_mnist("train", tf, config.data_dir)
    test_set = _make_mnist("test", tf, config.data_dir)

    train_loader = paddle.io.DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    test_loader = paddle.io.DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    load_record = {
        "dataset": "MNIST",
        "task": "手写数字识别 (0-9)",
        "source": "Paddle 内置自动下载（首次运行从百度CDN下载并缓存）",
        "cache_dir": config.data_dir,
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "num_classes": 10,
        "input_shape": [1, 28, 28],
        "batch_size": config.batch_size,
        "transform": "ToTensor + Normalize(mean=0.1307, std=0.3081)",
    }
    return train_loader, test_loader, load_record


def save_load_record(load_record, log_dir, run_name):
    """把数据集加载记录保存到 logs/<run_name>/dataset_record.json。"""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, run_name, "dataset_record.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(load_record, f, ensure_ascii=False, indent=2)
    return path
