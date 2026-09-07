"""logger.py —— 训练日志机制

日志同时输出到两个地方：
  ① 控制台（方便训练时实时看进度）
  ② 文件（方便课后复盘，命名：logs/<run_name>/train_YYYYMMDD_HHMMSS.log）

这样每一次训练都留下完整记录，符合"日志维护机制"的要求。
"""


import logging
import os
import sys
import time


def get_logger(log_dir, run_name, level=logging.INFO, prefix="train"):
    """创建一个同时写控制台和文件的 logger，返回 (logger, log_path)。

    prefix 决定日志文件名前缀：训练用默认 ``train``，运维巡检用 ``ops``。
    """
    os.makedirs(log_dir, exist_ok=True)
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(run_dir, f"{prefix}_{ts}.log")

    logger = logging.getLogger(f"handwriting_{run_name}")
    logger.setLevel(level)
    logger.handlers.clear()  # 避免重复添加 handler 导致日志刷两遍

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger, log_path
