"""train.py —— 训练流程（含日志、模型保存、曲线）

训练过程中持续记录：每轮 train_loss / train_acc / val_loss / val_acc，
并保存：模型权重(best/last)、config、history、曲线图。
"""


import json
import os
import time

import paddle

from data_utils import load_dataset, save_load_record
from model import build_model
from visualize import plot_history


def evaluate(model, loader, criterion):
    """在 loader 上跑一遍，返回 (平均损失, 准确率)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with paddle.no_grad():
        for batch_id, (x, y) in enumerate(loader):
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()                                    # 0-d 张量必须用 .item()
            preds = logits.argmax(axis=1)
            correct += int((preds == y).astype("int64").sum().item())   # 0-d 张量必须用 .item()
            total += y.shape[0]
    return total_loss / (batch_id + 1), correct / total


def train(config, logger, log_path=None):
    """主训练函数。"""
    paddle.set_device("cpu")
    logger.info("==== 启动训练 ====")
    logger.info("参数：%s", json.dumps(config.to_dict(), ensure_ascii=False))

    # 1) 加载数据 + 记录留档
    train_loader, test_loader, load_record = load_dataset(config)
    logger.info("数据集加载记录：%s", json.dumps(load_record, ensure_ascii=False))
    save_load_record(load_record, config.log_dir, config.run_name)

    # 2) 建模型 + 选优化器
    model = build_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("模型结构=%s，参数量=%d", config.model_type, num_params)

    criterion = paddle.nn.CrossEntropyLoss()
    if config.optimizer == "sgd":
        optimizer = paddle.optimizer.SGD(learning_rate=config.learning_rate, parameters=model.parameters())
    else:
        optimizer = paddle.optimizer.Adam(learning_rate=config.learning_rate, parameters=model.parameters())

    run_dir = os.path.join(config.log_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    # 3) 逐轮训练
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()
        for batch_id, (x, y) in enumerate(train_loader):
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            train_loss += loss.item()                                    # 0-d 张量必须用 .item()
            preds = logits.argmax(axis=1)
            train_correct += int((preds == y).astype("int64").sum().item())
            train_total += y.shape[0]
        train_loss /= (batch_id + 1)
        train_acc = train_correct / train_total

        # 每个 epoch 在测试集上评估（兼当验证）
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        logger.info(
            "Epoch %02d/%02d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f | %.1fs",
            epoch, config.epochs, train_loss, train_acc, val_loss, val_acc, time.time() - t0,
        )

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            paddle.save(model.state_dict(), os.path.join(run_dir, "best_model.pdparams"))

    # 4) 收尾：保存最后模型、config、history、曲线
    model_path = os.path.join(run_dir, "model.pdparams")
    paddle.save(model.state_dict(), model_path)
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    hist_path = os.path.join(run_dir, "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    curves_path = os.path.join(run_dir, "curves.png")
    plot_history(history, curves_path)
    logger.info("训练完成。最佳验证准确率=%.4f。模型/日志/曲线已存至 %s", best_acc, run_dir)

    # 5) 数据运维：汇总本次所有产物绝对路径，方便定位与归档
    record_path = os.path.join(run_dir, "dataset_record.json")
    artifacts = [
        ("训练日志", log_path),
        ("数据集加载记录", record_path if os.path.exists(record_path) else None),
        ("最佳模型权重", os.path.join(run_dir, "best_model.pdparams")),
        ("最终模型权重", model_path),
        ("本次参数配置", cfg_path),
        ("训练过程记录(history)", hist_path),
        ("损失/准确率曲线", curves_path),
    ]
    logger.info("==== 本次产物保存位置（绝对路径）====")
    for name, p in artifacts:
        if p:
            logger.info("  %s -> %s", name, os.path.abspath(p))
    return model, history, run_dir
