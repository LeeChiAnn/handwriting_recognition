"""model.py —— 三种可切换的网络结构（教学对比用）

MLP  ：最简单，全连接，适合讲"什么是神经网络"
LeNet：经典 CNN，带卷积，手写识别常用
CNN  ：稍深的卷积网络，准确率更高

切换方式：config.model_type = "mlp" / "lenet" / "cnn"
"""


import paddle


class MLP(paddle.nn.Layer):
    """多层感知机：784 -> 256 -> 128 -> 10。"""

    def __init__(self, hidden_size=256, dropout=0.2, num_classes=10):
        super().__init__()
        self.flatten = paddle.nn.Flatten()
        self.fc1 = paddle.nn.Linear(784, hidden_size)
        self.fc2 = paddle.nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = paddle.nn.Linear(hidden_size // 2, num_classes)
        self.relu = paddle.nn.ReLU()
        self.drop = paddle.nn.Dropout(dropout)

    def forward(self, x):
        x = self.flatten(x)                      # [B,1,28,28] -> [B,784]
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.fc3(x)


class LeNet(paddle.nn.Layer):
    """LeNet-5 风格卷积网络。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(1, 6, 5, padding=2)   # 28->28
        self.pool = paddle.nn.MaxPool2D(2, 2)               # 28->14
        self.conv2 = paddle.nn.Conv2D(6, 16, 5)             # 14->10
        self.relu = paddle.nn.ReLU()
        self.flatten = paddle.nn.Flatten()
        self.fc1 = paddle.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = paddle.nn.Linear(120, 84)
        self.fc3 = paddle.nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # [B,6,14,14]
        x = self.pool(self.relu(self.conv2(x)))   # [B,16,5,5]
        x = self.flatten(x)                       # [B,400]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class SimpleCNN(paddle.nn.Layer):
    """稍深的 CNN：两个卷积块 + 全连接分类头。"""

    def __init__(self, dropout=0.2, num_classes=10):
        super().__init__()
        self.conv_block = paddle.nn.Sequential(
            paddle.nn.Conv2D(1, 32, 3, padding=1), paddle.nn.ReLU(), paddle.nn.MaxPool2D(2, 2),
            paddle.nn.Conv2D(32, 64, 3, padding=1), paddle.nn.ReLU(), paddle.nn.MaxPool2D(2, 2),
        )
        self.head = paddle.nn.Sequential(
            paddle.nn.Flatten(),
            paddle.nn.Linear(64 * 7 * 7, 128), paddle.nn.ReLU(), paddle.nn.Dropout(dropout),
            paddle.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.conv_block(x))


def build_model(config):
    """按 config.model_type 返回对应模型实例。"""
    if config.model_type == "mlp":
        return MLP(hidden_size=config.hidden_size, dropout=config.dropout)
    if config.model_type == "lenet":
        return LeNet()
    if config.model_type == "cnn":
        return SimpleCNN(dropout=config.dropout)
    raise ValueError(f"未知 model_type={config.model_type!r}，可选 mlp/lenet/cnn")
