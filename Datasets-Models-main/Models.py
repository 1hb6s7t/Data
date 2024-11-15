import mindspore
from mindspore import nn
from mindspore import Tensor, context
import mindspore.ops as ops
import numpy as np

'''
Heartbeat dataset model:
'''
class CNNH(nn.Cell):
    def __init__(self, input_size, num_classes):
        super(CNNH, self).__init__()

        self.conv = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5, stride=1)

        self.conv_pad = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, pad_mode='pad', padding=2)
        self.drop_50 = nn.Dropout(keep_prob=0.5)

        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2)

        self.dense1 = nn.Dense(32 * 8, 32)
        self.dense2 = nn.Dense(32, 32)

        self.dense_final = nn.Dense(32, num_classes)
        self.softmax = nn.LogSoftmax(axis=1)

        # 定义操作
        self.relu = nn.ReLU()
        self.reshape = ops.Reshape()
        self.add = ops.Add()

    def construct(self, x):
        residual = self.conv(x)

        # block1
        x = self.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x = self.add(x, residual)
        x = self.relu(x)
        residual = self.maxpool(x)

        # block2
        x = self.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x = self.add(x, residual)
        x = self.relu(x)
        residual = self.maxpool(x)

        # block3
        x = self.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x = self.add(x, residual)
        x = self.relu(x)
        residual = self.maxpool(x)

        # block4
        x = self.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x = self.add(x, residual)
        x = self.relu(x)
        x = self.maxpool(x)

        # MLP
        batch_size = x.shape[0]
        x = self.reshape(x, (batch_size, -1))
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        x = self.softmax(self.dense_final(x))
        return x

'''
Seizure dataset model:
'''
class CNNS(nn.Cell):
    def __init__(self, input_size, num_classes):
        super(CNNS, self).__init__()

        self.conv = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5, stride=1)

        self.conv_pad = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, pad_mode='pad', padding=2)
        self.drop_50 = nn.Dropout(keep_prob=0.5)

        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2)

        self.dense1 = nn.Dense(32 * 28, 128)
        self.dense2 = nn.Dense(128, 64)

        self.dense_final = nn.Dense(64, num_classes)
        self.softmax = nn.LogSoftmax(axis=1)

        # 定义操作
        self.relu = nn.ReLU()
        self.reshape = ops.Reshape()
        self.add = ops.Add()

    def construct(self, x):
        residual = self.conv(x)

        # block1
        x = self.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x = self.add(x, residual)
        x = self.relu(x)
        residual = self.maxpool(x)

        # block2
        x = self.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x = self.add(x, residual)
        x = self.relu(x)
        residual = self.maxpool(x)

        # block3
        x = self.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x = self.add(x, residual)
        x = self.relu(x)
        residual = self.maxpool(x)

        # block4
        x = self.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x = self.add(x, residual)
        x = self.relu(x)
        x = self.maxpool(x)

        # MLP
        batch_size = x.shape[0]
        x = self.reshape(x, (batch_size, -1))
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        x = self.softmax(self.dense_final(x))
        return x

