import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入数据
mnist_train = pd.read_csv("mnist_train.csv")
mnist_test = pd.read_csv("mnist_test.csv")

# 提取出图像信息，并将内容从0~255的整数转换为0.0~1.0的浮点数
# 图像大小为28*28，数组中每一行代表一张图像
x_train = mnist_train.iloc[:, 1:].to_numpy().reshape(-1, 1, 28, 28) / 255
x_test = mnist_test.iloc[:, 1:].to_numpy().reshape(-1, 1, 28, 28) / 255
print(f'训练集大小：{len(x_train)}')
print(f'测试集大小：{len(x_test)}')

def display(data, m, n):
    # data：图像的像素数据，每行代表一张图像
    # m，n：按m行n列的方式展示前m * n张图像
    img = np.zeros((28 * m, 28 * n))
    for i in range(m):
        for j in range(n):
            # 填充第i行j列图像的数据
            img[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = \
                data[i * m + j].reshape(28, 28)
    plt.figure(figsize=(m * 1.5, n * 1.5))
    plt.imshow(img, cmap='gray')
    plt.show()


# CNN自编码器
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        # Encoder layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Decoder layers
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Dropout layer
        x = self.dropout(x)

        # Decoder
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))

        return x


# 训练超参数
learning_rate = 0.01
max_epoch = 20
batch_size = 256
display_step = 10
np.random.seed(0)
torch.manual_seed(0)

# 采用Adam优化器，编码器和解码器的参数共同优化
cnn_autoencoder = CNNAutoencoder()
optimizer = torch.optim.Adam(cnn_autoencoder.parameters(), lr=learning_rate)

# 开始训练
for i in range(max_epoch):
    # 打乱训练样本
    idx = np.arange(len(x_train))
    idx = np.random.permutation(idx)
    x_train = x_train[idx]
    st = 0
    ave_bce_loss = []  # 记录每一轮的平均二进制交叉熵损失
    ave_mse_loss = []  # 记录每一轮的平均均方误差损失
    while st < len(x_train):
        # 遍历数据集
        ed = min(st + batch_size, len(x_train))
        X = torch.from_numpy(x_train[st: ed]).to(torch.float32)
        X_rec = cnn_autoencoder(X)

        # 计算二进制交叉熵损失和均方误差损失
        bce_loss = F.binary_cross_entropy(X_rec, X)
        mse_loss = F.mse_loss(X_rec, X)

        # 记录损失值
        ave_bce_loss.append(bce_loss.item())
        ave_mse_loss.append(mse_loss.item())

        # 组合两种损失
        loss = bce_loss + mse_loss

        optimizer.zero_grad()
        loss.backward()  # 梯度反向传播
        optimizer.step()
        st = ed

    ave_bce_loss = np.average(ave_bce_loss)
    ave_mse_loss = np.average(ave_mse_loss)

    if i % display_step == 0 or i == max_epoch - 1:
        print(f'训练轮数：{i}，平均二进制交叉熵损失：{ave_bce_loss:.4f}，平均均方误差损失：{ave_mse_loss:.4f}')
        # 选取测试集中的部分图像重建并展示
        with torch.no_grad():
            X_test = torch.from_numpy(x_test[:3 * 5]).to(torch.float32)
            X_test_rec = cnn_autoencoder(X_test)
            X_test_rec = X_test_rec.cpu().numpy()
        print('原始图像')
        display(x_test[:3 * 5], 3, 5)
        print('重建图像')
        display(X_test_rec, 3, 5)
