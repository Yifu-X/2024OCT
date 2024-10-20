import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Fashion-MNIST 是一个 Zalando 服装图像数据集，包含60,000 个训练样本和 10,000 个测试样本。每个样本包含一个 28×28 的灰度图像，以及一个与 10 个类别中的一个相关的标签。
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 所有 TorchVision 数据集都有两个参数 -transform 用于修改特征，target_transform 用于修改标签
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

'''
# Dataset 一次获取数据集的特征和标签。
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) # 第一列图片名，第二列标签
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # 返回数据集中样本的数量，通常用于确定训练和测试的轮次。
    def __len__(self):
        return len(self.img_labels)

    # 获取特定样本
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# 标签映射
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
# 创建可视化图
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
# 随机选择样本
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# Dataset只是一个数据路，dataloader从其中取数并决定用于每次训练的数据是什么，具有更高的操作空间
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)   # shuffle=True：在每个 epoch 开始时随机打乱数据顺序，增加训练的随机性，帮助提高模型的泛化能力。
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
'''

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法
        self.flatten = nn.Flatten()  # 将输入图像展平为一维张量
        self.linear_relu_stack = nn.Sequential(  # 定义一个顺序模块
            nn.Linear(28*28, 512),  # 第一个全连接层：将784个输入节点映射到512个输出节点
            nn.ReLU(),  # ReLU激活函数，引入非线性
            nn.Linear(512, 512),  # 第二个全连接层：将512个输入节点映射到512个输出节点
            nn.ReLU(),  # 再次应用ReLU激活函数
            nn.Linear(512, 10),  # 输出层：将512个输入节点映射到10个输出节点（对应10个类别）
        )

    # 定义了网络的前向传播过程。
    def forward(self, x):
        x = self.flatten(x)  # 将输入展平
        logits = self.linear_relu_stack(x)  # 通过全连接层计算输出
        return logits  # 返回输出的logits

# 创建神经网络模型，并将其移动到指定设备（CPU或GPU）
model = NeuralNetwork().to(device)
print(model)  # 打印模型的结构和参数信息

# 生成一个随机输入张量，形状为 (1, 28, 28)，表示1个28x28的图像
X = torch.rand(1, 28, 28, device=device)

# 将输入张量传递给模型，获取输出logits
logits = model(X)

# 使用Softmax函数计算每个类别的预测概率
pred_probab = nn.Softmax(dim=1)(logits)

# 获取预测概率中最大值的索引，作为模型预测的类别
y_pred = pred_probab.argmax(1)

# 打印预测的类别
print(f"Predicted class: {y_pred}")