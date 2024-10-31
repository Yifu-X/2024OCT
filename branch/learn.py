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

# 获取设备
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 创建神经网络模型，并将其移动到指定设备（CPU或GPU）
model = NeuralNetwork().to(device)
print(model)  # 打印模型的结构和参数信息
X = torch.rand(1, 28, 28, device=device)# 生成一个随机输入张量，形状为 (1, 28, 28)，表示1个28x28的图像
logits = model(X)# 将输入张量传递给模型，获取输出logits
# 使用Softmax函数计算每个类别的预测概率
pred_probab = nn.Softmax(dim=1)(logits)
# 获取预测概率中最大值的索引，作为模型预测的类别
y_pred = pred_probab.argmax(1)
# 打印预测的类别
print(f"Predicted class: {y_pred}")

# 使用模型的 parameters() 或 named_parameters() 方法使所有参数可访问。
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
'''
'''
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)   # 我们需要对其进行优化,我们需要能够计算损失函数相对于这些变量的梯度,我们设置了这些张量的 requires_grad 属性。
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x, w)+b
print(z.requires_grad)

# 禁用梯度跟踪,我们只想对网络进行前向计算。我们可以通过用torch.no_grad()块包围我们的计算代码来停止跟踪计算。
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
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

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()  # 将模型设置为训练模式，这对 Batch Normalization 和 Dropout 层很重要
    size = len(dataloader.dataset)  # 获取训练集的总样本数
    for batch, (X, y) in enumerate(dataloader):  # 遍历数据加载器，获取每个批次的数据和标签
        # 将数据移到 GPU
        X, y = X.to(device), y.to(device)
        pred = model(X)  # 前向传播：计算模型对输入 X 的预测
        loss = loss_fn(pred, y)  # 计算预测值与真实标签 y 之间的损失

        loss.backward()  # 反向传播：计算损失相对于模型参数的梯度
        optimizer.step()  # 使用优化器更新模型参数
        optimizer.zero_grad()  # 清空之前的梯度，以便于下一次迭代

        if batch % 100 == 0:  # 每100个批次输出一次损失
            loss, current = loss.item(), batch * batch_size + len(X)  # 获取当前损失值和处理样本数
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # 打印损失和当前进度
    # 打印当前学习率
    current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
    print(f"Current learning rate: {current_lr:.6f}")  # 打印学习率
    # 更新学习率
    scheduler.step(loss)  # 使用当前的验证损失来更新学习率

def test_loop(dataloader, model, loss_fn):
    model.eval()  # 将模型设置为评估模式，这对 Batch Normalization 和 Dropout 层很重要
    size = len(dataloader.dataset)  # 获取测试集的总样本数
    num_batches = len(dataloader)  # 获取测试集的批次数
    test_loss, correct = 0, 0  # 初始化测试损失和正确预测计数

    # 使用 torch.no_grad() 来确保在测试模式下不计算梯度
    # 这可以减少不必要的梯度计算和内存使用
    with torch.no_grad():
        for X, y in dataloader:  # 遍历测试数据加载器
            # 将数据移到 GPU
            X, y = X.to(device), y.to(device)
            pred = model(X)  # 前向传播：计算模型对输入 X 的预测
            test_loss += loss_fn(pred, y).item()  # 累加测试损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 计算正确预测的数量

    test_loss /= num_batches  # 计算平均测试损失
    correct /= size  # 计算准确率
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # 打印测试结果

# 设置设备为 GPU，如果可用的话
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义模型
model = NeuralNetwork().to(device)

# 选择超参量
learning_rate = 1e-4
batch_size = 64
epochs = 10

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss() # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   # 优化器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True) # 学习率调度器
# 数据加载器
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 在训练开始之前，询问用户是否加载模型参数
load_parameters = input("Do you want to load model parameters? (y/n): ")

if load_parameters.lower() == 'y':
    # 使用try-except 块捕获可能发生的错误（如文件未找到等），并打印相应的错误信息。
    try:
        model.load_state_dict(torch.load("model_parameters.pth", weights_only=True))  # 加载参数
        print("Model parameters loaded.")
    except Exception as e:
        print(f"Failed to load model parameters: {e}")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Done!")
# 训练结束后保存模型参数
torch.save(model.state_dict(), "model_parameters.pth")
