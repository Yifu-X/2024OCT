import torch
import torch.nn as nn
import torch.optim as optim

# 生成一些数据
x = torch.rand(100, 1) * 10  # 输入特征
y = 2 * x + 1 + torch.randn(100, 1)  # 真实标签，加上噪声

# 定义模型
model = nn.Linear(1, 1)  # 输入特征数为1，输出特征数为1

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 训练模型
for epoch in range(100):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清空梯度

    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)  # 计算损失

    # 反向传播
    loss.backward()
    optimizer.step()  # 更新参数

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

print("训练完成！")
