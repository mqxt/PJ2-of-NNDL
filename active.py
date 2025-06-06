import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.activation = activation
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def get_activation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(x, 0.1)
        elif self.activation == 'elu':
            return F.elu(x)
        elif self.activation == 'gelu':
            return F.gelu(x)
        elif self.activation == 'swish':
            return x * torch.sigmoid(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        else:
            return F.relu(x)  # 默认使用ReLU
    
    def forward(self, x):
        out = self.get_activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        out = self.get_activation(out)
        return out


class EnhancedNet(nn.Module):
    def __init__(self, dropout_rate=0.5, activation='relu'):
        super(EnhancedNet, self).__init__()
        
        self.activation = activation
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(32, 32, activation=activation)
        self.res_block2 = ResidualBlock(32, 64, stride=2, activation=activation)  # Downsample
        self.res_block3 = ResidualBlock(64, 64, activation=activation)
        self.res_block4 = ResidualBlock(64, 128, stride=2, activation=activation)  # Downsample
        self.res_block5 = ResidualBlock(128, 128, activation=activation)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 10)

    def get_activation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(x, 0.1)
        elif self.activation == 'elu':
            return F.elu(x)
        elif self.activation == 'gelu':
            return F.gelu(x)
        elif self.activation == 'swish':
            return x * torch.sigmoid(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        else:
            return F.relu(x)  # 默认使用ReLU

    def forward(self, x):
        # Initial convolution
        x = self.get_activation(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers with dropout
        x = self.dropout1(x)
        x = self.get_activation(self.fc1(x))
        x = self.dropout2(x)
        x = self.get_activation(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return x


def train_model(activation_name, device, trainloader, testloader, num_epochs=15):
    """训练指定激活函数的模型并返回训练历史"""
    print(f"\n开始训练 {activation_name} 激活函数模型...")
    
    # 创建模型
    net = EnhancedNet(dropout_rate=0.5, activation=activation_name).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 记录训练历史
    train_losses = []
    train_accuracies = []
    
    # 训练循环
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:
                print(f'[{activation_name}] Epoch {epoch + 1}, Batch {i + 1}: Loss = {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # 计算epoch的平均损失和准确率
        net.eval()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        with torch.no_grad():
            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()
        
        avg_loss = epoch_loss / len(trainloader)
        accuracy = 100 * epoch_correct / epoch_total
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        scheduler.step()
        
        print(f'[{activation_name}] Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%')
    
    # 测试模型
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f'[{activation_name}] 测试准确率: {test_accuracy:.2f}%')
    
    return train_losses, train_accuracies, test_accuracy


def plot_comparison(results, save_path='activation_comparison.png'):
    """绘制不同激活函数的比较图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # 绘制训练损失图
    for i, (activation, data) in enumerate(results.items()):
        epochs = range(1, len(data['train_losses']) + 1)
        ax1.plot(epochs, data['train_losses'], 
                color=colors[i % len(colors)], 
                label=f'{activation} (Test Acc: {data["test_accuracy"]:.1f}%)',
                linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 绘制训练准确率图
    for i, (activation, data) in enumerate(results.items()):
        epochs = range(1, len(data['train_accuracies']) + 1)
        ax2.plot(epochs, data['train_accuracies'], 
                color=colors[i % len(colors)], 
                label=f'{activation} (Test Acc: {data["test_accuracy"]:.1f}%)',
                linewidth=2, marker='s', markersize=4)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Accuracy (%)', fontsize=12)
    ax2.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"图像已保存为: {save_path}")


def main():
    # 检查设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    batch_size = 128

    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    # 定义要比较的激活函数
    activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'tanh']
    
    # 存储结果
    results = {}
    
    # 训练每个激活函数的模型
    for activation in activations:
        train_losses, train_accuracies, test_accuracy = train_model(
            activation, device, trainloader, testloader, num_epochs=15)
        
        results[activation] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracy': test_accuracy
        }
    
    # 绘制比较图
    plot_comparison(results)
    
    # 打印最终结果总结
    print("\n=== 激活函数性能总结 ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    for i, (activation, data) in enumerate(sorted_results, 1):
        print(f"{i}. {activation}: 测试准确率 = {data['test_accuracy']:.2f}%")


if __name__ == '__main__':
    main()