import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        out = F.relu(out)
        return out


class EnhancedNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(EnhancedNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(32, 32)
        self.res_block2 = ResidualBlock(32, 64, stride=2)  # Downsample
        self.res_block3 = ResidualBlock(64, 64)
        self.res_block4 = ResidualBlock(64, 128, stride=2)  # Downsample
        self.res_block5 = ResidualBlock(128, 128)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
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
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        
        return x


def train_model(net, trainloader, testloader, optimizer, criterion, device, num_epochs=15, optimizer_name="Unknown"):
    """训练模型并返回训练损失和准确率历史"""
    train_losses = []
    test_accuracies = []
    
    print(f"\n开始训练 {optimizer_name} 优化器...")
    
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        total_batches = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            if optimizer is not None:
                optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            else:
                # 不使用优化器的情况 - 手动更新参数（非常简单的梯度下降）
                loss.backward()
                with torch.no_grad():
                    for param in net.parameters():
                        if param.grad is not None:
                            param -= 0.01 * param.grad  # 简单的学习率
                net.zero_grad()
            
            running_loss += loss.item()
            total_batches += 1
        
        # 记录平均训练损失
        avg_loss = running_loss / total_batches
        train_losses.append(avg_loss)
        
        # 计算测试准确率
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
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f'{optimizer_name} - Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_losses, test_accuracies


def main():
    # 检查CUDA是否可用并设置设备
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

    criterion = nn.CrossEntropyLoss()
    num_epochs = 15  # 减少训练轮数以节省时间
    
    # 定义要比较的优化器
    optimizers_config = [
        ("SGD", lambda params: optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)),
        ("Adam", lambda params: optim.Adam(params, lr=0.001, weight_decay=1e-4)),
        ("RMSprop", lambda params: optim.RMSprop(params, lr=0.001, weight_decay=1e-4)),
        ("Adagrad", lambda params: optim.Adagrad(params, lr=0.01, weight_decay=1e-4)),
        ("No Optimizer", lambda params: None)  # 不使用优化器
    ]
    
    # 存储结果
    results = {}
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 对每个优化器进行训练
    for i, (opt_name, opt_func) in enumerate(optimizers_config):
        print(f"\n{'='*50}")
        print(f"测试优化器: {opt_name}")
        print(f"{'='*50}")
        
        # 创建新的网络实例
        net = EnhancedNet(dropout_rate=0.5).to(device)
        
        # 初始化优化器
        optimizer = opt_func(net.parameters()) if opt_func(net.parameters()) is not None else None
        
        # 训练模型
        train_losses, test_accuracies = train_model(
            net, trainloader, testloader, optimizer, criterion, device, num_epochs, opt_name
        )
        
        # 保存结果
        results[opt_name] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'color': colors[i]
        }
    
    # 绘制结果
    plt.style.use('default')  # 使用默认样式
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制训练损失图
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    for opt_name, data in results.items():
        epochs = range(1, len(data['train_losses']) + 1)
        ax1.plot(epochs, data['train_losses'], 
                color=data['color'], linewidth=2, marker='o', markersize=4,
                label=opt_name)
    
    ax1.legend()
    ax1.set_xlim(1, num_epochs)
    
    # 绘制测试准确率图
    ax2.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    
    for opt_name, data in results.items():
        epochs = range(1, len(data['test_accuracies']) + 1)
        ax2.plot(epochs, data['test_accuracies'], 
                color=data['color'], linewidth=2, marker='s', markersize=4,
                label=opt_name)
    
    ax2.legend()
    ax2.set_xlim(1, num_epochs)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n图像已保存为 'optimizer_comparison.png'")
    
    # 显示图像
    plt.show()
    
    # 打印最终结果总结
    print(f"\n{'='*60}")
    print("最终结果总结:")
    print(f"{'='*60}")
    for opt_name, data in results.items():
        final_loss = data['train_losses'][-1]
        final_accuracy = data['test_accuracies'][-1]
        print(f"{opt_name:12s} - 最终损失: {final_loss:.4f}, 最终准确率: {final_accuracy:.2f}%")


if __name__ == '__main__':
    main()