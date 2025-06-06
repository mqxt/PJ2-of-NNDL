import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# 1. L1正则化 (Lasso)
def l1_reg(model, lambda_l1=0.01):
    """计算模型的L1正则化项"""
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

# 2. L2正则化 (Ridge)
def l2_reg(model, lambda_l2=0.01):
    """计算模型的L2正则化项"""
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.sum(param ** 2)
    return lambda_l2 * l2_loss

def evaluate_model(net, testloader, device):
    """评估模型准确率"""
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
    return 100 * correct / total

def train_model(config, trainloader, testloader, device, num_epochs=15):
    """训练单个模型配置"""
    print(f"Training: {config['name']}")
    
    # 创建模型
    net = EnhancedNet(dropout_rate=config['dropout_rate']).to(device)
    
    # 设置损失函数
    if config['loss_type'] == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif config['loss_type'] == 'Focal':
        criterion = FocalLoss(alpha=1, gamma=2)
    elif config['loss_type'] == 'LabelSmoothing':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 设置优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 记录训练过程
    train_losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        batch_count = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            # 添加正则化（如果需要）
            if config['l1_lambda'] > 0:
                loss += l1_reg(net, config['l1_lambda'])
            if config['l2_lambda'] > 0:
                loss += l2_reg(net, config['l2_lambda'])
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
        
        # 记录平均损失
        avg_loss = running_loss / batch_count
        train_losses.append(avg_loss)
        
        # 评估准确率
        accuracy = evaluate_model(net, testloader, device)
        accuracies.append(accuracy)
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_losses, accuracies

def main():
    # 设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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

    # 定义不同的实验配置
    configs = [
        {
            'name': 'CrossEntropy + L2(1e-4)',
            'loss_type': 'CrossEntropy',
            'l2_lambda': 1e-4,
            'l1_lambda': 0,
            'dropout_rate': 0.5,
            'color': 'blue'
        },
        {
            'name': 'CrossEntropy + L2(1e-3)',
            'loss_type': 'CrossEntropy',
            'l2_lambda': 1e-3,
            'l1_lambda': 0,
            'dropout_rate': 0.5,
            'color': 'red'
        },
        {
            'name': 'CrossEntropy + L1(1e-4)',
            'loss_type': 'CrossEntropy',
            'l2_lambda': 0,
            'l1_lambda': 1e-4,
            'dropout_rate': 0.5,
            'color': 'green'
        },
        {
            'name': 'Focal Loss + L2(1e-4)',
            'loss_type': 'Focal',
            'l2_lambda': 1e-4,
            'l1_lambda': 0,
            'dropout_rate': 0.5,
            'color': 'orange'
        },
        {
            'name': 'Label Smoothing + L2(1e-4)',
            'loss_type': 'LabelSmoothing',
            'l2_lambda': 1e-4,
            'l1_lambda': 0,
            'dropout_rate': 0.5,
            'color': 'purple'
        },
        {
            'name': 'CrossEntropy + No Regularization',
            'loss_type': 'CrossEntropy',
            'l2_lambda': 0,
            'l1_lambda': 0,
            'dropout_rate': 0.5,
            'color': 'brown'
        },
        {
            'name': 'CrossEntropy + L2(1e-5)',
            'loss_type': 'CrossEntropy',
            'l2_lambda': 1e-5,
            'l1_lambda': 0,
            'dropout_rate': 0.5,
            'color': 'yellow'
        }
    ]
    
    # 存储所有结果
    all_results = {}
    
    # 训练所有配置
    for config in configs:
        train_losses, accuracies = train_model(config, trainloader, testloader, device)
        all_results[config['name']] = {
            'train_losses': train_losses,
            'accuracies': accuracies,
            'color': config['color']
        }
        print(f"Final accuracy for {config['name']}: {accuracies[-1]:.2f}%\n")
    
    # 绘制对比图
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制训练损失曲线
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    for name, results in all_results.items():
        epochs = range(1, len(results['train_losses']) + 1)
        ax1.plot(epochs, results['train_losses'], 
                color=results['color'], linewidth=2, label=name, marker='o', markersize=4)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 绘制准确率曲线
    ax2.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    
    for name, results in all_results.items():
        epochs = range(1, len(results['accuracies']) + 1)
        ax2.plot(epochs, results['accuracies'], 
                color=results['color'], linewidth=2, label=name, marker='s', markersize=4)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('loss_regularization_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('loss_regularization_comparison.pdf', bbox_inches='tight')
    print("Comparison plots saved as 'loss_regularization_comparison.png' and '.pdf'")
    
    # 显示图像
    # plt.show()
    
    # 打印最终结果总结
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    # 按最终准确率排序
    sorted_results = sorted(all_results.items(), 
                          key=lambda x: x[1]['accuracies'][-1], reverse=True)
    
    for i, (name, results) in enumerate(sorted_results, 1):
        print(f"{i}. {name}: {results['accuracies'][-1]:.2f}% accuracy")
        print(f"   Final loss: {results['train_losses'][-1]:.4f}")
        print()

if __name__ == '__main__':
    main()