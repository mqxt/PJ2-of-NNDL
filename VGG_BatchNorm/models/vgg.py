"""
VGG Training and Comparison System with Loss Landscape Visualization
包含VGG-A和VGG-A with BatchNorm的训练与比较
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import seaborn as sns
from tqdm import tqdm
import time

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 原始VGG-A类 (从您的代码复制)
class VGG_A(nn.Module):
    """VGG_A model - 原始版本"""
    
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()

        self.features = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes))

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG_A_BatchNorm(nn.Module):
    """VGG-A with Batch Normalization"""
    
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()

        self.features = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes))

        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def get_number_of_parameters(model):
    """计算模型参数数量"""
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()
    return parameters_n


def load_cifar10_data(batch_size=128):
    """加载CIFAR-10数据集"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                           shuffle=False, num_workers=2)

    return trainloader, testloader


def train_model(model, trainloader, testloader, epochs=50, lr=0.01, device='cpu'):
    """训练模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"开始训练，设备: {device}")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:
                progress_bar.set_postfix({
                    'Loss': f'{running_loss/(i+1):.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        test_accuracies.append(test_acc)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }


def plot_training_comparison(results_vanilla, results_bn, save_path=None):
    """绘制训练结果比较图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(results_vanilla['train_losses']) + 1)
    
    # 训练损失对比
    axes[0, 0].plot(epochs, results_vanilla['train_losses'], 'b-', label='VGG-A (Vanilla)', linewidth=2)
    axes[0, 0].plot(epochs, results_bn['train_losses'], 'r-', label='VGG-A (BatchNorm)', linewidth=2)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练准确率对比
    axes[0, 1].plot(epochs, results_vanilla['train_accuracies'], 'b-', label='VGG-A (Vanilla)', linewidth=2)
    axes[0, 1].plot(epochs, results_bn['train_accuracies'], 'r-', label='VGG-A (BatchNorm)', linewidth=2)
    axes[0, 1].set_title('Training Accuracy Comparison', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 测试准确率对比
    axes[1, 0].plot(epochs, results_vanilla['test_accuracies'], 'b-', label='VGG-A (Vanilla)', linewidth=2)
    axes[1, 0].plot(epochs, results_bn['test_accuracies'], 'r-', label='VGG-A (BatchNorm)', linewidth=2)
    axes[1, 0].set_title('Test Accuracy Comparison', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 损失下降速度对比 (对数尺度)
    axes[1, 1].semilogy(epochs, results_vanilla['train_losses'], 'b-', label='VGG-A (Vanilla)', linewidth=2)
    axes[1, 1].semilogy(epochs, results_bn['train_losses'], 'r-', label='VGG-A (BatchNorm)', linewidth=2)
    axes[1, 1].set_title('Training Loss (Log Scale)', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_loss_landscape_2d(model, trainloader, device='cpu', steps=20):
    """简化的2D损失地形可视化"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # 获取模型参数
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    original_params = torch.cat(params)
    
    # 生成两个随机方向
    direction1 = torch.randn_like(original_params)
    direction1 = direction1 / torch.norm(direction1)
    
    direction2 = torch.randn_like(original_params)
    direction2 = direction2 / torch.norm(direction2)
    # 确保方向2与方向1正交
    direction2 = direction2 - torch.dot(direction1, direction2) * direction1
    direction2 = direction2 / torch.norm(direction2)
    
    # 设置步长范围
    alpha_range = torch.linspace(-1.0, 1.0, steps)
    beta_range = torch.linspace(-1.0, 1.0, steps)
    
    loss_surface = np.zeros((steps, steps))
    
    print("生成损失地形...")
    for i, alpha in enumerate(tqdm(alpha_range)):
        for j, beta in enumerate(beta_range):
            # 计算新的参数
            new_params = original_params + alpha * direction1 + beta * direction2
            
            # 设置模型参数
            param_idx = 0
            for param in model.parameters():
                param_size = param.numel()
                param.data = new_params[param_idx:param_idx + param_size].view(param.shape)
                param_idx += param_size
            
            # 计算损失
            total_loss = 0
            count = 0
            with torch.no_grad():
                for inputs, labels in trainloader:
                    if count >= 10:  # 仅使用部分数据加速计算
                        break
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    count += 1
            
            loss_surface[i, j] = total_loss / count
    
    # 恢复原始参数
    param_idx = 0
    for param in model.parameters():
        param_size = param.numel()
        param.data = original_params[param_idx:param_idx + param_size].view(param.shape)
        param_idx += param_size
    
    return loss_surface, alpha_range.numpy(), beta_range.numpy()


def plot_loss_landscape(loss_surface, alpha_range, beta_range, title="Loss Landscape"):
    """绘制损失地形图"""
    plt.figure(figsize=(10, 8))
    
    # 创建网格
    Alpha, Beta = np.meshgrid(alpha_range, beta_range)
    
    # 绘制等高线图
    contour = plt.contour(Alpha, Beta, loss_surface.T, levels=20, colors='black', alpha=0.4, linewidths=0.5)
    contourf = plt.contourf(Alpha, Beta, loss_surface.T, levels=20, cmap='viridis', alpha=0.8)
    
    plt.colorbar(contourf, label='Loss')
    plt.xlabel('Direction 1 (α)')
    plt.ylabel('Direction 2 (β)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 标记原点 (原始参数位置)
    plt.plot(0, 0, 'ro', markersize=8, label='Original Parameters')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载CIFAR-10数据集...")
    trainloader, testloader = load_cifar10_data(batch_size=128)
    
    # 创建模型
    print("\n创建模型...")
    model_vanilla = VGG_A(num_classes=10)
    model_bn = VGG_A_BatchNorm(num_classes=10)
    
    print(f"VGG-A (Vanilla) 参数数量: {get_number_of_parameters(model_vanilla):,}")
    print(f"VGG-A (BatchNorm) 参数数量: {get_number_of_parameters(model_bn):,}")
    
    # 训练模型
    print("\n=== 训练 VGG-A (Vanilla) ===")
    start_time = time.time()
    results_vanilla = train_model(model_vanilla, trainloader, testloader, 
                                epochs=20, lr=0.01, device=device)
    vanilla_time = time.time() - start_time
    
    print(f"\n=== 训练 VGG-A (BatchNorm) ===")
    start_time = time.time()
    results_bn = train_model(model_bn, trainloader, testloader, 
                           epochs=20, lr=0.01, device=device)
    bn_time = time.time() - start_time
    
    # 打印最终结果
    print(f"\n=== 训练结果总结 ===")
    print(f"VGG-A (Vanilla):")
    print(f"  - 训练时间: {vanilla_time:.2f}s")
    print(f"  - 最终训练准确率: {results_vanilla['train_accuracies'][-1]:.2f}%")
    print(f"  - 最终测试准确率: {results_vanilla['test_accuracies'][-1]:.2f}%")
    
    print(f"\nVGG-A (BatchNorm):")
    print(f"  - 训练时间: {bn_time:.2f}s")
    print(f"  - 最终训练准确率: {results_bn['train_accuracies'][-1]:.2f}%")
    print(f"  - 最终测试准确率: {results_bn['test_accuracies'][-1]:.2f}%")
    
    # 绘制比较图
    print("\n绘制训练结果比较图...")
    plot_training_comparison(results_vanilla, results_bn, 'vgg_comparison.png')
    
    # 可视化损失地形 (可选，计算量较大)
    visualize_landscape = input("\n是否生成损失地形可视化？(y/n): ").lower() == 'y'
    
    if visualize_landscape:
        print("\n生成损失地形可视化...")
        
        # VGG-A (Vanilla) 损失地形
        print("VGG-A (Vanilla) 损失地形...")
        surface_vanilla, alpha, beta = visualize_loss_landscape_2d(
            model_vanilla, trainloader, device, steps=15)
        plot_loss_landscape(surface_vanilla, alpha, beta, 
                          "VGG-A (Vanilla) Loss Landscape")
        
        # VGG-A (BatchNorm) 损失地形
        print("VGG-A (BatchNorm) 损失地形...")
        surface_bn, alpha, beta = visualize_loss_landscape_2d(
            model_bn, trainloader, device, steps=15)
        plot_loss_landscape(surface_bn, alpha, beta, 
                          "VGG-A (BatchNorm) Loss Landscape")
    
    print("\n训练和比较完成！")


if __name__ == '__main__':
    main()