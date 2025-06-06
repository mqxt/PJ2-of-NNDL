import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display
import copy

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
num_workers = 4
batch_size = 128


figures_path = "./reports/figures"
models_path = "./reports/models"

device = torch.device("cuda:0")
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)

# Test data loader
for X, y in train_loader:
    print(f"Batch shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Data type: {X.dtype}, Label type: {y.dtype}")
    print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")
    break

# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader, device):
    """Calculate model accuracy on given data loader"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return 100. * correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Modified training function to record loss values for landscape analysis
def train_with_loss_recording(model, optimizer, criterion, train_loader, val_loader, 
                            scheduler=None, epochs_n=100, best_model_path=None):
    """
    Train model and record loss values for each step
    Returns losses_list and grads for loss landscape analysis
    """
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []  # Store loss values for each epoch
    grads_list = []   # Store gradients for each epoch
    
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad_list = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for batch_idx, (data, target) in enumerate(train_loader):
            x, y = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            
            # Record loss value
            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()
            
            loss.backward()
            
            # Record gradient norm if classifier layer exists
            if hasattr(model, 'classifier') and len(model.classifier) > 4:
                if model.classifier[4].weight.grad is not None:
                    grad_norm = model.classifier[4].weight.grad.norm().item()
                    grad_list.append(grad_norm)
            
            optimizer.step()

        losses_list.append(loss_list)
        grads_list.append(grad_list)
        
        # Calculate average loss for this epoch
        learning_curve[epoch] /= batches_n
        
        # Calculate accuracies
        train_accuracy_curve[epoch] = get_accuracy(model, train_loader, device)
        val_accuracy_curve[epoch] = get_accuracy(model, val_loader, device)
        
        # Update best model
        if val_accuracy_curve[epoch] > max_val_accuracy:
            max_val_accuracy = val_accuracy_curve[epoch]
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
        
        # # Optional: Plot training progress
        # if epoch % 5 == 0 or epoch == epochs_n - 1:
        #     display.clear_output(wait=True)
        #     f, axes = plt.subplots(1, 3, figsize=(18, 4))

        #     # Plot learning curve
        #     axes[0].plot(learning_curve[:epoch+1])
        #     axes[0].set_title('Training Loss')
        #     axes[0].set_xlabel('Epoch')
        #     axes[0].set_ylabel('Loss')
            
        #     # Plot accuracy curves
        #     axes[1].plot(train_accuracy_curve[:epoch+1], label='Train')
        #     axes[1].plot(val_accuracy_curve[:epoch+1], label='Validation')
        #     axes[1].set_title('Accuracy')
        #     axes[1].set_xlabel('Epoch')
        #     axes[1].set_ylabel('Accuracy (%)')
        #     axes[1].legend()
            
        #     # Plot current batch losses
        #     if loss_list:
        #         axes[2].plot(loss_list[-min(100, len(loss_list)):])
        #         axes[2].set_title(f'Recent Batch Losses (Epoch {epoch+1})')
        #         axes[2].set_xlabel('Batch')
        #         axes[2].set_ylabel('Loss')
            
        #     plt.tight_layout()
        #     # plt.show()

    return losses_list, grads_list, learning_curve, train_accuracy_curve, val_accuracy_curve

def train_multiple_learning_rates(model_class, learning_rates, epochs=20, save_prefix=''):
    """
    Train models with different learning rates and collect loss curves
    """
    all_results = {}
    
    for lr in learning_rates:
        print(f"\n=== Training with learning rate: {lr} ===")
        
        # Reset random seed for reproducibility
        set_random_seeds(seed_value=2020, device=device)
        
        # Create fresh model
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        losses_list, grads_list, learning_curve, train_acc, val_acc = train_with_loss_recording(
            model, optimizer, criterion, train_loader, val_loader, epochs_n=epochs
        )
        
        all_results[lr] = {
            'losses_list': losses_list,
            'grads_list': grads_list,
            'learning_curve': learning_curve,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'final_train_acc': train_acc[-1],
            'final_val_acc': val_acc[-1]
        }
        
        print(f"Final train accuracy: {train_acc[-1]:.2f}%")
        print(f"Final validation accuracy: {val_acc[-1]:.2f}%")
    
    return all_results

def calculate_loss_envelope(all_results, learning_rates):
    """
    Calculate max_curve and min_curve from multiple learning rate experiments
    """
    flattened_list = {}
    for lr in learning_rates:
        all_loss = all_results[lr]['losses_list']
        flattened_list[lr] = [item for sublist in all_loss for item in sublist]
    steps = len(flattened_list[learning_rates[0]])
    max_curve = []
    min_curve = []
    
    for step in range(steps):
        step_losses = []
        for lr in learning_rates:
            step_losses.append(flattened_list[lr][step])
        
        if step_losses:
            max_curve.append(max(step_losses))
            min_curve.append(min(step_losses))
        else:
            # Handle edge case
            max_curve.append(np.nan)
            min_curve.append(np.nan)

    return max_curve, min_curve

def get_step_list(data, count=250):
    step = len(data) // count
    indices = np.arange(0, len(data), step)
    values = np.array(data)[indices]
    return values


def plot_loss_landscape_comparison(results_vanilla, results_bn, learning_rates, save_path=None):
    """
    Plot the final loss landscape comparison between VGG-A and VGG-A with BN
    """
    # Calculate envelopes for both models
    max_curve_vanilla, min_curve_vanilla = calculate_loss_envelope(results_vanilla, learning_rates)
    max_curve_bn, min_curve_bn = calculate_loss_envelope(results_bn, learning_rates)
    
    step = len(max_curve_vanilla) // 250
    indices = np.arange(0, len(max_curve_vanilla), step)

    max_curve_vanilla = get_step_list(max_curve_vanilla)
    min_curve_vanilla = get_step_list(min_curve_vanilla)
    
    max_curve_bn = get_step_list(max_curve_bn)
    min_curve_bn = get_step_list(min_curve_bn)

    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Loss envelopes comparison
    axes[0].fill_between(indices, min_curve_vanilla, max_curve_vanilla, 
                           alpha=0.3, color='blue', label='VGG-A (Vanilla) Range')
    axes[0].plot(indices, max_curve_vanilla, 'b-', linewidth=1, label='VGG-A Max')
    axes[0].plot(indices, min_curve_vanilla, 'b--', linewidth=1, label='VGG-A Min')
    
    axes[0].fill_between(indices, min_curve_bn, max_curve_bn, 
                           alpha=0.3, color='red', label='VGG-A+BN Range')
    axes[0].plot(indices, max_curve_bn, 'r-', linewidth=1, label='VGG-A+BN Max')
    axes[0].plot(indices, min_curve_bn, 'r--', linewidth=1, label='VGG-A+BN Min')
    
    axes[0].set_title('Loss Stability Comparison: Impact of Batch Normalization', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Loss range (max - min) comparison
    range_vanilla = [max_curve_vanilla[i] - min_curve_vanilla[i] for i in range(len(indices))]
    range_bn = [max_curve_bn[i] - min_curve_bn[i] for i in range(len(indices))]
    
    axes[1].plot(indices, range_vanilla, 'b-', linewidth=1, label='VGG-A (Vanilla)')
    axes[1].plot(indices, range_bn, 'r-', linewidth=1, label='VGG-A with BatchNorm')
    axes[1].set_title('Loss Variation Range (Max - Min)', fontsize=12)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss Range')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss landscape comparison saved to: {save_path}")
    
    # plt.show()
    
    return max_curve_vanilla, min_curve_vanilla, max_curve_bn, min_curve_bn

def print_stability_analysis(results_vanilla, results_bn, learning_rates):
    """
    Print quantitative analysis of loss stability
    """
    print("\n" + "="*60)
    print("LOSS STABILITY ANALYSIS")
    print("="*60)
    
    # Calculate statistics for each model type
    for model_name, results in [("VGG-A (Vanilla)", results_vanilla), ("VGG-A with BatchNorm", results_bn)]:
        print(f"\n{model_name}:")
        print("-" * 40)
        
        final_losses = []
        final_train_accs = []
        final_val_accs = []
        
        for lr in learning_rates:
            final_loss = results[lr]['learning_curve'][-1]
            final_train_acc = results[lr]['final_train_acc']
            final_val_acc = results[lr]['final_val_acc']
            
            final_losses.append(final_loss)
            final_train_accs.append(final_train_acc)
            final_val_accs.append(final_val_acc)
            
            print(f"  LR {lr:6.1e}: Loss={final_loss:.4f}, Train Acc={final_train_acc:.2f}%, Val Acc={final_val_acc:.2f}%")
        
        # Calculate stability metrics
        loss_std = np.std(final_losses)
        loss_range = max(final_losses) - min(final_losses)
        acc_std = np.std(final_val_accs)
        
        print(f"\n  Stability Metrics:")
        print(f"    Loss Std Dev: {loss_std:.4f}")
        print(f"    Loss Range: {loss_range:.4f}")
        print(f"    Val Acc Std Dev: {acc_std:.2f}%")
        print(f"    Best Val Acc: {max(final_val_accs):.2f}%")
        print(f"    Worst Val Acc: {min(final_val_accs):.2f}%")

# Main execution
if __name__ == "__main__":
    # Choose learning rates for the experiment
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]  # Different step sizes
    # learning_rates = [1e-4, 5e-4]
    epochs = 6
    
    print("Selected Learning Rates:", learning_rates)
    print(f"Training Epochs: {epochs}")
    
    # Train VGG-A (Vanilla) with different learning rates
    print("\n" + "="*50)
    print("TRAINING VGG-A (VANILLA) WITH DIFFERENT LEARNING RATES")
    print("="*50)
    results_vanilla = train_multiple_learning_rates(VGG_A, learning_rates, epochs, 'vanilla')
    
    # Train VGG-A with BatchNorm with different learning rates
    print("\n" + "="*50)
    print("TRAINING VGG-A WITH BATCHNORM WITH DIFFERENT LEARNING RATES")
    print("="*50)
    results_bn = train_multiple_learning_rates(VGG_A_BatchNorm, learning_rates, epochs, 'bn')
    
    # Print stability analysis
    print_stability_analysis(results_vanilla, results_bn, learning_rates)
    
    # Plot comprehensive comparison
    print("\n" + "="*50)
    print("GENERATING LOSS LANDSCAPE VISUALIZATION")
    print("="*50)
    
    save_path = os.path.join(figures_path, 'loss_landscape_comparison.png') if figures_path else 'loss_landscape_comparison.png'
    max_curve_vanilla, min_curve_vanilla, max_curve_bn, min_curve_bn = plot_loss_landscape_comparison(
        results_vanilla, results_bn, learning_rates, save_path
    )
    
    # Save numerical results
    if figures_path:
        # Save loss curves data
        np.savetxt(os.path.join(figures_path, 'vanilla_max_curve.txt'), max_curve_vanilla, fmt='%.6f')
        np.savetxt(os.path.join(figures_path, 'vanilla_min_curve.txt'), min_curve_vanilla, fmt='%.6f')
        np.savetxt(os.path.join(figures_path, 'bn_max_curve.txt'), max_curve_bn, fmt='%.6f')
        np.savetxt(os.path.join(figures_path, 'bn_min_curve.txt'), min_curve_bn, fmt='%.6f')
        print(f"Numerical results saved to: {figures_path}")
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print("\nKey Findings:")
    print("1. The filled areas show the loss variation range for different learning rates")
    print("2. Smaller filled areas indicate better stability (less sensitivity to learning rate)")
    print("3. BatchNorm typically shows:")
    print("   - Reduced loss variation across different learning rates")
    print("   - More stable training dynamics")
    print("   - Better convergence properties")
    print("4. The loss range plot shows the difference between max and min losses over time")