import matplotlib.pyplot as plt
import numpy as np
import mlx.core as mx
import os

CIFAR10_LABELS = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
    4: "deer", 5: "dog", 6: "frog", 7: "horse",
    8: "ship", 9: "truck"
}

def plot_predictions(model, test_batch, num_images=10):
    """Plot model predictions alongside true labels."""
    x = mx.array(test_batch["image"])
    y = mx.array(test_batch["label"])
    
    # Forward pass
    preds = model(x)
    pred_classes = mx.argmax(preds, axis=1)
    
    # Create plot
    fig, axes = plt.subplots(1, num_images, figsize=(2 * num_images, 2))
    for i in range(min(num_images, x.shape[0])):
        img = np.array(x[i])
        pred = pred_classes[i].item()
        true = y[i].item()
        
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(
            f"Pred: {CIFAR10_LABELS[pred]}\nTrue: {CIFAR10_LABELS[true]}", 
            fontsize=8
        )
    
    plt.tight_layout()
    return fig

def show_dataset_samples(data_iter, num_images=12):
    """Show random samples from the dataset."""
    batch = next(data_iter)
    x = mx.array(batch["image"])
    y = mx.array(batch["label"])
    
    fig, axes = plt.subplots(1, num_images, figsize=(2 * num_images, 2))
    
    for i in range(num_images):
        random_index = np.random.randint(0, x.shape[0])
        img = np.array(x[random_index])
        label_index = y[random_index].item()
        
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(CIFAR10_LABELS[label_index])
    
    plt.tight_layout()
    data_iter.reset()
    return fig



def generate_performance_plots(metrics, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    train_metrics = metrics['train_metrics']
    timestamps = metrics['timestamps']
    
    # Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot([m['epoch'] for m in train_metrics], [m['train_loss'] for m in train_metrics])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.savefig(os.path.join(save_dir, 'loss_vs_epoch.png'))
    plt.close()

    # Training Loss vs Wall Time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, [m['train_loss'] for m in train_metrics])
    plt.xlabel('Wall Time (s)')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Wall Time')
    plt.savefig(os.path.join(save_dir, 'loss_vs_walltime.png'))
    plt.close()

    # Throughput
    plt.figure(figsize=(10, 6))
    plt.plot([m['epoch'] for m in train_metrics], [m['throughput'] for m in train_metrics])
    plt.xlabel('Epoch')
    plt.ylabel('Images/Second')
    plt.title('Training Throughput')
    plt.savefig(os.path.join(save_dir, 'throughput.png'))
    plt.close()

    # GPU Memory Usage
    if any(m['gpu_memory_max'] > 0 for m in train_metrics):
        plt.figure(figsize=(10, 6))
        plt.plot([m['epoch'] for m in train_metrics], [m['gpu_memory_avg'] for m in train_metrics], label='Average')
        plt.plot([m['epoch'] for m in train_metrics], [m['gpu_memory_max'] for m in train_metrics], label='Max')
        plt.xlabel('Epoch')
        plt.ylabel('GPU Memory (MB)')
        plt.title('GPU Memory Usage')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'gpu_memory.png'))
        plt.close()

    # Batch Times Distribution
    all_batch_times = np.concatenate([m['batch_times'] for m in train_metrics])
    plt.figure(figsize=(10, 6))
    plt.hist(all_batch_times, bins=50)
    plt.xlabel('Batch Time (s)')
    plt.ylabel('Frequency')
    plt.title('Batch Processing Time Distribution')
    plt.savefig(os.path.join(save_dir, 'batch_times_hist.png'))
    plt.close()

    # Accuracy Progress
    plt.figure(figsize=(10, 6))
    plt.plot([m['epoch'] for m in train_metrics], [m['train_acc'] for m in train_metrics], label='Train')
    plt.plot([m['epoch'] for m in train_metrics], [m['test_acc'] for m in train_metrics], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()
