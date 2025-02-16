import matplotlib.pyplot as plt
import numpy as np
import mlx.core as mx
import os

CIFAR10_LABELS = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat",
    4: "deer", 5: "dog", 6: "frog", 7: "horse",
    8: "ship", 9: "truck"
}

CIFAR100_LABELS = { 
    0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver", 5: "bed", 6: "bee", 7: "beetle", 
    8: "bicycle", 9: "bottle", 10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly", 15: "camel", 
    16: "can", 17: "castle", 18: "caterpillar", 19: "cattle", 20: "chair", 21: "chimpanzee", 22: "clock", 
    23: "cloud", 24: "cockroach", 25: "couch", 26: "crab", 27: "crocodile", 28: "cup", 29: "dinosaur", 
    30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox", 35: "girl", 36: "hamster", 
    37: "house", 38: "kangaroo", 39: "keyboard", 40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 
    44: "lizard", 45: "lobster", 46: "man", 47: "maple_tree", 48: "motorcycle", 49: "mountain", 50: "mouse", 
    51: "mushroom", 52: "oak_tree", 53: "orange", 54: "orchid", 55: "otter", 56: "palm_tree", 57: "pear", 
    58: "pickup_truck", 59: "pine_tree", 60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum", 
    65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket", 70: "rose", 71: "sea", 72: "seal", 
    73: "shark", 74: "shrew", 75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider", 
    80: "squirrel", 81: "streetcar", 82: "sunflower", 83: "sweet_pepper", 84: "table", 85: "tank", 
    86: "telephone", 87: "television", 88: "tiger", 89: "tractor", 90: "train", 91: "trout", 92: "tulip", 
    93: "turtle", 94: "wardrobe", 95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman", 99: "worm"
}

def plot_predictions(model, test_batch, num_images=10):
    """Plot model predictions alongside true labels."""
    batch = next(test_batch)
    x = mx.array(batch["image"])
    y = mx.array(batch["label"])
    
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
    plt.show()
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
        axes[i].set_title(CIFAR100_LABELS[label_index])
    
    plt.tight_layout()
    plt.show()
    data_iter.reset()
    return fig



def generate_performance_plots(metrics, lr, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    train_metrics = metrics['train_metrics']
    timestamps = metrics['timestamps']
    
    # Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot([m['epoch'] for m in train_metrics], [m['loss'] for m in train_metrics])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.savefig(os.path.join(save_dir, f'loss_vs_epoch_{lr}.png'))
    plt.close()

    # Training Loss vs Wall Time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, [m['loss'] for m in train_metrics])
    plt.xlabel('Wall Time (s)')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Wall Time')
    plt.savefig(os.path.join(save_dir, f'loss_vs_walltime_{lr}.png'))
    plt.close()

    # Throughput
    plt.figure(figsize=(10, 6))
    plt.plot([m['epoch'] for m in train_metrics], [m['throughput'] for m in train_metrics])
    plt.xlabel('Epoch')
    plt.ylabel('Images/Second')
    plt.title('Training Throughput')
    plt.savefig(os.path.join(save_dir, f'throughput_{lr}.png'))
    plt.close()

    # GPU Memory Usage
    if any(m['gpu_mem_peak'] > 0 for m in train_metrics):
        plt.figure(figsize=(10, 6))
        plt.plot([m['epoch'] for m in train_metrics], [m['gpu_mem_peak'] for m in train_metrics], label='Max')
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
    plt.plot([m['epoch'] for m in train_metrics], [m['acc'] for m in train_metrics], label='Train')
    plt.plot([m['epoch'] for m in train_metrics], [m['test_acc'] for m in train_metrics], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'accuracy_{lr}.png'))
    plt.close()
