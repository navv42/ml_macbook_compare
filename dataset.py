import numpy as np
from mlx.data.datasets import load_cifar10, load_cifar100

def normalize(x, mean, std):
    x = x.astype("float32") / 255.0
    return (x - mean) / std 

def get_cifar10(batch_size, root=None):
    tr = load_cifar10(root=root)

    # CIFAR-10 specific statistics
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((1, 1, 3))
    std = np.array([0.2470, 0.2435, 0.2616]).reshape((1, 1, 3))

    # Enhanced training pipeline
    tr_iter = (
        tr.to_stream()
        .shuffle(buffer_size=10000)  # Per-epoch shuffling
        .image_random_h_flip("image", prob=0.5)
        .pad("image", 0, 4, 4, 0.0)  # Add 4px padding on all sides
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", lambda x: normalize(x, mean, std))
        .batch(batch_size)
        .prefetch(8, 4)  
    )

    # Validation pipeline (no augmentations)
    test = load_cifar10(root=root, train=False)
    test_iter = (
        test.to_stream()
        .key_transform("image", lambda x: normalize(x, mean, std))
        .batch(batch_size)
        .prefetch(4, 2)
    )

    return tr_iter, test_iter

def get_cifar100(batch_size, root=None):
    tr = load_cifar100(root=root)

    # CIFAR-100 specific statistics
    mean = np.array([0.5071, 0.4867, 0.4408]).reshape((1, 1, 3))
    std = np.array([0.2675, 0.2565, 0.2761]).reshape((1, 1, 3))

    # Enhanced training pipeline
    tr_iter = (
        tr.to_stream()
        .shuffle(buffer_size=10000)  # Per-epoch shuffling
        .image_random_h_flip("image", prob=0.5)
        .pad("image", 0, 4, 4, 0.0)  # Add 4px padding on all sides
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", lambda x: normalize(x, mean, std))
        .batch(batch_size)
        .prefetch(16, 8)  
    )

    # Validation pipeline (no augmentations)
    test = load_cifar100(root=root, train=False)
    test_iter = (
        test.to_stream()
        .key_transform("image", lambda x: normalize(x, mean, std))
        .batch(batch_size)
        .prefetch(4, 2)
    )

    return tr_iter, test_iter