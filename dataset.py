import numpy as np
from mlx.data.datasets import load_cifar10

def get_cifar10(batch_size, root=None):
    tr = load_cifar10(root=root)

    # CIFAR-10 specific statistics
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((1, 1, 3))
    std = np.array([0.2470, 0.2435, 0.2616]).reshape((1, 1, 3))

    def normalize(x):
        x = x.astype("float32") / 255.0
        return (x - mean) / std 

    def apply_cutout(img):
        """Apply Cutout regularization with 50% probability"""
        if np.random.rand() < 0.5:
            h, w, _ = img.shape
            mask_size = 16
            if h >= mask_size and w >= mask_size:
                x = np.random.randint(0, w - mask_size + 1)
                y = np.random.randint(0, h - mask_size + 1)
                img[y:y+mask_size, x:x+mask_size, :] = 0
        return img

    # Enhanced training pipeline
    tr_iter = (
        tr.to_stream()
        .shuffle(buffer_size=10000)  # Per-epoch shuffling
        .image_random_h_flip("image", prob=0.5)
        .pad("image", 0, 4, 4, 0.0)  # Add 4px padding on all sides
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", normalize)
        .batch(batch_size)
        .prefetch(8, 4)  
    )

    # Validation pipeline (no augmentations)
    test = load_cifar10(root=root, train=False)
    test_iter = (
        test.to_stream()
        .key_transform("image", normalize)
        .batch(batch_size)
        .prefetch(4, 2)
    )

    return tr_iter, test_iter