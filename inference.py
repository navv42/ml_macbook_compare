import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time

# Load your trained MLX model (replace with your actual model class)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 16 * 16, 10)
    
    def __call__(self, x):
        x = self.pool(nn.relu(self.conv1(x)))
        x = x.reshape(-1, 64 * 16 * 16)
        x = self.fc1(x)
        return x

model = CNN()
# Load your pretrained weights here (if saved)
# model.load_weights("path/to/weights.npz")

# Generate dummy input (or load real CIFAR-10 test data)
batch_size = 512
dummy_input = mx.random.normal((batch_size, 32, 32, 3))  # Shape: [N, H, W, C]

# Warmup (initialize GPU/Neural Engine)
_ = model(dummy_input)
mx.eval(model.parameters())

# Benchmark inference
num_repeats = 1000  # Adjust based on how long you want the test to run
times = []

for _ in range(num_repeats):
    start = time.time()
    output = model(dummy_input)  # Forward pass
    mx.eval(output)  # Force execution to measure accurate timing
    times.append(time.time() - start)

# Calculate stats
avg_time = np.mean(times) * 1000  # Convert to milliseconds
std_time = np.std(times) * 1000
fps = batch_size / np.mean(times)  # Frames per second

print(f"\nInference Benchmark (MLX)")
print(f"- Average time per batch: {avg_time:.2f} ms ± {std_time:.2f}")
print(f"- FPS: {fps:.2f}")
print(f"- Device: {mx.default_device()}")