import itertools
import subprocess
import os
from tqdm import tqdm


DEVICE = "m3pro"
# Define your experiment parameters. Make sure these match the argument names in main.py.
batch_sizes = [128, 256, 512, 1024]
# Map model names to the expected argument values (e.g., lower-case without hyphen)
models = {"ResNet-20": "resnet20", "ResNet-32": "resnet32", "ResNet-56": "resnet56", "ResNet-110": "resnet110"}
learning_rates = [0.1]
datasets = ["CIFAR-10"] 

# Create all experiment combinations
experiment_configs = list(itertools.product(batch_sizes, models.keys(), learning_rates, datasets))
print(f"Total experiments: {len(experiment_configs)}")

for bs, model, lr, dataset in tqdm(experiment_configs, desc="Experiments"):

    arch = models[model]
    filename = f"metrics/{DEVICE}/{arch}_{bs}_{lr}.json"
    if os.path.exists(filename):
        print(f"Skipping {filename} (already exists)")
        continue
    # Build the command line arguments
    # Note: Adjust argument names if needed; here, dataset is not used in main.py, so you might extend main.py to accept it.
    cmd = [
        "python", "main.py",
        "--batch_size", str(bs),
        "--arch", models[model],    
        "--lr", str(lr),
        "--epochs", str(40),
        "--seed", str(1),
        "--device", DEVICE,
        # You can add more flags as needed, e.g., setting seed or CPU flag
    ]
    tqdm.write("Running: " + " ".join(cmd))
    # Launch the experiment and wait for it to finish.
    subprocess.run(cmd, check=True)
