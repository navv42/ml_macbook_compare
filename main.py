import argparse
import time
import json
from pathlib import Path
import mlx.core as mx
from models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from trainers.mlx_trainer import MLXTrainer
from dataset import get_cifar10, get_cifar100
from utils.visualization import generate_performance_plots

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet44",
        choices=[f"resnet{d}" for d in [20, 32, 44, 56, 110, 1202]],
        help="model architecture",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--checkpoint", type=str, help="path to load checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    
    # Set random seed
    mx.random.seed(args.seed)
    
    # Initialize model and trainer
    model = globals()[args.arch]()
    trainer = MLXTrainer(model, learning_rate=args.lr)
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    print(f"Number of params: {model.num_params() / 1e6:.4f}M")
    print(f"Using device: {mx.default_device()}")
    
    # Get data
    train_data, test_data = get_cifar100(args.batch_size)
    
    # Train and collect metrics using the new fit() method
    metrics = trainer.fit(train_data, test_data, epochs=args.epochs)
    
    # Save checkpoint and metrics after training
    trainer.save_checkpoint(f"checkpoint_epoch_{args.epochs - 1}.pkl")
    metrics_path = Path("metrics") / f"metrics_{args.arch}_{int(time.time())}.json"
    metrics_path.parent.mkdir(exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'train_metrics': metrics['train_metrics'],
            'timestamps': metrics['timestamps']
        }, f)
    print(metrics)
    generate_performance_plots(metrics, save_dir="figures")

if __name__ == "__main__":
    main()

