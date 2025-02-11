import argparse
import mlx.core as mx
from models.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from trainers.mlx_trainer import MLXTrainer
from dataset import get_cifar10
from utils.visualization import plot_predictions, show_dataset_samples, generate_performance_plots
import time
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet20",
        choices=[f"resnet{d}" for d in [20, 32, 44, 56, 110, 1202]],
        help="model architecture",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
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
    train_data, test_data = get_cifar10(args.batch_size)

    metrics = {
        'config': vars(args),
        'train_metrics': [],
        'test_accs': [],
        'timestamps': []
    }
    start_time = time.time()
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # Train
        train_metrics = trainer.train_epoch(train_data, epoch)
        test_acc = trainer.evaluate(test_data)
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['acc'],
            'throughput': train_metrics['throughput'],
            'gpu_memory_avg': sum(train_metrics['gpu_memories'])/len(train_metrics['gpu_memories']),
            'gpu_memory_max': max(train_metrics['gpu_memories']),
            'epoch_time': train_metrics['epoch_time'],
            'batch_times': train_metrics['batch_times'],
            'test_acc': test_acc,
        }
        metrics['train_metrics'].append(epoch_metrics)
        metrics['timestamps'].append(time.time() - start_time)

        
        # Evaluate
        print(f"Epoch: {epoch} | Test acc {test_acc:.3f}")
        
        # # Visualize predictions periodically
        # if epoch % 5 == 0:
        #     test_data.reset()
        #     fig = plot_predictions(model, next(test_data))
        #     fig.savefig(f"predictions_epoch_{epoch}.png")
        
        # Reset iterators
        train_data.reset()
        test_data.reset()
        
    # Save checkpoint
    trainer.save_checkpoint(f"checkpoint_epoch_{epoch}.pkl")
    metrics_path = Path("metrics") / f"metrics_{args.arch}_{int(start_time)}.json"
    metrics_path.parent.mkdir(exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    generate_performance_plots(metrics, save_dir="figures")

if __name__ == "__main__":
    main()
