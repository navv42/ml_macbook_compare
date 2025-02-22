import time
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pickle
from utils.visualization import show_dataset_samples, plot_predictions

class MLXTrainer:
    def __init__(self, model, learning_rate=0.01, weight_decay=5e-4, momentum=0.9):
        self.model = model
        self.optimizer = optim.SGD(learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = optim.cosine_decay(learning_rate, 50 * 39)
        
    def load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        self.model.state.update(checkpoint["model"])
        self.optimizer.state.update(checkpoint["optimizer"])
        
    def save_checkpoint(self, checkpoint_path):
        checkpoint = {
            "model": self.model.state,
            "optimizer": self.optimizer.state
        }
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
            
    def train_epoch(self, train_iter, epoch):
        def train_step(model, inp, tgt):
            output = model(inp)
            loss = mx.mean(nn.losses.cross_entropy(output, tgt))
            acc = mx.mean(mx.argmax(output, axis=1) == tgt)
            return loss, acc

        losses = []
        accs = []
        throughputs = []
        gpu_memories = []
        batch_times = []
        state = [self.model.state, self.optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(inp, tgt):
            train_step_fn = nn.value_and_grad(self.model, train_step)
            (loss, acc), grads = train_step_fn(self.model, inp, tgt)
            self.optimizer.update(self.model, grads)
            return loss, acc
        mx.metal.reset_peak_memory()
        peak_cache = 0
        for batch_counter, batch in enumerate(train_iter):
            x = mx.array(batch["image"])
            y = mx.array(batch["label"])
            cache_memory = mx.metal.get_cache_memory() / (1024 ** 3)
            peak_cache = max(peak_cache, cache_memory)
            tic = time.perf_counter()
            loss, acc = step(x, y)
            mx.eval(state)
            toc = time.perf_counter()
            
            loss_val = loss.item()
            acc_val = acc.item()
            losses.append(loss_val)
            accs.append(acc_val)
            throughput = x.shape[0] / (toc - tic)
            throughputs.append(throughput)
            batch_time = toc - tic
            batch_times.append(batch_time)
            
            if batch_counter % 20 == 0:
                print(
                    " | ".join(
                        (
                            f"Epoch {epoch:02d} [{batch_counter:03d}]",
                            f"Train loss {loss_val:.3f}",
                            f"Train acc {acc_val:.3f}",
                            f"Throughput: {throughput:.2f} images/sec",
                            f"Cache memory peak: {peak_cache:.2f} GB",
                            f"GPU memory peak: {mx.metal.get_peak_memory() / (1024 ** 3):.2f} GB",
                        )
                    )
                )
        peak_mem = mx.metal.get_peak_memory() / (1024 ** 3)

        if epoch < 5:
            self.optimizer.learning_rate = self.scheduler(epoch) * (epoch / 5)

        return {
            'loss': mx.mean(mx.array(losses)).item(),
            'acc': mx.mean(mx.array(accs)).item(),
            'throughput': mx.mean(mx.array(throughputs)).item(),
            'gpu_mem_peak': peak_mem,
            'cache_mem_peak': peak_cache,
            'batch_times': batch_times,
            'batch_time_avg': mx.mean(mx.array(batch_times)).item(),
            'epoch_time': sum(batch_times),
        }

    def evaluate(self, test_iter):
        def eval_fn(model, inp, tgt):
            return mx.mean(mx.argmax(model(inp), axis=1) == tgt)
        # show_dataset_samples(test_iter, num_images=12)
        # plot_predictions(self.model, test_iter)
        accs = []
        for batch in test_iter:
            x = mx.array(batch["image"])
            y = mx.array(batch["label"])
            acc = eval_fn(self.model, x, y)
            accs.append(acc.item())
            
        return mx.mean(mx.array(accs)).item()

    def fit(self, train_data, test_data, epochs):
        """
        Runs the training and evaluation loop.
        Returns a metrics dictionary containing per-epoch metrics and timestamps.
        """
        metrics = {'train_metrics': [], 'timestamps': []}
        start_time = time.time()
        for epoch in range(epochs):
            # Train for one epoch and evaluate
            epoch_metrics = self.train_epoch(train_data, epoch)
            test_acc = self.evaluate(test_data)
            epoch_metrics.update({'epoch': epoch, 'test_acc': test_acc})
            metrics['train_metrics'].append(epoch_metrics)
            metrics['timestamps'].append(time.time() - start_time)
            print(f"Epoch: {epoch} | Test acc {test_acc:.3f}")
            
            # Reset data iterators for the next epoch
            train_data.reset()
            test_data.reset()
        return metrics
