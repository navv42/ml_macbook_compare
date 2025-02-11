import time
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pickle

class MLXTrainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)

        
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
        samples_per_sec = []
        state = [self.model.state, self.optimizer.state]
        gpu_memories = []
        batch_times = []

        @partial(mx.compile, inputs=state, outputs=state)
        def step(inp, tgt):
            train_step_fn = nn.value_and_grad(self.model, train_step)
            (loss, acc), grads = train_step_fn(self.model, inp, tgt)
            self.optimizer.update(self.model, grads)
            return loss, acc

        for batch_counter, batch in enumerate(train_iter):
            x = mx.array(batch["image"])
            y = mx.array(batch["label"])
            gpu_memory = mx.metal.get_active_memory() / (1024 ** 2)
            gpu_memories.append(gpu_memory)
            tic = time.perf_counter()
            loss, acc = step(x, y)
            
            mx.eval(state)
            toc = time.perf_counter()
            
            loss = loss.item()
            acc = acc.item()
            losses.append(loss)
            accs.append(acc)
            throughput = x.shape[0] / (toc - tic)
            samples_per_sec.append(throughput)
            batch_time = toc - tic
            batch_times.append(batch_time)
            
            if batch_counter % 10 == 0:
                print(
                    " | ".join(
                        (
                            f"Epoch {epoch:02d} [{batch_counter:03d}]",
                            f"Train loss {loss:.3f}",
                            f"Train acc {acc:.3f}",
                            f"Throughput: {throughput:.2f} images/second",
                            f"GPU memory: {gpu_memory:.2f} MB",
                        )
                    )
                )

        return {
            'loss': mx.mean(mx.array(losses)).item(),
            'acc': mx.mean(mx.array(accs)).item(),
            'throughput': mx.mean(mx.array(samples_per_sec)).item(),
            'gpu_memories': gpu_memories,
            'batch_times': batch_times,
            'epoch_time': sum(batch_times),
        }

    def evaluate(self, test_iter):
        def eval_fn(model, inp, tgt):
            return mx.mean(mx.argmax(model(inp), axis=1) == tgt)
            
        accs = []
        for batch in test_iter:
            x = mx.array(batch["image"])
            y = mx.array(batch["label"])
            acc = eval_fn(self.model, x, y)
            accs.append(acc.item())
            
        return mx.mean(mx.array(accs)).item()
