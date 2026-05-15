import math
from pathlib import Path
from dataclasses import dataclass

from tqdm import trange
import torch
from torch.optim import AdamW
import torch.nn.functional as F

from sprout.base import TransformerDecoder
from sprout.data import TokenDataLoader


@dataclass
class Config:
    # Model hyperparameters
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    max_seq_len: int = 512
    vocab_size: int = 32768
    dropout: float = 0.0
    # Optimizer hyperparameters
    weight_decay: float = 0.1
    max_learning_rate: float = 3e-4
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    # Training hyperparameters
    num_training_tokens: int = 2_500_000_000
    micro_batch_size: int = 64
    grad_accum_steps: int = 16
    grad_clip: float = 1.0
    batches_in_val: int = 100
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    # Directories
    save_dir: Path = Path(__file__).parent.parent / "checkpoints" / "base"
    shards_dir: Path = Path(__file__).parent.parent / "data" / "shards"

    def __post_init__(self):
        self.effective_batch_size = self.micro_batch_size * self.grad_accum_steps * self.max_seq_len
        self.max_iterations = self.num_training_tokens // self.effective_batch_size
        self.warmup_steps = max(1, int(self.max_iterations*0.02))
        self.min_learning_rate = self.max_learning_rate * 0.1

class Trainer:
    """
    Trainer class to handle the training loop, evaluation, and checkpointing.
    Args:
        model: The TransformerDecoder model to train.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        optimizer: The optimizer to use for training.
        config: Configuration object containing hyperparameters and settings.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        assert self.config.checkpoint_interval % self.config.eval_interval == 0, "Checkpoint interval must be a multiple of evaluation interval for consistent logging."

    def train(self) -> None:
        """Main training loop that iterates over the specified number of iterations, performs training steps, evaluates periodically, and saves checkpoints."""
        self._print_summary()
        for step in trange(1, self.config.max_iterations + 1, desc="Training..."):
            train_loss = 0.0
            self.model.train()
            self.optimizer.zero_grad()

            for _ in range(self.config.grad_accum_steps):
                x, y = self.train_loader.get_batch()
                with torch.autocast(device_type=self.config.device, dtype=self.config.dtype):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                    loss = loss / self.config.grad_accum_steps
                loss.backward()
                train_loss += loss.item() # Keep scaled down so it reflects the average loss per micro batch

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            lr = self._get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.optimizer.step()

            if step % self.config.eval_interval == 0:
                eval_loss = self.evaluate()
                print(f"Step {step}: Train Loss = {train_loss:.4f}, Eval Loss = {eval_loss:.4f}")

            if step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(train_loss, eval_loss, step)

    def evaluate(self):
        """Evaluates the model on the validation set and returns the average loss."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for _ in trange(self.config.batches_in_val, desc="Evaluating...", leave=False):
                x, y = self.val_loader.get_batch()
                with torch.autocast(device_type=self.config.device, dtype=self.config.dtype):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                total_loss += loss.item()
        return total_loss / self.config.batches_in_val

    def _save_checkpoint(self, total_train_loss, total_eval_loss, step_num):
        """Saves the model checkpoint to disk, including model state, optimizer state, config, and training statistics."""
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "total_train_loss": total_train_loss,
            "total_eval_loss": total_eval_loss,
            "step_num": step_num
        }
        checkpoint_path = self.config.save_dir / f"checkpoint_step_{step_num}.pt"
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, checkpoint_path)

    def _get_lr(self, step: int) -> float:
        """Computes the learning rate for the current step based on a cosine decay schedule with linear warmup."""
        if step < self.config.warmup_steps: # Initial warmup
            return self.config.max_learning_rate * (step + 1) / self.config.warmup_steps
        if step >= self.config.max_iterations: # After decay, clamp at min
            return self.config.min_learning_rate

        progress = (step - self.config.warmup_steps) / (self.config.max_iterations - self.config.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_learning_rate + coeff * (self.config.max_learning_rate - self.config.min_learning_rate)

    def _print_summary(self):
        """Prints a startup summary of the training and model configuration."""
        border = "=" * 60
        print(border)
        print(f"{'TRAINING PIPELINE INITIALIZED':^60}")
        print(border)
        print(f"\tDevice/Dtype: {self.config.device} / {self.config.dtype}")
        print(f"\tModel Parameters: {self.model.num_parameters():,}")
        print(f"\tMax Iterations: {self.config.max_iterations}")
        print(f"\tWarmup Steps: {self.config.warmup_steps}")
        print(f"\tEffective Batch Size: {self.config.effective_batch_size} tokens per iteration")
        print(border)

if __name__ == "__main__":
    config = Config(
        num_training_tokens=100_000_000,
        micro_batch_size=24,
        grad_accum_steps=42,
        eval_interval=20,
        checkpoint_interval=40,
    )
    model = TransformerDecoder(
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
    ).to(config.device)
    train_loader = TokenDataLoader(config.shards_dir, config.micro_batch_size, config.max_seq_len, config.device, split="train")
    val_loader = TokenDataLoader(config.shards_dir, config.micro_batch_size, config.max_seq_len, config.device, split="val")
    optimizer = AdamW(model.parameters(), lr=config.max_learning_rate, weight_decay=config.weight_decay, betas=config.betas, eps=config.eps)

    trainer = Trainer(model, train_loader, val_loader, optimizer, config)
    trainer.train()