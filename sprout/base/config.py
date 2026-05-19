from dataclasses import dataclass, asdict
import math
from pathlib import Path

import torch


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
    log_path: Path = Path(__file__).parent.parent / "logs" / "base_training_log.json"

    def __post_init__(self):
        self.effective_batch_size = self.micro_batch_size * self.grad_accum_steps * self.max_seq_len
        self.max_iterations = self.num_training_tokens // self.effective_batch_size
        self.warmup_steps = max(1, int(self.max_iterations*0.02))
        self.min_learning_rate = self.max_learning_rate * 0.1

    def to_dict(self) -> dict:
        """Converts the Config dataclass to a dictionary and converts all Path objects to strings."""
        config_dict = asdict(self)
        config_dict["save_dir"] = str(self.save_dir)
        config_dict["shards_dir"] = str(self.shards_dir)
        config_dict["log_path"] = str(self.log_path)
        return config_dict

    def convert_paths(self):
        """Converts string paths back to Path objects after loading from a checkpoint."""
        self.save_dir = Path(self.save_dir)
        self.shards_dir = Path(self.shards_dir)
        self.log_path = Path(self.log_path)

    def get_lr(self, step: int) -> float:
        """Computes the learning rate for the current step based on a cosine decay schedule with linear warmup."""
        if step < self.warmup_steps: # Initial warmup
            return self.max_learning_rate * (step + 1) / self.warmup_steps
        if step >= self.max_iterations: # After decay, clamp at min
            return self.min_learning_rate

        progress = (step - self.warmup_steps) / (self.max_iterations - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_learning_rate + coeff * (self.max_learning_rate - self.min_learning_rate)