from dataclasses import dataclass, asdict
import math
from pathlib import Path

import torch


@dataclass
class Config:
    # Model hyperparameters
    d_model: int = 1024
    num_heads: int = 16
    num_kv_heads: int = 4
    num_layers: int = 8
    max_seq_len: int = 1024
    vocab_size: int = 32768
    mlp_expansion: float = 4.75
    dropout: float = 0.0
    rope_base_freq: float = 5_000.0
    # Optimizer hyperparameters
    muon_momentum: float = 0.95
    weight_decay: float = 0.1
    max_learning_rate: float = 1e-3
    min_learning_rate: float = 0.0
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    # Training hyperparameters
    num_training_tokens: int = 15_000_000_000
    micro_batch_size: int = 64
    grad_accum_steps: int = 16
    grad_clip: float = 1.0
    batches_in_val: int = 100
    eval_benchmark_samples: int = 1000
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    warmup: float = 0.02
    decay: float = 0.10
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    # Directories
    save_dir: Path = Path(__file__).parent.parent / "checkpoints" / "modern" / "sapling"
    shards_dir: Path = Path(__file__).parent.parent / "data" / "shards" / "train_shards_15b"
    log_path: Path = Path(__file__).parent.parent / "logs" / "train" / "sapling_training_log.json"
    tokenizer_path: Path = Path(__file__).parent.parent / "tokenizer" / "states" / "bpe_mixed_32k_20k_state.json"

    def __post_init__(self):
        self.head_dim = self.d_model // self.num_heads
        self.effective_batch_size = self.micro_batch_size * self.grad_accum_steps * self.max_seq_len
        self.max_iterations = self.num_training_tokens // self.effective_batch_size
        self.warmup_steps = max(1, int(self.max_iterations * self.warmup))
        self.decay_steps = self.max_iterations - max(1, int(self.max_iterations * self.decay))

    def to_dict(self) -> dict:
        """Converts the Config dataclass to a dictionary and converts all Path objects to strings."""
        config_dict = asdict(self)
        config_dict["save_dir"] = str(self.save_dir)
        config_dict["shards_dir"] = str(self.shards_dir)
        config_dict["log_path"] = str(self.log_path)
        config_dict["tokenizer_path"] = str(self.tokenizer_path)
        return config_dict

    def convert_paths(self):
        """Converts string paths back to Path objects after loading from a checkpoint."""
        self.save_dir = Path(self.save_dir)
        self.shards_dir = Path(self.shards_dir)
        self.log_path = Path(self.log_path)
        self.tokenizer_path = Path(self.tokenizer_path)

    def get_lr(self, step: int) -> float:
        """Computes the learning rate for the current step based on a cosine decay schedule with linear warmup."""
        if step < self.warmup_steps: # Initial warmup
            return self.max_learning_rate * (step + 1) / self.warmup_steps

        if step >= self.max_iterations: # Ensure learning rate does not cause divergence after max iterations
            return self.min_learning_rate       

        if step >= self.decay_steps:
            progress = (step - self.decay_steps) / (self.max_iterations - self.decay_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
            return coeff * self.max_learning_rate
        
        return self.max_learning_rate