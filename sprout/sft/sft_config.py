from dataclasses import dataclass
from pathlib import Path
import math

from sprout.modern.config import Config


@dataclass
class SFTConfig(Config):
    vocab_size: int = 32770

    max_learning_rate: float = 5e-4
    min_learning_rate: float = 0.0

    num_training_tokens: int = 200_000_000
    micro_batch_size: int = 8
    grad_accum_steps: int = 16
    warmup: float = 0.02
    eval_interval: int = 250
    checkpoint_interval: int = 500

    save_dir: Path = Path(__file__).parent.parent / "checkpoints" / "sft" / "sft_200M"
    shards_dir: Path = Path(__file__).parent.parent / "data" / "shards" / "sft_shards_100M"
    log_path: Path = Path(__file__).parent.parent / "logs" / "train" / "sft_200M_log.json"
    tokenizer_path: Path = Path(__file__).parent.parent / "tokenizer" / "bpe_mixed_32k_20k_v4_instruct_state.json"

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_learning_rate * (step + 1) / self.warmup_steps
        if step >= self.max_iterations:
            return self.min_learning_rate

        progress = (step - self.warmup_steps) / (self.max_iterations - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_learning_rate + coeff * (self.max_learning_rate - self.min_learning_rate)