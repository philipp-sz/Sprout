from pathlib import Path

import torch

from sprout.train import Trainer
from sprout.modern.config import Config
from sprout.modern.model import TransformerDecoder
from sprout.data import TokenDataLoader


torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "modern" / "checkpoint_step_60.pt"

config = Config(
    micro_batch_size=24,
    grad_accum_steps=42,
    eval_interval=10,
    checkpoint_interval=10,
)
train_loader = TokenDataLoader(config.shards_dir, config.micro_batch_size, config.max_seq_len, config.device, split="train")
val_loader = TokenDataLoader(config.shards_dir, config.micro_batch_size, config.max_seq_len, config.device, split="val")

if checkpoint_path and checkpoint_path.exists():
    trainer = Trainer.from_checkpoint(checkpoint_path, TransformerDecoder, Config, train_loader, val_loader)
else:
    model = TransformerDecoder(config).to(config.device)
    model = torch.compile(model)
    trainer = Trainer(model, train_loader, val_loader, config)
trainer.train()