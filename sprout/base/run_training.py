from pathlib import Path

import torch

from sprout.train import Trainer
from sprout.base.config import Config
from sprout.base.model import TransformerDecoder
from sprout.data import TokenDataLoader


torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "vanilla3" / "checkpoint_step_3500.pt"

config = Config(
    micro_batch_size=10,
    grad_accum_steps=50,
    eval_interval=250,
    checkpoint_interval=500,
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