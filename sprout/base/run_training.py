from pathlib import Path

import torch
from torch.optim import AdamW

from sprout.base.train import Trainer, Config
from sprout.base.model import TransformerDecoder
from sprout.data import TokenDataLoader


torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

checkpoint_path = None # Path(__file__).parent.parent / "checkpoints" / "base" / "checkpoint_step_2.pt"

config = Config(
    micro_batch_size=24,
    grad_accum_steps=42,
    eval_interval=250,
    checkpoint_interval=500,
)
train_loader = TokenDataLoader(config.shards_dir, config.micro_batch_size, config.max_seq_len, config.device, split="train")
val_loader = TokenDataLoader(config.shards_dir, config.micro_batch_size, config.max_seq_len, config.device, split="val")

if checkpoint_path and checkpoint_path.exists():
    trainer = Trainer.from_checkpoint(checkpoint_path, train_loader, val_loader)
else:
    model = TransformerDecoder(
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        dropout=config.dropout
    ).to(config.device)
    model = torch.compile(model)
    optimizer = AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
        fused=True
    )
    trainer = Trainer(model, train_loader, val_loader, optimizer, config)
trainer.train()