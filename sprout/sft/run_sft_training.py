from pathlib import Path

import torch

from sprout.train import Trainer
from sprout.sft import SFTConfig
from sprout.modern.model import TransformerDecoder
from sprout.sft import SFTDataLoader
from sprout.tokenizer import FastBPETokenizer


torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

tokenizer_path = Path(__file__).parent.parent / "tokenizer" / "states" / "bpe_mixed_32k_20k_instruct_state.json"
eot_token = FastBPETokenizer.load(tokenizer_path).eot_token
checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "modern" / "sapling" / "model_final_instruct.pt"

config = SFTConfig(
    micro_batch_size=8,
    grad_accum_steps=16,
    eval_interval=250,
    checkpoint_interval=500,
)
train_loader = SFTDataLoader(config.shards_dir, config.micro_batch_size, config.max_seq_len, config.device, split="train", eot_token=eot_token)
val_loader = SFTDataLoader(config.shards_dir, config.micro_batch_size, config.max_seq_len, config.device, split="val", eot_token=eot_token)

trainer = Trainer.from_checkpoint(checkpoint_path, TransformerDecoder, config, train_loader, val_loader, is_final=True)
trainer.train()


