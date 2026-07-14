from pathlib import Path

import torch

from sprout.modern.model import TransformerDecoder
from sprout.utils import resize_model_embeddings
from sprout.modern.config import Config


old_model_path = Path(__file__).parent.parent / "checkpoints" / "modern" / "sapling" / "model_final.pt"
checkpoint = torch.load(old_model_path, map_location="cpu")
config = Config(**checkpoint["config"])
config.convert_paths()

model = TransformerDecoder(config).to(config.device)
model = model.to(torch.bfloat16)
model = torch.compile(model, options={"max_autotune": False, "max_autotune_gemm": False})
model.load_state_dict(checkpoint["model"])

model_embeddings = model.token_embed
new_embeddings = resize_model_embeddings(model_embeddings, num_new_tokens=2) # Tokens for <|im_start|> and <|im_end|>
model.token_embed = new_embeddings

_, embedding_dim = new_embeddings.weight.shape
new_lm_head = torch.nn.Linear(embedding_dim, new_embeddings.num_embeddings, bias=False)
new_lm_head.weight = model.token_embed.weight
model.lm_head = new_lm_head
config.vocab_size = new_embeddings.num_embeddings

new_model_path = Path(__file__).parent.parent / "checkpoints" / "modern"/ "sapling" / "model_final_instruct.pt"
torch.save({
    "model": model.state_dict(),
    "config": config.to_dict(),
}, new_model_path)