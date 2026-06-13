import json
from pathlib import Path

import torch


def extend_tokenizer_state(tokenizer_state_path: Path, new_tokenizer_state_path: Path, new_special_tokens: list[str]):
    with open(tokenizer_state_path, "r") as f:
        tokenizer_state = json.load(f)

    existing_special_tokens = tokenizer_state["special_tokens"]
    max_token_id = max(tokenizer_state["merges"].values()) + len(existing_special_tokens)
    for i, token in enumerate(new_special_tokens, start=1):
        if token not in existing_special_tokens:
            new_id = max_token_id + i
            tokenizer_state["special_tokens"][token] = new_id
            tokenizer_state["vocab"][str(new_id)] = token.encode("utf-8").hex()

    with open(new_tokenizer_state_path, "w") as f:
        json.dump(tokenizer_state, f, indent=4)

def resize_model_embeddings(old_embeddings: torch.nn.Embedding, num_new_tokens: int) -> torch.nn.Embedding:
    old_num_tokens, embedding_dim = old_embeddings.weight.shape
    new_num_tokens = old_num_tokens + num_new_tokens
    new_embeddings = torch.nn.Embedding(new_num_tokens, embedding_dim)

    with torch.no_grad():
        new_embeddings.weight[:old_num_tokens].copy_(old_embeddings.weight)
        mean_weight = old_embeddings.weight.mean(dim=0, keepdim=True)
        new_embeddings.weight[old_num_tokens:] = mean_weight.repeat(num_new_tokens, 1)
    return new_embeddings