import logging
from pathlib import Path

import torch

from sprout.modern.config import Config
from sprout.modern.model import TransformerDecoder, KVCache
from sprout.tokenizer import FastBPETokenizer


def apply_top_k_top_p_filtering(logits: torch.Tensor, top_k: int = None, top_p: float = None) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k)
        logits[logits < top_k_values[:, -1, None]] = -float("inf")

    if top_p is not None and top_p > 0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone() # Keep the token that crosses the threshold
        sorted_indices_to_remove[:, 0] = False # Always keep the first token
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove) # Scatter back to original indices
        logits[indices_to_remove] = -float("inf")

    return logits

def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    for token_id in set(generated_ids):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits

@torch.no_grad()
def generate(model: TransformerDecoder, tokenizer: FastBPETokenizer, config: Config, prompt: str,
            max_new_tokens: int, temperature: float, top_k: int = None, top_p: float = None, repetition_penalty: float = 1.2) -> str:
    model.eval()
    print(prompt, end="", flush=True)
    token_ids_list = tokenizer.encode(prompt)
    token_ids = torch.tensor(token_ids_list, dtype=torch.long, device=config.device).unsqueeze(0) # Shape: [1, seq_len]
    kv_cache = KVCache(config, batch_size=1)
    model_output = model(token_ids, kv_cache=kv_cache) # Prefill the KV cache with the prompt tokens

    generated_tokens = list(token_ids_list)
    for _ in range(max_new_tokens):
        next_token_logits = model_output[:, -1, :].clone()
        new_token_logits = apply_repetition_penalty(next_token_logits, generated_tokens, penalty=repetition_penalty)
        new_token_logits /= temperature
        new_token_logits = apply_top_k_top_p_filtering(new_token_logits, top_k=top_k, top_p=top_p)

        probabilities = torch.softmax(new_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        if next_token.item() == tokenizer.eot_token:
            break

        generated_tokens.append(next_token.item())
        model_output = model(next_token, kv_cache=kv_cache)
        print(tokenizer.decode([next_token.item()]), end="", flush=True) # Stream generated tokens

    return tokenizer.decode(generated_tokens)

if __name__ == "__main__":
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "modern" / "sapling" / "checkpoint_step_19500.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    logging.getLogger("torch._inductor").setLevel(logging.ERROR) # Suppressmax_autotune_gemm mode warning

    config = Config(**checkpoint["config"])
    config.convert_paths()

    model = TransformerDecoder(config).to(config.device)

    model = model.to(torch.bfloat16)
    model = torch.compile(model, options={"max_autotune": False, "max_autotune_gemm": False})
    model.load_state_dict(checkpoint["model"])
    generated_text = generate(
        model=model,
        tokenizer=FastBPETokenizer.load(config.tokenizer_path),
        config=config,
        prompt=r"""The water cycle describes how water moves through the environment. First,""",
        max_new_tokens=100,
        temperature=0.7,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.15
    )