import logging
from pathlib import Path

import torch
from torch import Tensor

from sprout.modern.config import Config
from sprout.modern.model import TransformerDecoder, KVCache
from sprout.tokenizer import FastBPETokenizer
from sprout.utils import apply_repetition_penalty, apply_top_k_top_p_filtering


@torch.no_grad()
def feed_tokens(model: TransformerDecoder, kv_cache: KVCache, token_ids: list[int], num_new_tokens: int, device: str) -> Tensor:
    """Feeds the last num_new_tokens of token_ids through the model and returns the logits of the last position."""
    assert num_new_tokens > 0, "At least one new token is needed."
    new_tokens = token_ids[-num_new_tokens:]
    max_seq_len = kv_cache.max_seq_len
    if kv_cache.current_step + num_new_tokens > max_seq_len:
        # Sliding window: restart the cache with the most recent tokens, so positions stay within the trained range.
        # Keeping half a context means this full re-prefill only happens every ~max_seq_len / 2 tokens.
        window = min(max_seq_len, max(max_seq_len // 2, num_new_tokens))
        new_tokens = token_ids[-window:]
        kv_cache.reset()

    input_ids = torch.tensor(new_tokens, dtype=torch.long, device=device).unsqueeze(0) # Shape: [1, seq_len]
    return model(input_ids, kv_cache=kv_cache)[:, -1, :]

@torch.no_grad()
def generate(model: TransformerDecoder, tokenizer: FastBPETokenizer, config: Config, prompt: str,
            max_new_tokens: int, temperature: float, top_k: int = None, top_p: float = None, repetition_penalty: float = 1.2) -> str:
    model.eval()
    print(prompt, end="", flush=True)
    generated_tokens = tokenizer.encode(prompt)
    kv_cache = KVCache(config, batch_size=1)
    next_token_logits = feed_tokens(model, kv_cache, generated_tokens, len(generated_tokens), config.device) # Prefill the KV cache with the prompt tokens

    for _ in range(max_new_tokens):
        new_token_logits = apply_repetition_penalty(next_token_logits, generated_tokens, penalty=repetition_penalty)
        new_token_logits /= temperature
        new_token_logits = apply_top_k_top_p_filtering(new_token_logits, top_k=top_k, top_p=top_p)

        probabilities = torch.softmax(new_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        if next_token.item() == tokenizer.eot_token:
            break

        generated_tokens.append(next_token.item())
        next_token_logits = feed_tokens(model, kv_cache, generated_tokens, 1, config.device)
        print(tokenizer.decode([next_token.item()]), end="", flush=True) # Stream generated tokens

    return tokenizer.decode(generated_tokens)

if __name__ == "__main__":
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "modern" / "sapling" / "model_final.pt"
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