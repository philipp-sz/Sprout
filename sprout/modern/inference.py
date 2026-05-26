from pathlib import Path

import torch

from sprout.modern.config import Config
from sprout.modern.model import TransformerDecoder, KVCache
from sprout.tokenizer import FastBPETokenizer


@torch.no_grad()
def generate(model: TransformerDecoder, tokenizer: FastBPETokenizer, config: Config, prompt: str,
            max_new_tokens: int, temperature: float, top_k: int = None, top_p: float = None) -> str:
    model.eval()
    token_ids_list = tokenizer.encode(prompt)
    token_ids = torch.tensor(token_ids_list, dtype=torch.long, device=config.device).unsqueeze(0) # Shape: [1, seq_len]
    kv_cache = KVCache(config, batch_size=1)
    model_output = model(token_ids, kv_cache=kv_cache) # Prefill the KV cache with the prompt tokens

    generated_tokens = list(token_ids_list)
    for _ in range(max_new_tokens):
        new_token_logits = model_output[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None and top_k > 0:
            top_k_values, _ = torch.topk(new_token_logits, top_k)
            new_token_logits[new_token_logits < top_k_values[:, -1, None]] = -float("inf")

        if top_p is not None and top_p > 0:
            sorted_logits, sorted_indices = torch.sort(new_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone() # Keep the token that crosses the threshold
            sorted_indices_to_remove[:, 0] = False # Always keep the first token
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove) # Scatter back to original indices
            new_token_logits[indices_to_remove] = -float("inf")

        probabilities = torch.softmax(new_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        if next_token.item() == tokenizer.eot_token:
            break

        generated_tokens.append(next_token.item())
        model_output = model(next_token, kv_cache=kv_cache)


    return tokenizer.decode(generated_tokens)

if __name__ == "__main__":
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "modern" / "base" / "model_final.pt"
    tokenizer_path = Path(__file__).parent.parent / "tokenizer" / "bpe_tokenizer_state.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = Config(**checkpoint["config"])
    config.convert_paths()

    model = TransformerDecoder(config).to(config.device)

    model = model.to(torch.bfloat16)
    model = torch.compile(model)
    model.load_state_dict(checkpoint["model"])
    generated_text = generate(
        model=model,
        tokenizer=FastBPETokenizer.load(tokenizer_path),
        config=config,
        prompt=r"""The water cycle describes how water moves through the environment. First, the sun heats up water in rivers, lakes, or oceans, causing it to evaporate and turn into vapor. This vapor rises into the air and cools down, forming clouds through condensation. When the clouds become heavy with water droplets, they release the water as precipitation, such as rain or snow. The precipitation then collects in bodies of water, and the cycle starts again.""",
        max_new_tokens=100,
        temperature=0.6,
        top_k=None,
        top_p=0.9,
    )
    print(generated_text)