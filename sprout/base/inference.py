import logging
from pathlib import Path

import torch

from sprout.base.config import Config
from sprout.base.model import TransformerDecoder
from sprout.tokenizer import FastBPETokenizer


@torch.no_grad()
def generate(model: TransformerDecoder, tokenizer: FastBPETokenizer, config: Config, prompt: str,
            max_new_tokens: int, temperature: float, top_k: int = None, top_p: float = None) -> str:
    model.eval()
    print(prompt, end="", flush=True)
    token_ids = tokenizer.encode(prompt)
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=config.device).unsqueeze(0) # Shape: [1, seq_len]

    for _ in range(max_new_tokens):
        if token_ids.size(1) > config.max_seq_len:
            token_ids = token_ids[:, -config.max_seq_len:]

        model_output = model(token_ids)
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
        token_ids = torch.cat((token_ids, next_token), dim=-1)
        print(tokenizer.decode([next_token.item()]), end="", flush=True)

    return tokenizer.decode(token_ids.squeeze(0).tolist())

if __name__ == "__main__":
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "base" / "model_final.pt"
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
        prompt="""The water cycle describes how water moves through the environment. First,""",
        max_new_tokens=100,
        temperature=0.8,
        top_k=None,
        top_p=0.9,
    )
    print(generated_text)