import logging
from pathlib import Path

import torch

from sprout.base.config import Config
from sprout.base.model import TransformerDecoder
from sprout.tokenizer import FastBPETokenizer
from sprout.utils import apply_repetition_penalty, apply_top_k_top_p_filtering


@torch.no_grad()
def generate(model: TransformerDecoder, tokenizer: FastBPETokenizer, config: Config, prompt: str,
            max_new_tokens: int, temperature: float, top_k: int = None, top_p: float = None, repetition_penalty: float = 1.2) -> str:
    model.eval()
    print(prompt, end="", flush=True)
    token_ids = tokenizer.encode(prompt)
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=config.device).unsqueeze(0) # Shape: [1, seq_len]

    for _ in range(max_new_tokens):
        if token_ids.size(1) > config.max_seq_len:
            token_ids = token_ids[:, -config.max_seq_len:]

        model_output = model(token_ids)
        next_token_logits = model_output[:, -1, :].clone()
        new_token_logits = apply_repetition_penalty(next_token_logits, token_ids.squeeze(0).tolist(), penalty=repetition_penalty)
        new_token_logits /= temperature
        new_token_logits = apply_top_k_top_p_filtering(new_token_logits, top_k=top_k, top_p=top_p)

        probabilities = torch.softmax(new_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        if next_token.item() == tokenizer.eot_token:
            break
        token_ids = torch.cat((token_ids, next_token), dim=-1)
        print(tokenizer.decode([next_token.item()]), end="", flush=True)

    return tokenizer.decode(token_ids.squeeze(0).tolist())

if __name__ == "__main__":
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "vanilla3" / "model_final.pt"
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
        max_new_tokens=50,
        temperature=0.8,
        top_k=None,
        top_p=0.9,
    )
    print(generated_text)