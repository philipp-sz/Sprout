import logging
from pathlib import Path

import torch

from sprout.sft.sft_config import SFTConfig
from sprout.modern.model import TransformerDecoder, KVCache
from sprout.tokenizer import FastBPETokenizer
from sprout.utils import apply_repetition_penalty, apply_top_k_top_p_filtering


@torch.no_grad()
def generate(model: TransformerDecoder, tokenizer: FastBPETokenizer, config: SFTConfig, max_new_tokens: int, temperature: float,
            top_k: int = None, top_p: float = None, repetition_penalty: float = 1.2, eot_token: int = 32769, system_prompt: str = None) -> None:
    model.eval()
    running = True
    kv_cache = KVCache(config, batch_size=1)

    if system_prompt is not None:
        system_message = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        system_tokens = tokenizer.encode(system_message)
        system_token_tensor = torch.tensor(system_tokens, dtype=torch.long, device=config.device).unsqueeze(0)
        model_output = model(system_token_tensor, kv_cache=kv_cache)

    while running:
        prompt = input("Enter your prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            running = False
            break

        print(f"\nYou: {prompt}")
        prompt_text = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_token_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=config.device).unsqueeze(0) # Shape: [1, seq_len]
        model_output = model(prompt_token_tensor, kv_cache=kv_cache) # Prefill the KV cache with the prompt tokens

        turn_tokens = []
        for _ in range(max_new_tokens):
            next_token_logits = model_output[:, -1, :].clone()
            new_token_logits = apply_repetition_penalty(next_token_logits, turn_tokens, penalty=repetition_penalty)
            new_token_logits /= temperature
            new_token_logits = apply_top_k_top_p_filtering(new_token_logits, top_k=top_k, top_p=top_p)

            probabilities = torch.softmax(new_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            turn_tokens.append(next_token.item())
            model_output = model(next_token, kv_cache=kv_cache)

            if next_token.item() == eot_token:
                break

        if turn_tokens and turn_tokens[-1] == eot_token:
            out_tokens = turn_tokens[:-1]
        elif turn_tokens[-1] != eot_token:
            out_tokens = turn_tokens

        response = tokenizer.decode(turn_tokens)
        out_response = tokenizer.decode(out_tokens)
        print(f"\nSprout: {out_response}")

        if not response.strip().endswith("<|im_end|>"):
            added_text = "<|im_end|>\n"
            response += added_text
        else:
            added_text = "\n"
            response += added_text
        missing_tokens = tokenizer.encode(added_text)
        missing_token_tensor = torch.tensor(missing_tokens, dtype=torch.long, device=config.device).unsqueeze(0)
        _ = model(missing_token_tensor, kv_cache=kv_cache)

if __name__ == "__main__":
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "sft" / "sft_200M" / "model_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    logging.getLogger("torch._inductor").setLevel(logging.ERROR) # Suppressmax_autotune_gemm mode warning

    config = SFTConfig(**checkpoint["config"])
    config.convert_paths()

    model = TransformerDecoder(config).to(config.device)

    model = model.to(torch.bfloat16)
    model = torch.compile(model, options={"max_autotune": False, "max_autotune_gemm": False})
    model.load_state_dict(checkpoint["model"])
    generate(
        model=model,
        tokenizer=FastBPETokenizer.load(config.tokenizer_path),
        config=config,
        max_new_tokens=500,
        temperature=0.5,
        top_k=50,
        top_p=0.80,
        repetition_penalty=1.20,
        system_prompt="Your name is Sprout. You are Sprout, a helpful and concise AI assistant. Follow user instructions strictly. If you do not know, say 'I don't know'."
    )