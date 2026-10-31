import logging
from pathlib import Path

import torch

from sprout.modern.inference import feed_tokens
from sprout.sft.sft_config import SFTConfig
from sprout.modern.model import TransformerDecoder, KVCache
from sprout.tokenizer import FastBPETokenizer
from sprout.utils import apply_repetition_penalty, apply_top_k_top_p_filtering


@torch.no_grad()
def generate(model: TransformerDecoder, tokenizer: FastBPETokenizer, config: SFTConfig, max_new_tokens: int, temperature: float,
            top_k: int = None, top_p: float = None, repetition_penalty: float = 1.2, system_prompt: str = None) -> None:
    model.eval()
    running = True
    kv_cache = KVCache(config, batch_size=1)
    im_end_token = tokenizer.special_tokens["<|im_end|>"]
    conversation = [] # All tokens of the conversation so far, needed to re-prefill the sliding window

    if system_prompt is not None:
        # Header and content are encoded separately, exactly like in the SFT data (sft_data_processing.py)
        conversation = tokenizer.encode("<|im_start|>system\n") + tokenizer.encode(f"{system_prompt}<|im_end|>\n")
        feed_tokens(model, kv_cache, conversation, len(conversation), config.device)

    while running:
        prompt = input("Enter your prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            running = False
            break

        print(f"\nYou: {prompt}")
        prompt_tokens = tokenizer.encode("<|im_start|>user\n") + tokenizer.encode(f"{prompt}<|im_end|>\n")
        prompt_tokens += tokenizer.encode("<|im_start|>assistant\n")
        conversation.extend(prompt_tokens)
        next_token_logits = feed_tokens(model, kv_cache, conversation, len(prompt_tokens), config.device) # Prefill the KV cache with the prompt tokens

        turn_tokens = []
        for _ in range(max_new_tokens):
            new_token_logits = apply_repetition_penalty(next_token_logits, turn_tokens, penalty=repetition_penalty)
            new_token_logits /= temperature
            new_token_logits = apply_top_k_top_p_filtering(new_token_logits, top_k=top_k, top_p=top_p)

            probabilities = torch.softmax(new_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            if next_token.item() == im_end_token:
                break
            turn_tokens.append(next_token.item())
            conversation.append(next_token.item())
            next_token_logits = feed_tokens(model, kv_cache, conversation, 1, config.device)

        print(f"\nSprout: {tokenizer.decode(turn_tokens)}")
        closing_tokens = tokenizer.encode("<|im_end|>\n") # Close the turn exactly like in the training data
        conversation.extend(closing_tokens)
        feed_tokens(model, kv_cache, conversation, len(closing_tokens), config.device)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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