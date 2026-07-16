from pathlib import Path

from sprout.utils import extend_tokenizer_state


tokenizer_state_path = Path(__file__).parent / "bpe_mixed_32k_20k_v4_state.json"
new_tokenizer_state_path = Path(__file__).parent / "bpe_mixed_32k_20k_v4_instruct_state.json"
new_special_tokens = ["<|im_start|>", "<|im_end|>"]
extend_tokenizer_state(tokenizer_state_path, new_tokenizer_state_path, new_special_tokens)