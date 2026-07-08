import numpy as np

from sprout.data import DatasetProcessor
from sprout.data.data_processing import _worker_init
from sprout.tokenizer.fast_bpe_tokenizer import FastBPETokenizer


_worker_tokenizer = None

def _worker_init(tokenizer_state_path: str) -> None:
    global _worker_tokenizer
    _worker_tokenizer = FastBPETokenizer.load(tokenizer_state_path)

def _tokenize_mask_doc(conversation_turns: list[dict[str, str]]) -> dict[str, list[int]]:
    """Tokenizes a conversation turn-by-turn and create a mask for each turn."""
    mask = []
    tokenized_conv = []
    for turn in conversation_turns:
        role_text = f"<|im_start|>{turn['role']}\n"
        context_text = f"{turn['content']}<|im_end|>\n"
        role_tokens = _worker_tokenizer.encode(role_text)
        context_tokens = _worker_tokenizer.encode(context_text)
        tokenized_conv.extend(role_tokens + context_tokens)

        context_mask = [1] * len(context_tokens) if turn["role"] == "assistant" else [0] * len(context_tokens)
        turn_mask = [0] * len(role_tokens) + context_mask
        mask.extend(turn_mask)
    tokenized_conv.append(_worker_tokenizer.eot_token)
    mask.append(0)  # Mask the EOT token
    return {"tokens": tokenized_conv, "mask": mask}

class SFTDatasetProcessor(DatasetProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, tokenize_fn=_tokenize_mask_doc, worker_init=_worker_init, **kwargs)

    def _save_shard(self, metadata: dict, tokens_to_process: int, buffers: dict[str, list[int]], rows_consumed: int) -> None:
        """Saves a shard of tokenized data to disk and updates metadata with the new cumulative counts and shard information."""
        shard_idx = len(metadata["shards"])
        shard_path = self.output_dir / f"shard_{shard_idx:06}.npz"
        arrays = {
            key: np.array(values[:tokens_to_process], dtype=np.uint8 if key == "mask" else np.uint16)
            for key, values in buffers.items()
        }
        np.savez(shard_path, **arrays)
        self._update_metadata(metadata, rows_consumed, tokens_to_process)