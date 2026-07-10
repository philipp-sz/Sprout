from pathlib import Path

import numpy as np
import torch

from sprout.data import DatasetProcessor, TokenDataLoader
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
    """
    Processes a dataset of conversations for supervised fine-tuning (SFT) by tokenizing the data and creating masks for each turn.
    """
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

class SFTDataLoader(TokenDataLoader):
    def __init__(self, *args, eot_token: int, **kwargs):
        super().__init__(*args, shard_extension="npz", **kwargs)
        self.shard_masks = None
        self.eot_positions = None
        self.eot_token = eot_token

    def get_batch(self):
        B, T = self.batch_size, self.seq_len
        token_batch = torch.zeros((B, T), dtype=torch.long)
        mask_batch = torch.zeros((B, T), dtype=torch.uint8)
        for idx in range(B):
            token_batch[idx], mask_batch[idx] = self._build_row(T)

        x = token_batch[:, :-1].to(self.device)
        y = token_batch[:, 1:].to(self.device)
        mask = mask_batch[:, 1:].to(self.device)
        return x, y, mask
    
    def _build_row(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shard_tokens is None:
            self._load_next_shard()

        idx_first = torch.searchsorted(self.eot_positions, self.cur_pos)
        idx_last = torch.searchsorted(self.eot_positions, self.cur_pos + seq_len - 1, right=True) - 1
        if idx_first >= len(self.eot_positions): # Shard is exhausted
            self._load_next_shard()
            return self._build_row(seq_len)
        elif idx_last >= idx_first: # No remaining batch budget
            conv_len = self.eot_positions[idx_last] - self.cur_pos + 1
            pad_len = seq_len - conv_len
            pad_chunk = torch.tensor([self.eot_token] * pad_len, dtype=torch.long)

            tokens_chunk = self.shard_tokens[self.cur_pos:self.cur_pos + conv_len]
            tokens_chunk = torch.cat((tokens_chunk, pad_chunk), dim=0)
            mask_chunk = self.shard_masks[self.cur_pos:self.cur_pos + conv_len]
            mask_chunk = torch.cat((mask_chunk, torch.zeros(pad_len, dtype=torch.uint8)), dim=0)
            self.cur_pos += conv_len
            return tokens_chunk, mask_chunk
        else: # First sequence is too long
            if idx_first + 1 >= len(self.eot_positions):
                self._load_next_shard()
            else:
                self.cur_pos = self.eot_positions[idx_first + 1] + 1
            return self._build_row(seq_len)

    def _load_next_shard(self) -> None:
        """Loads the next shard of token IDs and masks into memory and resets the current position."""
        self.shard_files = self._get_shard_files()
        self.shard_idx = (self.shard_idx + 1) % len(self.shard_files)
        loaded_file = np.load(self.shard_files[self.shard_idx])
        self.shard_tokens = torch.tensor(loaded_file["tokens"], dtype=torch.long)
        self.shard_masks = torch.tensor(loaded_file["mask"], dtype=torch.uint8)
        self.eot_positions = torch.where(self.shard_tokens == self.eot_token)[0]
        self.cur_pos = 0