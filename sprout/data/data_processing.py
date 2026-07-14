from collections import defaultdict
from itertools import islice
import json
from multiprocessing import Pool
from pathlib import Path
from collections.abc import Iterator
from typing import Callable

import torch
import numpy as np
from tqdm import tqdm
from datasets import IterableDataset, interleave_datasets

from sprout.tokenizer.fast_bpe_tokenizer import FastBPETokenizer


_worker_tokenizer = None

def _worker_init(tokenizer_state_path: str) -> None:
    global _worker_tokenizer
    _worker_tokenizer = FastBPETokenizer.load(tokenizer_state_path)

def _tokenize_doc(text: str) -> list[int]:
    tokens = _worker_tokenizer.encode(text)
    tokens.append(_worker_tokenizer.eot_token)
    return tokens

class DatasetProcessor:
    """
    Loads a dataset, tokenizes it, and saves the tokenized data in shards.
    Args:
        dataset_mixture (dict[IterableDataset, float]): A dictionary mapping datasets to their mixture probabilities.
        metadata_path (str): The path where metadata is saved.
        output_dir (str): The directory where the tokenized data will be saved.
        tokenizer_state_path (str): The path to the tokenizer state.
        num_workers (int): The number of worker processes for tokenization.
        imap_chunksize (int): The chunk size for imap when tokenizing documents.
        shard_size (int): The number of samples per shard.
        num_tokens (int): The number of tokens to stream from the dataset.
        token_filter (Callable[[list[int]], bool], optional): A function to filter tokenized documents (e.g. by length). Defaults to None.
        tokenize_fn (Callable[[str], list[int]]) : A function to tokenize documents. Defaults to _tokenize_doc.
        worker_init (Callable[[str], None]): A function to initialize the worker processes. Defaults to _worker_init.
        text_key (str): The key in the dataset that contains the text to be tokenized. Defaults to "text".
    """
    def __init__(self, dataset_mixture: dict[IterableDataset, float], metadata_path: str, output_dir: str,
                tokenizer_state_path: str,num_workers: int, imap_chunksize: int, shard_size: int, num_tokens: int,
                token_filter: Callable[[list[int]], bool] | None = None, tokenize_fn: Callable[[str], list[int]] = _tokenize_doc,
                worker_init: Callable[[str], None] = _worker_init, text_key: str = "text") -> None:
        self.dataset_mixture = dataset_mixture
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.tokenizer_state_path = tokenizer_state_path
        self.num_workers = num_workers
        self.imap_chunksize = imap_chunksize
        self.shard_size = shard_size
        self.num_tokens = num_tokens
        self.token_filter = token_filter
        self.tokenize_fn = tokenize_fn
        self.worker_init = worker_init
        self.text_key = text_key

    def process(self) -> None:
        """Processes the dataset by streaming, tokenizing, and saving in shards while tracking progress in metadata."""
        metadata, ds_iter, num_total_tokens = self._initialize_processing()

        buffers = defaultdict(list)
        rows_consumed = 0
        with Pool(self.num_workers, initializer=self.worker_init, initargs=(self.tokenizer_state_path,)) as pool:
            text_stream = (row[self.text_key] for row in ds_iter)

            with tqdm(total=self.num_tokens, initial=num_total_tokens, desc="Processing Dataset") as pbar:
                # Use imap to keep track of rows in shard and chunk size instead of the default of 1 to speed up processing
                for results in pool.imap(self.tokenize_fn, text_stream, chunksize=self.imap_chunksize):
                    rows_consumed += 1
                    if self.token_filter is not None and not self.token_filter(results["tokens"]):
                            continue # Skip documents that do not pass (e.g. length)
                    for key, chunk in results.items():
                        buffers[key].extend(chunk)
                    buffers, num_total_tokens, rows_consumed = self._write_shards(
                        metadata, buffers, num_total_tokens, rows_consumed, pbar
                    )
                    if num_total_tokens >= self.num_tokens:
                        break
        self._flush_remaining(metadata, buffers, num_total_tokens, rows_consumed, pbar)

    def _write_shards(self, metadata: dict, buffers: dict[str, list[int]], num_total_tokens: int, rows_consumed: int, pbar: tqdm) -> tuple[dict, int, int]:
        """Writes shards to disk while there are enough tokens in the buffer and updates metadata."""
        while len(buffers["tokens"]) >= self.shard_size and num_total_tokens < self.num_tokens:
            tokens_to_process = min(self.shard_size, self.num_tokens - num_total_tokens)
            shard_buffers = {}
            for key in buffers.keys():
                shard_buffers[key] = buffers[key][:tokens_to_process]
                buffers[key] = buffers[key][tokens_to_process:] # Remove the processed tokens from the buffer
            self._save_shard(metadata, tokens_to_process, shard_buffers, rows_consumed)
            num_total_tokens += tokens_to_process
            rows_consumed = 0
            pbar.update(tokens_to_process)
        return buffers, num_total_tokens, rows_consumed

    def _flush_remaining(self, metadata: dict, buffers: dict[str, list[int]], num_total_tokens: int, rows_consumed: int, pbar: tqdm) -> None:
        """Flushes any remaining tokens in the buffer to a final shard if the end of the dataset is reached."""
        if buffers["tokens"] and num_total_tokens < self.num_tokens:
            remaining = min(len(buffers["tokens"]), self.num_tokens - num_total_tokens)
            self._save_shard(metadata, remaining, buffers, rows_consumed)
            pbar.update(remaining)

    def _initialize_processing(self) -> tuple[dict, Iterator, int]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = self.load_metadata()

        # Get tracking information
        proj_state = metadata["project_state"]
        num_total_rows = proj_state["num_total_rows"]
        num_total_tokens = proj_state["num_total_tokens"]

        ds = interleave_datasets(
            datasets=list(self.dataset_mixture.keys()),
            probabilities=list(self.dataset_mixture.values()),
            seed=42,
            stopping_strategy="all_exhausted"
        )

        ds = ds.skip(num_total_rows)
        ds_iter = iter(ds)
        return metadata, ds_iter, num_total_tokens

    def _save_shard(self, metadata: dict, tokens_to_process: int, buffers: dict[str, list[int]], rows_consumed: int) -> None:
        """Saves a shard of tokenized data to disk and updates metadata with the new cumulative counts and shard information."""
        shard_idx = len(metadata["shards"])
        shard_path = self.output_dir / f"shard_{shard_idx:06}.npy"
        arr = np.array(buffers["tokens"][:tokens_to_process], dtype=np.uint16) 
        np.save(shard_path, arr)
        self._update_metadata(metadata, rows_consumed, tokens_to_process)

    def load_metadata(self) -> dict:
        """Loads existing metadata or creates new metadata if it does not exist."""
        default_metadata = {
            "project_state": {
                "num_total_rows": 0,
                "num_total_tokens": 0,
            },
            "shards": []
        }

        if not self.metadata_path.exists() or self.metadata_path.stat().st_size == 0:
            with open(self.metadata_path, "w") as f: 
                json.dump(default_metadata, f, indent=4)
            return default_metadata

        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata
    
    def _update_metadata(self, metadata: dict, row_count: int, token_count: int) -> None:
        """Updates the metadata with new cumulative counts and shard information after processing a shard."""
        metadata["project_state"]["num_total_rows"] += row_count
        metadata["project_state"]["num_total_tokens"] += token_count
        metadata["shards"].append({
            "shard_idx": len(metadata["shards"]),
            "num_rows": row_count,
            "num_tokens": token_count
        })
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

class TokenDataLoader:
    """
    Loads tokenized data from shards and prepares batches for training.
    Args:
    shards_dir (str): The directory where the tokenized shards are saved.
    batch_size (int): The number of samples per batch.
    seq_len (int): The sequence length for each sample.
    device (str): The device to load the batches onto ("cpu" or "cuda").
    split (str): The dataset split to load ("train", "val")
    """
    def __init__(self, shards_dir: str, batch_size: int, seq_len: int, device: str, split: str = "train", shard_extension: str = "npy") -> None:
        self.shards_dir = shards_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.split = split
        self.shard_extension = shard_extension
        self.shard_idx = -1 # No shard loaded yet
        self.cur_pos = 0
        self.shard_tokens = None # Initialize the current shard tokens
        self.shard_files = self._get_shard_files()

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a batch of tokenized data from the current shard, loading new shards as needed."""
        B, T = self.batch_size, self.seq_len
        if self.shard_tokens is None or self.cur_pos + B * T + 1 > len(self.shard_tokens):
            self._load_next_shard() # Skip to the next shard if the current shard does not have enough tokens for the batch

        chunk_end = self.cur_pos + B * T + 1
        chunk = self.shard_tokens[self.cur_pos:chunk_end] # Get the chunk of token IDs for the batch
        x = torch.tensor(chunk[:-1], dtype=torch.long).view(B, T).to(self.device) # Input tokens without the last token
        y = torch.tensor(chunk[1:], dtype=torch.long).view(B, T).to(self.device) # Target tokens without the first token
        self.cur_pos += B * T 
        return x, y

    def _get_shard_files(self) -> list[Path]:
        """Retrieves and sorts the list of shard files for the specified split."""
        shard_files = sorted(Path(self.shards_dir).glob(f"shard_*.{self.shard_extension}"))
        if self.split == "val":
            shard_files = shard_files[:1]
        elif self.split == "train":
            shard_files = shard_files[1:]
        if not shard_files:
            raise ValueError(f"No shard files found in {self.shards_dir} for split '{self.split}'")
        return shard_files

    def _load_next_shard(self) -> None:
        """Loads the next shard of token IDs into memory and resets the current position."""
        self.shard_files = self._get_shard_files() # Refresh the list of shard files in case new shards have been added
        self.shard_idx = (self.shard_idx + 1) % len(self.shard_files) # Increment index and wrap around if end of shard list is reached
        self.shard_tokens = np.load(self.shard_files[self.shard_idx])
        self.cur_pos = 0

    @property
    def get_state(self) -> dict[str, int]:
        """Returns the current state of the data loader, including shard index and position."""
        return {
            "shard_idx": self.shard_idx,
            "cur_pos": self.cur_pos
        }

    @get_state.setter
    def set_state(self, state: dict[str, int]) -> None:
        """Sets the state of the data loader to the specified shard index and position."""
        self.shard_idx = state["shard_idx"]
        self.cur_pos = state["cur_pos"]
        self.shard_tokens = np.load(self.shard_files[self.shard_idx])