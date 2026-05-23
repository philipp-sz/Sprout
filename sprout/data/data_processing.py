import json
from multiprocessing import Pool
from pathlib import Path
from collections.abc import Iterator

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, interleave_datasets

from sprout.tokenizer.fast_bpe_tokenizer import FastBPETokenizer


_worker_tokenizer = None

def _worker_init(tokenizer_state_path):
    global _worker_tokenizer
    _worker_tokenizer = FastBPETokenizer.load(tokenizer_state_path)

def _tokenize_doc(text):
    tokens = _worker_tokenizer.encode(text)
    tokens.append(_worker_tokenizer.eot_token)
    return tokens

class DatasetProcessor:
    """
    Loads a dataset, tokenizes it, and saves the tokenized data in shards.
    Args:
        mixtures (list[float]): The mixture probabilities for base, and math datasets.
        metadata_path (str): The path where metadata is saved.
        output_dir (str): The directory where the tokenized data will be saved.
        tokenizer_state_path (str): The path to the tokenizer state.
        num_workers (int): The number of worker processes for tokenization.
        imap_chunksize (int): The chunk size for imap when tokenizing documents.
        shard_size (int): The number of samples per shard.
        num_tokens (int): The number of tokens to stream from the dataset.
    """
    def __init__(self, mixture_probabilites: list[float], metadata_path: str, output_dir: str,
                tokenizer_state_path: str,num_workers: int, imap_chunksize: int, shard_size: int, num_tokens: int) -> None:
        self.mixture_probabilites = mixture_probabilites
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.tokenizer_state_path = tokenizer_state_path
        self.num_workers = num_workers
        self.imap_chunksize = imap_chunksize
        self.shard_size = shard_size
        self.num_tokens = num_tokens

    def process(self) -> None:
        """Processes the dataset by streaming, tokenizing, and saving in shards while tracking progress in metadata."""
        metadata, ds_iter, num_total_tokens = self._initialize_processing() # Initialize processing and get metadata, dataset iterator, and total token count

        token_buffer = []
        rows_in_shard = 0
        with Pool(self.num_workers, initializer=_worker_init, initargs=(self.tokenizer_state_path,)) as pool:
            text_stream = (row["text"] for row in ds_iter)

            with tqdm(total=self.num_tokens, initial=num_total_tokens, desc="Processing Dataset") as pbar:
                # Use imap to keep track of rows in shard and chunk size instead of the default of 1 to speed up processing
                for tokenized_doc in pool.imap(_tokenize_doc, text_stream, chunksize=self.imap_chunksize):
                    token_buffer.extend(tokenized_doc) 
                    rows_in_shard += 1

                    while len(token_buffer) >= self.shard_size and num_total_tokens < self.num_tokens:
                        tokens_to_process = min(self.shard_size, self.num_tokens - num_total_tokens)
                        shard_data = token_buffer[:tokens_to_process]
                        token_buffer = token_buffer[tokens_to_process:] # Remove the processed tokens from the buffer
                        self._save_shard(metadata, tokens_to_process, shard_data, rows_in_shard)
                        num_total_tokens += tokens_to_process
                        rows_in_shard = 0
                        pbar.update(tokens_to_process)
                    
                    if num_total_tokens >= self.num_tokens:
                        break

                if token_buffer and num_total_tokens < self.num_tokens: # Process any remaining tokens if needed
                    remaining = min(len(token_buffer), self.num_tokens - num_total_tokens)
                    self._save_shard(metadata, remaining, token_buffer[:remaining], rows_in_shard)
                    pbar.update(remaining)
    
    def _initialize_processing(self) -> tuple[dict, Iterator, int]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = self.load_metadata() # Load or create metadata 

        # Get tracking information
        proj_state = metadata["project_state"]
        num_total_rows = proj_state["num_total_rows"]
        num_total_tokens = proj_state["num_total_tokens"]

        base_ds = load_dataset("HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled", split="train", streaming=True)
        math_ds = load_dataset("HuggingFaceTB/finemath", name="finemath-4plus", split="train", streaming=True)
        ds = interleave_datasets(
            datasets=[base_ds, math_ds],
            probabilities=self.mixture_probabilites,
            seed=42, # Reproducibility of the interleaving for consistency
            stopping_strategy="all_exhausted"
        )

        ds = ds.skip(num_total_rows)
        ds_iter = iter(ds)
        return metadata, ds_iter, num_total_tokens

    def _save_shard(self, metadata: dict, tokens_to_process: int, shard_data: list, rows_in_shard: int) -> None:
        """Saves a shard of tokenized data to disk and updates metadata with the new cumulative counts and shard information."""
        shard_idx = len(metadata["shards"])
        shard_path = self.output_dir / f"shard_{shard_idx:06}.npy"
        arr = np.array(shard_data, dtype=np.uint16) 
        np.save(shard_path, arr)
        self._update_metadata(metadata, rows_in_shard, tokens_to_process)

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
        
        # If the metadata file already exists, load and return metadata
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata
    
    def _update_metadata(self, metadata: dict, row_count: int, token_count: int) -> None:
        """Updates the metadata with new cumulative counts and shard information after processing a shard."""
        # Update metadata with the new cumulative counts and shard information
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
    def __init__(self, shards_dir: str, batch_size: int, seq_len: int, device: str, split: str = "train") -> None:
        self.shards_dir = shards_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.split = split
        self.shard_idx = -1 # No shard loaded yet
        self.cur_pos = 0
        self.shard_tokens = None # Initialize the current shard tokens
        self.shard_files = self._get_shard_files()

    def get_batch(self) -> tuple:
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

    def _get_shard_files(self) -> list:
        """Retrieves and sorts the list of shard files for the specified split."""
        shard_files = sorted(Path(self.shards_dir).glob("shard_*.npy"))
        if self.split == "val":
            shard_files = shard_files[:1]
        elif self.split == "train":
            shard_files = shard_files[1:]
        if not shard_files:
            raise ValueError(f"No shard files found in {self.shards_dir} for split '{self.split}'")
        return shard_files

    def _load_next_shard(self) -> None:
        """Loads the next shard of token IDs into memory and resets the current position."""
        self.shard_idx = (self.shard_idx + 1) % len(self.shard_files) # Increment index and wrap around if end of shard list is reached
        self.shard_tokens = np.load(self.shard_files[self.shard_idx])
        self.cur_pos = 0

    @property
    def get_state(self) -> dict:
        """Returns the current state of the data loader, including shard index and position."""
        return {
            "shard_idx": self.shard_idx,
            "cur_pos": self.cur_pos
        }

    @get_state.setter
    def set_state(self, state: dict) -> None:
        """Sets the state of the data loader to the specified shard index and position."""
        self.shard_idx = state["shard_idx"]
        self.cur_pos = state["cur_pos"]
        self.shard_tokens = np.load(self.shard_files[self.shard_idx])