import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


class DatasetProcessor:
    """
    Loads a dataset, tokenizes it, and saves the tokenized data in shards.
    Args:
        dataset_id (str): The identifier of the dataset to load.
        metadata_path (str): The path where metadata is saved.
        output_dir (str): The directory where the tokenized data will be saved.
        tokenizer: The tokenizer to use for tokenizing the dataset.
        shard_size (int): The number of samples per shard.
        num_tokens (int): The number of tokens to stream from the dataset.
    """
    def __init__(self, dataset_id: str, metadata_path: str, output_dir: str, tokenizer, shard_size: int, num_tokens: int) -> None:
        self.dataset_id = dataset_id
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.shard_size = shard_size
        self.num_tokens = num_tokens

    def process(self) -> None:
        """Processes the dataset by streaming, tokenizing, and saving in shards while tracking progress in metadata."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = self.load_metadata() # Load or create metadata 

        # Get tracking information
        proj_state = metadata["project_state"]
        num_total_rows = proj_state["num_total_rows"]
        num_total_tokens = proj_state["num_total_tokens"]

        ds = load_dataset(self.dataset_id, split="train", streaming=True)
        ds = ds.skip(num_total_rows) # Skip rows that have already been processed based on metadata
        ds_iter = iter(ds) # Create an iterator for the dataset to stream data row by row

        buffer = []
        with tqdm(total=self.num_tokens, initial=num_total_tokens, desc="Processing Dataset") as pbar:
            while num_total_tokens < self.num_tokens:
                shard_idx = len(metadata["shards"])
                tokens_to_process = min(self.shard_size, self.num_tokens - num_total_tokens) # Handle remainder tokens for the last shard
                row_count, dataset_exhausted = self._process_shard(ds_iter, buffer, tokens_to_process)
                actual_tokens_processed = min(tokens_to_process, len(buffer)) # In case the dataset is exhausted, this will be less than tokens_to_process

                # Update metadata with the new cumulative counts and save the shard
                self._update_metadata(metadata, row_count, actual_tokens_processed)
                self._save_shard(shard_idx, buffer[:actual_tokens_processed])
                buffer = buffer[actual_tokens_processed:] # Start the new buffer with any remaining tokens
                num_total_tokens += actual_tokens_processed
                pbar.update(actual_tokens_processed)

                if dataset_exhausted:
                    print("Dataset exhausted before reaching the target number of tokens.")
                    break

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

    def _process_shard(self, ds_iter: iter, buffer: list, tokens_needed: int) -> tuple:
        """Processes a shard by tokenizing rows until the required number of tokens is reached."""
        row_count = 0
        dataset_exhausted = False

        while len(buffer) < tokens_needed:
            try: # Handle the case where the dataset runs out of data before reaching the required number of tokens
                row = next(ds_iter) # Get the next row/sample from the dataset
                row_count += 1
            except StopIteration:
                dataset_exhausted = True
    
            tokens = self.tokenizer.encode(row["text"]) # Tokenize a single row/sample
            tokens.append(self.tokenizer.eot_token) # Add eot token at the end of each row/sample
            buffer.extend(tokens)

            
        return row_count, dataset_exhausted
    
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

    def _save_shard(self, shard_idx: int, tokenized_data: list) -> None:
        """Saves the tokenized data for a shard as a numpy array file."""
        shard_path = self.output_dir / f"shard_{shard_idx:06}.npy"
        arr = np.array(tokenized_data, dtype=np.uint16) 
        np.save(shard_path, arr)

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
        self.cur_pos = 0
        self.shard_tokens = None # Initialize the current shard tokens
        self.shard_iterator = self._get_shard_iterator() # Create an iterator for the shard files

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

    def _get_shard_iterator(self) -> iter:
        """Creates an iterator over the shard files in the specified directory, filtering by split if necessary."""
        shard_files = sorted(Path(self.shards_dir).glob("shard_*.npy")) # sort list of shard files
        if self.split == "val": # If validation data, only use the first shard for validation
            shard_files = shard_files[:1]
        elif self.split == "train": 
            shard_files = shard_files[1:]

        if not shard_files:
            raise FileNotFoundError("No shard files found in the specified directory.")
        for shard_file in shard_files:
            yield np.load(shard_file) # Yield each shard as a numpy array of token IDs

    def _load_next_shard(self) -> None:
        """Loads the next shard of token IDs into memory and resets the current position."""
        try:
            self.shard_tokens = next(self.shard_iterator) # Load the next shard of token IDs
            self.cur_pos = 0 # Reset the current position in the shard
        except StopIteration: # If there are no more shard, restart iterator for next epoch
            self.shard_iterator = self._get_shard_iterator()
            self.shard_tokens = next(self.shard_iterator)
            self.cur_pos = 0