import os
from pathlib import Path

from sprout.data.data_processing import DatasetProcessor


num_tokens = 15_000_000_000
shard_size = 100_000_000
tokenizer_state_path = Path(__file__).parent.parent / "tokenizer" / "bpe_tokenizer_mixed_state.json"
mixture_probs = [0.80, 0.20] # Base, Math
metadata_path = "./metadata_mix.json"
output_dir = "./shards/mixture_shards"

processor = DatasetProcessor(
    mixture_probabilites=mixture_probs,
    metadata_path=metadata_path,
    output_dir=output_dir,
    tokenizer_state_path=tokenizer_state_path,
    num_workers=8,
    imap_chunksize=100, # Number of documents to process with each imap call
    shard_size=shard_size,
    num_tokens=num_tokens
)
processor.process()
os._exit(0) # Skip cleanup to avoid known pyarrow/datasets bug