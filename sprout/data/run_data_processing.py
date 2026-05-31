import os
from pathlib import Path

from datasets import load_dataset

from sprout.data.data_processing import DatasetProcessor


num_tokens = 7_500_000_000
shard_size = 100_000_000
tokenizer_state_path = Path(__file__).parent.parent / "tokenizer" / "bpe_mixed_32k_20k_state.json"
metadata_path = "./metadata_mixed_7_5b.json"
output_dir = "./shards/mixed_7_5b_shards"
base_ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
code_ds = load_dataset("allenai/olmo-mix-1124", name="starcoder", split="train", streaming=True)
math_ds = load_dataset("HuggingFaceTB/finemath", name="finemath-4plus", split="train", streaming=True)
dataset_mixture = {
    base_ds: 0.85,
    code_ds: 0.1,
    math_ds: 0.05,
}

processor = DatasetProcessor(
    dataset_mixture=dataset_mixture,
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