import os
from pathlib import Path

from datasets import load_dataset

from sprout.data.data_processing import DatasetProcessor


num_tokens = 5_000_000_000
shard_size = 100_000_000
tokenizer_state_path = Path(__file__).parent.parent / "tokenizer" / "bpe_tokenizer_mixed_state.json"
metadata_path = "./metadata_mix.json"
output_dir = "./shards/mixture_shards"
base_ds = load_dataset("HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled", split="train", streaming=True)
math_ds = load_dataset("HuggingFaceTB/finemath", name="finemath-4plus", split="train", streaming=True)
dataset_mixture = {
    base_ds: 0.9,
    math_ds: 0.1,
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