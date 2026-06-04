import os
from pathlib import Path

from datasets import load_dataset

from sprout.data.data_processing import DatasetProcessor


num_tokens = 2_500_000_000
shard_size = 100_000_000
tokenizer_state_path = Path(__file__).parent.parent / "tokenizer" / "bpe_mixed_32k_20k_v4_state.json"
metadata_path = "./metadata_train_2_5b.json"
output_dir = "./shards/train_shards_2_5b"
base1_ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True).select_columns(["text"])
base2_ds = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True).select_columns(["text"])
code_ds = load_dataset("allenai/olmo-mix-1124", name="starcoder", split="train", streaming=True).select_columns(["text"])
math_ds = load_dataset("HuggingFaceTB/finemath", name="finemath-4plus", split="train", streaming=True).select_columns(["text"])
dataset_mixture = {
    base1_ds: 0.45,
    base2_ds: 0.45,
    code_ds: 0.04,
    math_ds: 0.06,
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