import os
from pathlib import Path

from datasets import load_dataset

from sprout.sft import SFTDatasetProcessor
from sprout.utils import chat_ml_formatter, token_length_filter


num_tokens = 200_000_000
shard_size = 10_000_000
tokenizer_state_path = Path(__file__).parent.parent / "tokenizer" / "bpe_mixed_32k_20k_v4_instruct_state.json"
metadata_path = Path(__file__).parent.parent / "data" / "metadata" / "metadata_sft_200M.json"
output_dir = Path(__file__).parent.parent / "data"/ "shards" / "sft_shards_200M"
sft_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True).select_columns(["messages"])
dataset_mixture = {sft_ds: 1.0}

processor = SFTDatasetProcessor(
    dataset_mixture=dataset_mixture,
    metadata_path=metadata_path,
    output_dir=output_dir,
    tokenizer_state_path=tokenizer_state_path,
    num_workers=8,
    imap_chunksize=100, # Number of documents to process with each imap call
    shard_size=shard_size,
    num_tokens=num_tokens,
    token_filter=token_length_filter,
    text_key="messages"
)
processor.process()
os._exit(0) # Skip cleanup to avoid known pyarrow/datasets bug