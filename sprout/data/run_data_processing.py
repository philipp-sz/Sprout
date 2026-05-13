import os
from pathlib import Path

from sprout.data.data_processing import DatasetProcessor


shard_size = 10_000_000 # 100_000_000
num_tokens = 100_000_000 # 2_500_000_000
tokenizer_state_path = Path(__file__).parent.parent / "tokenizer" / "bpe_tokenizer_state.json"
dataset_id = "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled"
metadata_path = "./metadata.json"
output_dir = "./shards"

processor = DatasetProcessor(
    dataset_id=dataset_id,
    metadata_path=metadata_path,
    output_dir=output_dir,
    tokenizer_state_path=tokenizer_state_path,
    num_workers=8,
    batch_size=100, # Number of documents to process with each imap call
    shard_size=shard_size,
    num_tokens=num_tokens
)
processor.process()
os._exit(0) # Skip cleanup to avoid known pyarrow/datasets bug