import os
from pathlib import Path

from sprout.tokenizer.fast_bpe_tokenizer import FastBPETokenizer
from sprout.data.data_processing import DatasetProcessor


shard_size = 10_000 # 100_000_000
num_tokens = 100_000 # 2_500_000_000
tokenizer_state_path = Path(__file__).parent.parent / "tokenizer" / "bpe_tokenizer_state2.json"
dataset_id = "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled"
metadata_path = "./metadata.json"
output_dir = "./shards"

tokenizer = FastBPETokenizer.load(tokenizer_state_path)
processor = DatasetProcessor(dataset_id, metadata_path, output_dir, tokenizer, shard_size, num_tokens)
processor.process()
os._exit(0) # Skip cleanup to avoid known pyarrow/datasets bug