import os

from datasets import load_dataset

num_base_docs = 5_000
num_math_docs = 5_000
training_data_path = "./training_data_mixed.txt"

base_ds = load_dataset(
    "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled", 
    split="train", 
    streaming=True
)
base_iter = iter(base_ds)

math_ds = load_dataset(
    "HuggingFaceTB/finemath", 
    name="finemath-4plus", 
    split="train", 
    streaming=True
)
math_iter = iter(math_ds)

with open(training_data_path, "w", encoding="utf-8") as f:
    # Pull Base Documents
    written_base = 0
    while written_base < num_base_docs:
        row = next(base_iter)
        text = row.get("text", "")
        if text.strip():
            f.write(text + "\n")
            written_base += 1
            
    # Pull Math Documents
    written_math = 0
    while written_math < num_math_docs:
        row = next(math_iter)
        text = row.get("text", "")
        if text.strip():
            f.write(text + "\n")
            written_math += 1

os._exit(0) # Skip cleanup to avoid known pyarrow/datasets bug