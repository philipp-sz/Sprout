import os

from datasets import load_dataset


num_docs = 10_000
dataset_id = "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled"
training_data_path = "./training_data.txt"
ds = load_dataset(dataset_id, split="train", streaming=True)
ds_iter = iter(ds)

num_docs_written = 0
with open(training_data_path, "w", encoding="utf-8") as f:
    while num_docs_written < num_docs:
        row = next(ds_iter)
        f.write(row["text"] + "\n")
        num_docs_written += 1

os._exit(0) # Skip cleanup to avoid known pyarrow/datasets bug