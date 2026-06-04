import os

from datasets import load_dataset, interleave_datasets
from tqdm import tqdm


total_documents = 20_000
training_data_path = "./tokenizer_data_mixed_20k_v4.txt"
base1_ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True).select_columns(["text"])
base2_ds = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True).select_columns(["text"])
code_ds = load_dataset("allenai/olmo-mix-1124", name="starcoder", split="train", streaming=True).select_columns(["text"])
math_ds = load_dataset("HuggingFaceTB/finemath", name="finemath-4plus", split="train", streaming=True).select_columns(["text"])
dataset_mixture = {
    base1_ds: 0.35,
    base2_ds: 0.35,
    code_ds: 0.20,
    math_ds: 0.10,
}

train_ds = interleave_datasets(
    datasets=list(dataset_mixture.keys()),
    probabilities=list(dataset_mixture.values()),
    seed=42,
)
train_iter = iter(train_ds)

with open(training_data_path, "w", encoding="utf-8") as f:
    # Pull Base Documents
    written_docs = 0
    with tqdm(total=total_documents, desc="Writing training data...", unit="doc") as pbar:
        while written_docs < total_documents:
            row = next(train_iter)
            text = row.get("text", "")
            if text.strip():
                f.write(text + "\n")
                written_docs += 1
                pbar.update(1)

os._exit(0) # Skip cleanup to avoid known pyarrow/datasets bug