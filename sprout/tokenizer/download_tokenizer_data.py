import os

from datasets import load_dataset

num_base_docs = 8_000 # 40%
num_code_docs = 10_000 # 50%
num_math_docs = 2_000 # 10%
training_data_path = "./tokenizer_data_mixed_20k.txt"

base_ds = load_dataset(
    "HuggingFaceFW/fineweb-edu", 
    split="train", 
    streaming=True
)
base_iter = iter(base_ds)

code_ds = load_dataset(
    "allenai/olmo-mix-1124",
    name="starcoder",
    split="train",
    streaming=True
)

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

    # Pull Code Documents
    written_code = 0
    while written_code < num_code_docs:
        row = next(code_ds)
        text = row.get("code", "")
        if text.strip():
            f.write(text + "\n")
            written_code += 1
            
    # Pull Math Documents
    written_math = 0
    while written_math < num_math_docs:
        row = next(math_iter)
        text = row.get("text", "")
        if text.strip():
            f.write(text + "\n")
            written_math += 1

os._exit(0) # Skip cleanup to avoid known pyarrow/datasets bug