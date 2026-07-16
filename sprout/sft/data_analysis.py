from pathlib import Path
from typing import Callable

import pandas as pd
from datasets import IterableDataset, interleave_datasets, load_dataset
from tqdm import tqdm

from sprout.tokenizer import FastBPETokenizer
from sprout.utils import chat_ml_formatter, token_length_filter
from sprout.utils import plot_sft_filter_stats


def analyze_dataset(dataset_mixture: dict[IterableDataset, float], tokenizer_state_path: str, 
                    text_formatter: Callable[[str], str] | None = None, token_filter: Callable[[list[int]], bool] | None = None,
                    max_samples: int | None = None) -> pd.DataFrame:
    tokenizer = FastBPETokenizer.load(tokenizer_state_path)
    
    ds = interleave_datasets(
        datasets=list(dataset_mixture.keys()),
        probabilities=list(dataset_mixture.values()),
        seed=42,
        stopping_strategy="all_exhausted"
    )
    
    records = []
    for row in tqdm(ds, desc="Analyzing...", total=max_samples):
        messages = row["messages"]
        source = row.get("source")
        
        if text_formatter:
            messages = text_formatter(messages)
        
        tokens = tokenizer.encode(messages)
        tokens.append(tokenizer.eot_token)
        
        records.append({
            "source": source,
            "token_length": len(tokens),
            "filtered": token_filter is not None and not token_filter(tokens),
        })
        
        if max_samples and len(records) >= max_samples:
            break
    
    return pd.DataFrame(records)

if __name__ == "__main__":
    max_samples = 50_000
    plot_path = Path(__file__).parent / "plots" / "sft_data_analysis.png"
    tokenizer_state_path = Path(__file__).parent.parent / "tokenizer" / "states" / "bpe_mixed_32k_20k_instruct_state.json"
    sft_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True)
    dataset_mixture = {sft_ds: 1.0}
    df = analyze_dataset(
        dataset_mixture, tokenizer_state_path,
        text_formatter=chat_ml_formatter,
        token_filter=token_length_filter,
        max_samples=max_samples
    )
    plot_sft_filter_stats(df, out_path=plot_path)

