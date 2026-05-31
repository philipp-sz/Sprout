from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sprout.eval import BenchmarkConfig, Evaluator


log_path = Path(__file__).parent.parent / "logs" / "eval" / "gpt2_eval_results.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
evaluator = Evaluator(model=model, tokenizer=tokenizer, device=device)

def format_arc(text: str, choices: list[str], label: int) -> dict:
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
    context = f"Question: {text}\nAnswer:"
    choices = [f" {choice.strip()}" for choice in choices]
    if label not in label_map:
        return None
    ground_truth = label_map[label]
    return {"text": context, "choices": choices, "label": ground_truth}

def format_hellaswag(text: str, choices: list[str], label: int) -> dict:
    context = text.strip()
    choices = [f" {choice.strip()}" for choice in choices]
    ground_truth = int(label)
    return {"text": context, "choices": choices, "label": ground_truth}

lambada_config = BenchmarkConfig(extract_text=lambda x: x["text"], eval_func=evaluator.evaluate_cloze_benchmark)
arc_config = BenchmarkConfig(
    extract_text=lambda x: x["question"],
    extract_choices=lambda x: x["choices"]["text"],
    extract_label=lambda x: x["answerKey"],
    eval_func=evaluator.evaluate_multi_token_benchmark,
    format_func=format_arc,
)
hellaswag_config = BenchmarkConfig(
    extract_text=lambda x: x["ctx"],
    extract_choices=lambda x: x["endings"],
    extract_label=lambda x: x["label"],
    format_func=format_hellaswag,
    eval_func=evaluator.evaluate_multi_token_benchmark,
)

print("Loading datasets...")
lambada_ds = load_dataset("EleutherAI/lambada_openai", split="test")
arc_easy_ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
arc_challenge_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
hellaswag_ds = load_dataset("Rowan/hellaswag", split="validation")

lambada_acc = evaluator.evaluate_benchmark(lambada_ds, lambada_config)
arc_easy_acc = evaluator.evaluate_benchmark(arc_easy_ds, arc_config)
arc_challenge_acc = evaluator.evaluate_benchmark(arc_challenge_ds, arc_config)
hellaswag_acc = evaluator.evaluate_benchmark(hellaswag_ds, hellaswag_config)

evaluator.print_evaluation_results(lambada_acc, arc_easy_acc, arc_challenge_acc, hellaswag_acc)
evaluator.save(
    path=log_path,
    hellaswag_acc=hellaswag_acc,
    arc_easy_acc=arc_easy_acc,
    arc_challenge_acc=arc_challenge_acc,
    lambada_acc=lambada_acc
)