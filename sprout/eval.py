import json
import os
from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
import warnings

import torch
import torch.nn.functional as F
from tqdm import trange
from datasets import Dataset, load_dataset


@dataclass
class BenchmarkConfig:
    extract_text: Callable
    eval_func: Callable
    extract_choices: Callable | None = None
    extract_label: Callable | None = None
    format_func: Callable | None = None
    total_samples: int | None = None

class Evaluator:
    """Evaluates a model on different metrics and benchmarks."""
    def __init__(self, model, tokenizer, device, prefix_token_id=None):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.prefix_tokens = [prefix_token_id] if prefix_token_id is not None else []

    def run_standard_suite(self, total_samples: int | None = None) -> dict[str, float]:
        """Loads datasets and runs the standard benchmark suite, including LAMBADA, ARC Easy, ARC Challenge, and HellaSwag."""
        config_kwargs = {"total_samples": total_samples} if total_samples is not None else {}

        lambada_config = BenchmarkConfig(
            extract_text=lambda x: x["text"], 
            eval_func=self.evaluate_cloze_benchmark,
            **config_kwargs
        )
        arc_config = BenchmarkConfig(
            extract_text=lambda x: x["question"],
            extract_choices=lambda x: x["choices"]["text"],
            extract_label=lambda x: x["answerKey"],
            eval_func=self.evaluate_multi_token_benchmark,
            format_func=self.format_arc,
            **config_kwargs
        )
        hellaswag_config = BenchmarkConfig(
            extract_text=lambda x: x["ctx"],
            extract_choices=lambda x: x["endings"],
            extract_label=lambda x: x["label"],
            eval_func=self.evaluate_multi_token_benchmark,
            format_func=self.format_hellaswag,
            **config_kwargs
        )

        lambada_ds = load_dataset("EleutherAI/lambada_openai", split="test")
        arc_easy_ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        arc_challenge_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        hellaswag_ds = load_dataset("Rowan/hellaswag", split="validation")

        return {
            "lambada": self.evaluate_benchmark(lambada_ds, lambada_config),
            "arc_easy": self.evaluate_benchmark(arc_easy_ds, arc_config),
            "arc_challenge": self.evaluate_benchmark(arc_challenge_ds, arc_config),
            "hellaswag": self.evaluate_benchmark(hellaswag_ds, hellaswag_config)
        }

    def calc_perplexity(self, input_tokens: torch.Tensor, label_tokens: torch.Tensor) -> float:
        """Calculate the perplexity of the model on a given text."""
        self.model.eval()
        total_loss_sum = 0.0
        total_valid_tokens = 0

        with torch.no_grad():
            for batch_input, batch_labels in zip(input_tokens, label_tokens):
                if self.prefix_tokens:
                    # Prepend the prefix and drop the last token, so the sequence keeps its length (max_seq_len)
                    prefix = torch.tensor(self.prefix_tokens, dtype=torch.long, device=batch_input.device)
                    batch_labels = torch.cat([batch_input[:1], batch_labels[:-1]])
                    batch_input = torch.cat([prefix, batch_input[:-1]])

                batch_input = batch_input.unsqueeze(0).to(self.device)
                batch_labels = batch_labels.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    output = self.model(batch_input)

                # Handle HF models and custom models
                if hasattr(output, "logits"):
                    logits = output.logits
                else:
                    logits = output
                logits = logits[0]

                loss = F.cross_entropy(logits, batch_labels, reduction="mean")
                total_loss_sum += loss.item() * batch_labels.numel()
                total_valid_tokens += batch_labels.numel()

        if total_valid_tokens == 0:
            warnings.warn("No valid tokens found for perplexity calculation.")
            return float("inf")

        perplexity = math.exp(total_loss_sum / total_valid_tokens) 
        return perplexity

    def evaluate_multi_token_benchmark(self, text: str, choices: list[str], label: int) -> bool:
        """Evaluate the model on a multiple-choice benchmark that requires multi-token continuation (e.g. HellaSwag)."""
        scores = []
        for choice in choices:
            log_prob, _ = self._loglikelihood_context(text, choice)
            scores.append(log_prob)

        predicted_label = scores.index(max(scores))
        return predicted_label == label

    def evaluate_cloze_benchmark(self, text: str) -> bool:
        """Evaluate the model on a cloze-style benchmark (e.g. LAMBADA) where the model must predict the last word of a passage."""
        self.model.eval()
        words = text.split(" ")
        context_str = " ".join(words[:-1])
        target_str = " " + words[-1]
    
        _, is_correct = self._loglikelihood_context(context_str, target_str)
        return is_correct

    def evaluate_benchmark(self, dataset: Dataset, config: BenchmarkConfig) -> float:
        """Evaluate the model on a given dataset using the provided evaluation function."""
        total_samples = config.total_samples or len(dataset)
        correct_predictions = 0

        for idx in trange(total_samples, leave=False, desc="Running Benchmark..."):
            sample = dataset[idx]
            kwargs = {
                "text": config.extract_text(sample),
                "choices": config.extract_choices(sample) if config.extract_choices else None,
                "label": config.extract_label(sample) if config.extract_label else None
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            if config.format_func:
                kwargs = config.format_func(**kwargs)

            eval_results = config.eval_func(**kwargs)
            if eval_results:
                correct_predictions += 1
        accuracy = correct_predictions / total_samples
        return accuracy

    def save(self, path: Path, hellaswag_acc: float, arc_easy_acc: float, arc_challenge_acc: float, lambada_acc: float, perplexity: float | None = None) -> None:
        """Save evaluation results to given file path."""
        avg_acc = (hellaswag_acc + arc_easy_acc + arc_challenge_acc + lambada_acc) / 4
        results = {
            "hellaswag_acc": hellaswag_acc,
            "arc_easy_acc": arc_easy_acc,
            "arc_challenge_acc": arc_challenge_acc,
            "lambada_acc": lambada_acc,
            "average_acc": avg_acc
        }
        if perplexity is not None:
            results["perplexity"] = perplexity
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

    def _loglikelihood_context(self, context: str, choice: str) -> tuple[float, bool]:
        """Compute length-normalized log-likelihood of a completion given a context."""
        self.model.eval()
        context_tokens = self.prefix_tokens + self.tokenizer.encode(context)
        context_len = len(context_tokens)

        full_string = context + choice
        full_tokens = self.prefix_tokens + self.tokenizer.encode(full_string)
        choice_len = len(full_tokens) - context_len
        if choice_len <= 0:
            warnings.warn("Choice is empty or fully truncated, cannot compute log-likelihood.")
            return float("-inf"), False
        
        input_ids = torch.tensor(full_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            output = self.model(input_ids)

            if hasattr(output, "logits"):
                logits = output.logits
            else:
                logits = output
            logits = logits[0]

            # Calculate log-likelihood of the completion (for multi-token benchmarks)
            completion_logits = logits[context_len - 1:-1, :]
            completion_labels = torch.tensor(full_tokens[context_len:], dtype=torch.long, device=self.device)
            log_probs = -F.cross_entropy(completion_logits, completion_labels, reduction="sum").item() # Convert Nll to log-prob
            normalized_logprob = log_probs / choice_len

            # Calculate greedy predictions (for LAMBADA-style eval)
            predictions = torch.argmax(completion_logits, dim=-1)
            is_correct = torch.equal(predictions, completion_labels)

        return normalized_logprob, is_correct

    @staticmethod
    def format_arc(text: str, choices: list[str], label: int) -> dict:
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        context = f"Question: {text}\nAnswer:"
        choices = [f" {choice.strip()}" for choice in choices]
        if label not in label_map:
            return None
        ground_truth = label_map[label]
        return {"text": context, "choices": choices, "label": ground_truth}

    @staticmethod
    def format_hellaswag(text: str, choices: list[str], label: int) -> dict:
        context = text.strip()
        choices = [f" {choice.strip()}" for choice in choices]
        ground_truth = int(label)
        return {"text": context, "choices": choices, "label": ground_truth}

    @staticmethod
    def print_evaluation_results(lambada_acc: float, arc_easy_acc: float, arc_challenge_acc: float, hellaswag_acc: float, perplexity: float | None = None) -> None:
        """Print the evaluation results in a readable format."""
        avg_acc = (lambada_acc + arc_easy_acc + arc_challenge_acc + hellaswag_acc) / 4
        border = "=" * 60
        print(border)
        print(f"{'Evaluation Results':^60}")
        print(border)
        print(f"{'Perplexity':<30}: {perplexity:.2f}" if perplexity is not None else f"{'Perplexity':<30}: N/A")
        print(f"{'LAMBADA Accuracy':<30}: {lambada_acc:.2%}")
        print(f"{'ARC (Easy) Accuracy':<30}: {arc_easy_acc:.2%}")
        print(f"{'ARC (Challenge) Accuracy':<30}: {arc_challenge_acc:.2%}")
        print(f"{'HellaSwag Accuracy':<30}: {hellaswag_acc:.2%}")
        print(f"{'Average Accuracy':<30}: {avg_acc:.2%}")
        print(border)