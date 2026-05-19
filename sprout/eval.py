import torch
import torch.nn.functional as F
from datasets import load_dataset


class Evaluator:
    """Evaluates a model on different metrics and benchmarks."""
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def calc_perplexity(self, text: str) -> float:
        """Calculate the perplexity of the model on a given text."""
        token_ids = self.tokenizer.encode(text)
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.config.device) # Shape: [1, seq_len]
        with torch.no_grad():
            logits = self.model(input_ids)[0]
            shift_logits = logits[:-1, :]
            shift_labels = torch.tensor(token_ids[1:], dtype=torch.long).to(self.config.device)

            log_probs = F.log_softmax(shift_logits, dim=-1)
            target_log_probs = log_probs[torch.arange(len(shift_labels)), shift_labels]
            avg_log_prob = target_log_probs.mean().item()
            perplexity = torch.exp(torch.tensor(-avg_log_prob)).item()
        return perplexity

    def evaluate_knowledge_benchmark(self, context: list[str], endings: list[list[str]], labels: list[int]) -> float:
        """Evaluate the model on a completion benchmark (e.g. Hellaswag, MMLU, etc.) where the model must choose the correct ending based on context."""
        correct = 0
        total = len(context)
        for i, ctx in enumerate(context):
            ending_list = endings[i]
            label = int(labels[i] )

            scores = []
            ending_list = endings[i]
            for ending in ending_list:
                full_string = ctx + ending
                full_tokens = self.tokenizer.encode(full_string)
                context_tokens = self.tokenizer.encode(ctx)
                context_len = len(context_tokens)
                if len(full_tokens) > self.config.max_seq_len:
                    shift = len(full_tokens) - self.config.max_seq_len
                    full_tokens = full_tokens[-self.config.max_seq_len:]
                    context_len = max(0, context_len - shift)

                input_ids = torch.tensor(full_tokens, dtype=torch.long).unsqueeze(0).to(self.config.device)
                with torch.no_grad():
                    logits = self.model(input_ids)[0]
                    log_probs = F.log_softmax(logits, dim=-1)

                    score = 0.0
                    count = 0
                    for t in range(context_len, len(full_tokens)):
                        token_id = full_tokens[t]
                        token_position = t - 1
                        score += log_probs[token_position, token_id].item()
                        count += 1
                    scores.append(score / count if count > 0 else float("-inf"))

            prediction = scores.index(max(scores))
            if prediction == label:
                correct += 1
        return correct / total

    def evaluate_completion_benchmark(self, texts: str) -> float:
        """Evaluate the model on a completion benchmark (e.g. LAMBADA) where the model must predict the last word of a passage."""
        targets_correct = 0
        for text in texts:
            context = " ".join(text.split()[:-1]) + " "

            full_tokens = self.tokenizer.encode(text)
            context_tokens = self.tokenizer.encode(context)
            target_tokens = full_tokens[len(context_tokens):]
            if not target_tokens:
                continue

            input_ids = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0).to(self.config.device)
            input_ids = input_ids[:, -self.config.max_seq_len:]

            tokens_correct = 0
            for token in target_tokens:
                with torch.no_grad():
                    logits = self.model(input_ids)[0]
                    next_token_logits = logits[-1, :]
                    predicted_token = torch.argmax(next_token_logits).item()
                    if predicted_token == token:
                        tokens_correct += 1
                        input_ids = torch.cat((input_ids, torch.tensor([[token]], dtype=torch.long).to(self.config.device)), dim=-1)
                        input_ids = input_ids[:, -self.config.max_seq_len:]
                    else:
                        break

            if tokens_correct == len(target_tokens):
                targets_correct += 1
        return targets_correct / len(texts)

    def load_hellaswag(self) -> tuple[list[str], list[list[str]], list[int]]:
        """Load and return the Hellaswag benchmark dataset from Rowan/hellaswag."""
        dataset = load_dataset("Rowan/hellaswag", split="validation")
        context = dataset["ctx"]
        endings = dataset["endings"]
        labels = dataset["label"]
        return context, endings, labels

    def load_mmlu(self) -> tuple[list[str], list[list[str]], list[int]]:
        """Load and return the MMLU from cais/mmlu."""
        dataset = load_dataset("cais/mmlu", "all", split="test")
        context = [q + "\nAnswer: " for q in dataset["question"]]
        endings = dataset["choices"]
        labels = dataset["answer"]
        return context, endings, labels

    def load_lambada(self) -> tuple[list[str], list[list[str]], list[int]]:
        """Load and return the LAMBADA benchmark dataset from EleutherAI/lambada_openai."""
        dataset = load_dataset("EleutherAI/lambada_openai", split="test")
        text = dataset["text"]
        return text

    def print_evaluation_results(self, perplexity: float, hellaswag_acc: float, mmlu_acc: float, lambada_acc: float) -> None:
        """Print the evaluation results in a readable format."""
        border = "=" * 60
        print(border)
        print(f"{'Evaluation Results':^60}")
        print(border)
        print(f"{'Perplexity':<30}: {perplexity:.4f}")
        print(f"{'Hellaswag Accuracy':<30}: {hellaswag_acc:.2%}")
        print(f"{'MMLU Accuracy':<30}: {mmlu_acc:.2%}")
        print(f"{'LAMBADA Accuracy':<30}: {lambada_acc:.2%}")
        print(border)