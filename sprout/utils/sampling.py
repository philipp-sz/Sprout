import torch


def apply_top_k_top_p_filtering(logits: torch.Tensor, top_k: int = None, top_p: float = None) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k)
        logits[logits < top_k_values[:, -1, None]] = -float("inf")

    if top_p is not None and top_p > 0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone() # Keep the token that crosses the threshold
        sorted_indices_to_remove[:, 0] = False # Always keep the first token
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove) # Scatter back to original indices
        logits[indices_to_remove] = -float("inf")

    return logits

def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    for token_id in set(generated_ids):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits