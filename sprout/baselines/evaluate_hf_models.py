from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sprout.eval import Evaluator


log_path = Path(__file__).parent.parent / "logs" / "eval" / "gpt2_eval_results.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
evaluator = Evaluator(model=model, tokenizer=tokenizer, device=device)

results = evaluator.run_standard_suite()

evaluator.print_evaluation_results(
    lambada_acc=results["lambada"], 
    arc_easy_acc=results["arc_easy"], 
    arc_challenge_acc=results["arc_challenge"], 
    hellaswag_acc=results["hellaswag"], 
)

evaluator.save(
    path=log_path,
    lambada_acc=results["lambada"],
    arc_easy_acc=results["arc_easy"],
    arc_challenge_acc=results["arc_challenge"],
    hellaswag_acc=results["hellaswag"]
)