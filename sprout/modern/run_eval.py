from pathlib import Path

import torch

from sprout.eval import Evaluator
from sprout.modern.model import Config, TransformerDecoder
from sprout.tokenizer import FastBPETokenizer
from sprout.data import TokenDataLoader


checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "modern" / "test_medium_model" / "model_final.pt"
log_path = Path(__file__).parent.parent / "logs" / "eval" / "modern_test_medium_model_eval_results.json"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

model_config = Config(**checkpoint["config"])
model_config.convert_paths()

model = TransformerDecoder(model_config).to(model_config.device)
model = model.to(torch.bfloat16)
model = torch.compile(model)
model.load_state_dict(checkpoint["model"])

tokenizer = FastBPETokenizer.load(model_config.tokenizer_path)
data_loader = TokenDataLoader(model_config.shards_dir, batch_size=8, seq_len=model_config.max_seq_len, device=model_config.device, split="val")
evaluator = Evaluator(model, tokenizer, device=model_config.device, prefix_token_id=tokenizer.eot_token)

input_tokens, label_tokens = data_loader.get_batch()
perplexity = evaluator.calc_perplexity(input_tokens, label_tokens)

results = evaluator.run_standard_suite()

evaluator.print_evaluation_results(
    lambada_acc=results["lambada"], 
    arc_easy_acc=results["arc_easy"], 
    arc_challenge_acc=results["arc_challenge"], 
    hellaswag_acc=results["hellaswag"], 
    perplexity=perplexity
)

evaluator.save(
    path=log_path,
    lambada_acc=results["lambada"],
    arc_easy_acc=results["arc_easy"],
    arc_challenge_acc=results["arc_challenge"],
    hellaswag_acc=results["hellaswag"]
)