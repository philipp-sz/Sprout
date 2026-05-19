from pathlib import Path

import torch

from sprout.eval import Evaluator
from sprout.base.model import TransformerDecoder
from sprout.base.train import Config
from sprout.tokenizer import FastBPETokenizer


checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "base" / "model_final.pt"
tokenizer_path = Path(__file__).parent.parent / "tokenizer" / "bpe_tokenizer_state.json"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

config = Config(**checkpoint["config"])
config.convert_paths()

model = TransformerDecoder(
    num_layers=config.num_layers,
    d_model=config.d_model,
    num_heads=config.num_heads,
    max_seq_len=config.max_seq_len,
    vocab_size=config.vocab_size,
    dropout=config.dropout
).to(config.device)
model = model.to(torch.bfloat16)
model = torch.compile(model)
model.load_state_dict(checkpoint["model"])

tokenizer = FastBPETokenizer.load(tokenizer_path)
evaluator = Evaluator(model, tokenizer, config)

context, endings, labels = evaluator.load_hellaswag()
hellaswag_acc = evaluator.evaluate_knowledge_benchmark(context, endings, labels)
context, endings, labels = evaluator.load_mmlu()
mmlu_acc = evaluator.evaluate_knowledge_benchmark(context, endings, labels)
lambada_texts = evaluator.load_lambada()
lambada_acc = evaluator.evaluate_completion_benchmark(lambada_texts)
perplexity = evaluator.calc_perplexity("The cat sat on the mat.")
evaluator.print_evaluation_results(perplexity, hellaswag_acc, mmlu_acc, lambada_acc)