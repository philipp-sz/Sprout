import json
import os
from pathlib import Path
import time

from tqdm import trange
import torch
import torch.nn.functional as F

from sprout.eval import Evaluator
from sprout.modern.config import Config
from sprout.tokenizer import FastBPETokenizer


class TrainingLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.records = []
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, train_loss: float, eval_loss: float, lr: float, grad_norm: float, token_per_sec: float, benchmark_results: dict[str, float], time_since_start: float):
        if not self.records and os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                loaded_records = json.load(f)

            # Remove records that are at or after the current step to avoid duplicates when resuming from a checkpoint
            self.records = [record for record in loaded_records if record["step"] < step]

        self.records.append({
            "step": step,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "token_per_sec": token_per_sec,
            "benchmarks": benchmark_results,
            "time_since_start": time_since_start
        })
        with open(self.log_path, "w") as f:
            json.dump(self.records, f, indent=2)

class Trainer:
    """
    Trainer class to handle the training loop, evaluation, and checkpointing.
    Args:
        model: The TransformerDecoder model to train.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        optimizer: The optimizer to use for training.
        config: Configuration object containing hyperparameters and settings.
    """
    _ARCHITECTURE_FIELDS = {
        "d_model", "num_heads", "num_kv_heads", "num_layers",
        "max_seq_len", "vocab_size", "mlp_expansion", "rope_base_freq",
    }

    def __init__(self, model, train_loader, val_loader, config, optimizers: tuple | list | None = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.start_step = 1
        self.start_time = time.perf_counter()
        self.logger = TrainingLogger(config.log_path)
        self.tokenizer = FastBPETokenizer.load(config.tokenizer_path)
        self.evaluator = Evaluator(model, self.tokenizer, device=config.device, prefix_token_id=self.tokenizer.eot_token)

        raw_optims = optimizers if optimizers is not None else model.configure_optimizers(config)
        if isinstance(raw_optims, (list, tuple)):
            self.optimizers = list(raw_optims)
        else:
            self.optimizers = [raw_optims]

        assert self.config.checkpoint_interval % self.config.eval_interval == 0, "Checkpoint interval must be a multiple of evaluation interval for consistent logging."

    def train(self) -> None:
        """Main training loop that iterates over the specified number of iterations, performs training steps, evaluates periodically, and saves checkpoints."""
        self._print_summary()
        tokens_since_last_eval = 0
        last_eval_time = time.perf_counter()
        for step in trange(self.start_step, self.config.max_iterations + 1, desc="Training..."):
            train_loss, grad_norm, step_tokens = self._perform_train_step(step)
            tokens_since_last_eval += step_tokens

            if step % self.config.eval_interval == 0:
                now = time.perf_counter()
                token_per_sec = tokens_since_last_eval / (now - last_eval_time)
                time_since_start = now - self.start_time

                eval_loss, benchmark_results = self.evaluate()
                self.logger.log(step, train_loss, eval_loss, self.config.get_lr(step), grad_norm, token_per_sec, benchmark_results, time_since_start)
                print(f"Step {step}: Train Loss = {train_loss:.4f}, Eval Loss = {eval_loss:.4f}, Tokens/sec = {token_per_sec:.2f}")

                tokens_since_last_eval = 0
                last_eval_time = now

            if step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(train_loss, eval_loss, step)
        self._save_final()

    def evaluate(self) -> tuple[float, dict[str, float]]:
        """Evaluates the model on the validation set and returns the average loss and benchmark results."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for _ in trange(self.config.batches_in_val, desc="Evaluating...", leave=False):
                x, y = self.val_loader.get_batch()
                with torch.autocast(device_type=self.config.device, dtype=self.config.dtype):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                total_loss += loss.item()
        eval_loss = total_loss / self.config.batches_in_val
        benchmark_results = self.evaluator.run_standard_suite(total_samples=self.config.eval_benchmark_samples)
        return eval_loss, benchmark_results
    
    def _perform_train_step(self, step: int) -> tuple[float, float, int]:
        train_loss = 0.0
        self.model.train()
        step_tokens = 0
        for optim in self.optimizers:
            optim.zero_grad(set_to_none=True)

        for _ in range(self.config.grad_accum_steps):
            x, y = self.train_loader.get_batch()
            with torch.autocast(device_type=self.config.device, dtype=self.config.dtype):
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1), ignore_index=-100)
                loss = loss / self.config.grad_accum_steps
            loss.backward()
            train_loss += loss.item() # Keep scaled down so it reflects the average loss per micro batch
            step_tokens += x.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        lr = self.config.get_lr(step)
        for optim in self.optimizers:
            for param_group in optim.param_groups:
                param_group["lr"] = lr
            optim.step()

        return train_loss, grad_norm.item(), step_tokens

    def _save_checkpoint(self, total_train_loss: float, total_eval_loss: float, step_num: int):
        """Saves the model checkpoint to disk, including model state, optimizer state, config, and training statistics."""
        state_dict = {
            "model": self.model.state_dict(),
            "optimizers_states": [optim.state_dict() for optim in self.optimizers],
            "config": self.config.to_dict(),
            "train_loader_state": self.train_loader.get_state,
            "val_loader_state": self.val_loader.get_state,
            "total_train_loss": total_train_loss,
            "total_eval_loss": total_eval_loss,
            "step_num": step_num,
            "time_since_start": time.perf_counter() - self.start_time
        }
        checkpoint_path = self.config.save_dir / f"checkpoint_step_{step_num}.pt"
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, checkpoint_path)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, model_class, config, train_loader, val_loader, is_final: bool = False):
        """Creates a Trainer instance from a saved checkpoint, restoring model state, optimizer state, and training progress."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        for field in cls._ARCHITECTURE_FIELDS:
            setattr(config, field, checkpoint["config"][field])
        config.__post_init__()
        config.convert_paths() # Convert string paths back to Path objects

        model = model_class(config).to(config.device)
        model = torch.compile(model)
        model.load_state_dict(checkpoint["model"])

        raw_optims = model.configure_optimizers(config)
        if isinstance(raw_optims, (list, tuple)):
            optimizers = list(raw_optims)
        else:
            optimizers = [raw_optims]

        if not is_final:
            if "optimizers_states" in checkpoint:
                for optim, optim_state in zip(optimizers, checkpoint["optimizers_states"]):
                    optim.load_state_dict(optim_state)
            else: # Backward compatibility for old checkpoints
                if "adamw_optimizer" in checkpoint and len(optimizers) >= 1:
                    optimizers[0].load_state_dict(checkpoint["adamw_optimizer"])
                if "muon_optimizer" in checkpoint and len(optimizers) >= 2:
                    optimizers[1].load_state_dict(checkpoint["muon_optimizer"])

            train_loader.set_state = checkpoint["train_loader_state"]
            val_loader.set_state = checkpoint["val_loader_state"]
        trainer = cls(model, train_loader, val_loader, config, optimizers)
        if not is_final:
            trainer.start_step = checkpoint["step_num"] + 1
            trainer.start_time = time.perf_counter() - checkpoint.get("time_since_start", 0)
            print(f"Resumed training from checkpoint at step {checkpoint['step_num']}.")
        else:
            trainer.start_step = 1
            trainer.start_time = time.perf_counter()
            print(f"Initialized Trainer from final model checkpoint. Starting new training run from step 1.")

        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return trainer

    def _print_summary(self):
        """Prints a startup summary of the training and model configuration."""
        border = "=" * 60
        print(border)
        print(f"{'TRAINING PIPELINE INITIALIZED':^60}")
        print(border)
        print(f"\tDevice/Dtype: {self.config.device} / {self.config.dtype}")
        print(f"\tModel Parameters: {self.model.num_parameters():,}")
        print(f"\tMax Iterations: {self.config.max_iterations:,}")
        print(f"\tWarmup Steps: {self.config.warmup_steps:,}")
        print(f"\tEffective Batch Size: {self.config.effective_batch_size:,} tokens per iteration")
        print(border)

    def _save_final(self):
        torch.save({
            "model": self.model.state_dict(),
            "config": self.config.to_dict(),
        }, self.config.save_dir / "model_final.pt")