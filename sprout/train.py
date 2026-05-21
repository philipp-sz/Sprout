import json
import os
from pathlib import Path
from dataclasses import asdict

from tqdm import trange
import torch
import torch.nn.functional as F


class TrainingLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.records = []
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, train_loss: float, eval_loss: float, lr: float, grad_norm: float):
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
    def __init__(self, model, train_loader, val_loader, config, optimizers: tuple | None = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.adamw_optim, self.muon_optim = optimizers if optimizers is not None else model.configure_optimizers(config)
        self.config = config
        self.start_step = 1
        self.logger = TrainingLogger(config.log_path)
        assert self.config.checkpoint_interval % self.config.eval_interval == 0, "Checkpoint interval must be a multiple of evaluation interval for consistent logging."

    def train(self) -> None:
        """Main training loop that iterates over the specified number of iterations, performs training steps, evaluates periodically, and saves checkpoints."""
        self._print_summary()
        for step in trange(self.start_step, self.config.max_iterations + 1, desc="Training..."):
            train_loss = 0.0
            self.model.train()
            self.adamw_optim.zero_grad(set_to_none=True)
            self.muon_optim.zero_grad(set_to_none=True)

            for _ in range(self.config.grad_accum_steps):
                x, y = self.train_loader.get_batch()
                with torch.autocast(device_type=self.config.device, dtype=self.config.dtype):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                    loss = loss / self.config.grad_accum_steps
                loss.backward()
                train_loss += loss.item() # Keep scaled down so it reflects the average loss per micro batch

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            lr = self.config.get_lr(step)
            for opt in [self.muon_optim, self.adamw_optim]:
                for param_group in opt.param_groups:
                    param_group["lr"] = lr

            self.adamw_optim.step()
            self.muon_optim.step()

            if step % self.config.eval_interval == 0:
                eval_loss = self.evaluate()
                self.logger.log(step, train_loss, eval_loss, lr, grad_norm.item())
                print(f"Step {step}: Train Loss = {train_loss:.4f}, Eval Loss = {eval_loss:.4f}")

            if step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(train_loss, eval_loss, step)
        self._save_final()

    def evaluate(self) -> float:
        """Evaluates the model on the validation set and returns the average loss."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for _ in trange(self.config.batches_in_val, desc="Evaluating...", leave=False):
                x, y = self.val_loader.get_batch()
                with torch.autocast(device_type=self.config.device, dtype=self.config.dtype):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), y.view(-1))
                total_loss += loss.item()
        return total_loss / self.config.batches_in_val

    def _save_checkpoint(self, total_train_loss, total_eval_loss, step_num):
        """Saves the model checkpoint to disk, including model state, optimizer state, config, and training statistics."""
        state_dict = {
            "model": self.model.state_dict(),
            "adamw_optimizer": self.adamw_optim.state_dict(),
            "muon_optimizer": self.muon_optim.state_dict(),
            "config": self.config.to_dict(),
            "train_loader_state": self.train_loader.get_state,
            "val_loader_state": self.val_loader.get_state,
            "total_train_loss": total_train_loss,
            "total_eval_loss": total_eval_loss,
            "step_num": step_num
        }
        checkpoint_path = self.config.save_dir / f"checkpoint_step_{step_num}.pt"
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, checkpoint_path)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, model_class, config_class, train_loader, val_loader):
        """Creates a Trainer instance from a saved checkpoint, restoring model state, optimizer state, and training progress."""
        checkpoint = torch.load(checkpoint_path)
        config = config_class(**checkpoint["config"])
        config.convert_paths() # Convert string paths back to Path objects
        model = model_class(config).to(config.device)
        model = torch.compile(model)
        model.load_state_dict(checkpoint["model"])

        adamw_optim, muon_optim = model.configure_optimizers(config)
        adamw_optim.load_state_dict(checkpoint["adamw_optimizer"])
        muon_optim.load_state_dict(checkpoint["muon_optimizer"])

        train_loader.set_state = checkpoint["train_loader_state"]
        val_loader.set_state = checkpoint["val_loader_state"]
        trainer = cls(model, train_loader, val_loader, config, (adamw_optim, muon_optim))
        trainer.start_step = checkpoint["step_num"] + 1
        print(f"Resumed training from checkpoint at step {checkpoint['step_num']}.")
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