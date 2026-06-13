from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


_STYLE: dict = {
    "figure.facecolor": "#111111",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#2e2e2e",
    "axes.labelcolor":  "#c8c8c8",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "xtick.color":      "#777777",
    "ytick.color":      "#777777",
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "grid.color":       "#262626",
    "grid.linestyle":   "-",
    "grid.linewidth":   0.6,
    "text.color":       "#c8c8c8",
    "legend.facecolor": "#1e1e1e",
    "legend.edgecolor": "#2e2e2e",
    "legend.fontsize":  9,
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titlepad":    10,
    "axes.labelsize":   9,
    "figure.dpi":       120,
}

_ALPHA_RAW    = 0.20
_ALPHA_SMOOTH = 0.92
_LW_RAW       = 0.9
_LW_SMOOTH    = 2.0

_C = {
    "train_loss":    "#4a9eff",
    "eval_loss":     "#ff6b6b",
    "grad_norm":     "#a78bfa",
    "throughput":    "#34d399",
    "lambada":       "#f59e0b",
    "arc_easy":      "#60a5fa",
    "arc_challenge": "#f472b6",
    "hellaswag":     "#a3e635",
    "average":       "#e5e5e5",
}

_BENCH_KEYS = ["lambada", "arc_easy", "arc_challenge", "hellaswag"]


def load_training_log(path: str | Path) -> pd.DataFrame:
    """Load a training log JSON (list of step-dicts) into a DataFrame with flattened benchmarks."""
    with open(path, "r") as f:
        train_data = json.load(f)

    df = pd.DataFrame(train_data)
    benchmarks_df = pd.json_normalize(df["benchmarks"])
    average = benchmarks_df[_BENCH_KEYS].mean(axis=1)
    benchmarks_df["average"] = average
    df = pd.concat([df.drop(columns=["benchmarks"]), benchmarks_df], axis=1)
    return df

def load_eval_results(path: str | Path) -> dict[str, float]:
    """Load the final evaluation results JSON into a dictionary."""
    with open(path) as f:
        return json.load(f)

def _save(fig: plt.Figure, out_path: Optional[str | Path]) -> None:
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {out_path}")

def plot_loss(log_df: dict[str, np.ndarray], alpha: float = 0.06, out_path: Optional[str | Path] = None) -> plt.Figure:
    """Plot train loss and eval loss vs. steps."""
    steps = log_df["step"]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))

        for key, label, color in (
            ("train_loss", "Train loss", _C["train_loss"]),
            ("eval_loss",  "Eval loss",  _C["eval_loss"]),
        ):
            ax.plot(steps, log_df[key],
                    color=color, alpha=_ALPHA_RAW, lw=_LW_RAW)
            ax.plot(steps, log_df[key].ewm(alpha=alpha).mean(),
                    color=color, alpha=_ALPHA_SMOOTH, lw=_LW_SMOOTH, label=label)

        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title("Training & Eval Loss")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

    _save(fig, out_path)
    return fig

def plot_grad_norm(log_df: dict[str, np.ndarray], alpha: float = 0.06, out_path: Optional[str | Path] = None) -> plt.Figure:
    """Plot gradient norm and eval loss vs. steps."""
    steps = log_df["step"]

    with plt.rc_context(_STYLE):
        fig, ax1 = plt.subplots(figsize=(10, 4))

        ax1.plot(steps, log_df["grad_norm"],
                 color=_C["grad_norm"], alpha=_ALPHA_RAW, lw=_LW_RAW)
        ax1.plot(steps, log_df["grad_norm"].ewm(alpha=alpha).mean(),
                 color=_C["grad_norm"], alpha=_ALPHA_SMOOTH, lw=_LW_SMOOTH,
                 label="Grad norm")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Gradient norm", color=_C["grad_norm"])
        ax1.tick_params(axis="y", labelcolor=_C["grad_norm"])

        ax2 = ax1.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color("#2e2e2e")
        ax2.plot(steps, log_df["eval_loss"],
                 color=_C["eval_loss"], alpha=0.45, lw=1.4, label="Eval loss")
        ax2.set_ylabel("Eval loss", color=_C["eval_loss"])
        ax2.tick_params(axis="y", labelcolor=_C["eval_loss"])

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper right")

        ax1.set_title("Gradient Norm")
        ax1.grid(True)
        fig.tight_layout()

    _save(fig, out_path)
    return fig

def plot_throughput(log_df: dict[str, np.ndarray], alpha: float = 0.12, out_path: Optional[str | Path] = None) -> plt.Figure:
    """Tokens per second vs. steps."""
    steps = log_df["step"]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 3))

        ax.plot(steps, log_df["token_per_sec"],
                color=_C["throughput"], alpha=_ALPHA_RAW, lw=_LW_RAW)
        ax.plot(steps, log_df["token_per_sec"].ewm(alpha=alpha).mean(),
                color=_C["throughput"], alpha=_ALPHA_SMOOTH, lw=_LW_SMOOTH,
                label="Tokens / sec")

        ax.set_xlabel("Step")
        ax.set_ylabel("Tokens / second")
        ax.set_title("Training Throughput")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x / 1_000:.1f}k")
        )
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

    _save(fig, out_path)
    return fig

def plot_benchmarks(log_df: dict[str, np.ndarray], eval_results: Optional[dict[str, float]] = None,
                    alpha: float = 0.08, out_path: Optional[str | Path] = None) -> plt.Figure:
    """Plot benchmark accuracies vs. steps, with optional final eval results as scatter points."""
    steps = log_df["step"]
    last_step = steps.max()

    # Per-step composite average (NaN-safe)
    bench_matrix = np.stack([log_df[k] for k in _BENCH_KEYS], axis=1)  # (T, 4)
    step_avg = np.nanmean(bench_matrix, axis=1)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))

        for key in _BENCH_KEYS:
            label = key.replace("_", " ").title()
            ax.plot(steps, log_df[key],
                    color=_C[key], alpha=0.40, lw=1.3, label=label)

        ax.plot(steps, log_df["average"].ewm(alpha=alpha).mean(),
                color=_C["average"], alpha=_ALPHA_SMOOTH, lw=2.2,
                label="Average (smoothed)")

        if eval_results:
            for key in _BENCH_KEYS:
                result_key = f"{key}_acc"
                if result_key in eval_results:
                    ax.scatter(
                        last_step, eval_results[result_key],
                        color=_C[key], s=70, zorder=5,
                        edgecolors="white", linewidths=0.6,
                    )
            # overall average point — diamond marker
            accs = [
                eval_results[f"{k}_acc"]
                for k in _BENCH_KEYS
                if f"{k}_acc" in eval_results
            ]
            if accs:
                ax.scatter(
                    last_step, np.mean(accs),
                    color=_C["average"], s=110, zorder=6,
                    marker="H", edgecolors="white", linewidths=0.6,
                )

        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_title("Benchmark Accuracy")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower right", ncol=2)
        ax.grid(True)
        fig.tight_layout()

    _save(fig, out_path)
    return fig

def plot_all(log_path: str | Path, eval_results_path: Optional[str | Path] = None, 
            out_dir: Optional[str | Path] = None, smooth_alpha: float = 0.06) -> None:
    """Convenience function to plot all metrics from training log and eval results."""
    print(f"Loading training log:  {log_path}")
    log = load_training_log(log_path)
    eval_results = None
    if eval_results_path is not None:
        print(f"Loading eval results:  {eval_results_path}")
        eval_results = load_eval_results(eval_results_path)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    def _out(name: str) -> Optional[Path]:
        return (out_dir / name) if out_dir else None

    plot_loss(log, alpha=smooth_alpha, out_path=_out("loss.png"))
    plot_grad_norm(log, alpha=smooth_alpha, out_path=_out("grad_norm.png"))
    plot_throughput(log, alpha=0.12, out_path=_out("throughput.png"))
    plot_benchmarks(log, eval_results, alpha=smooth_alpha, out_path=_out("benchmarks.png"))


if __name__ == "__main__":
    plot_all(
        log_path=Path(__file__).parent.parent / "logs" / "train" / "sapling_training_log.json",
        eval_results_path=Path(__file__).parent.parent / "logs" / "eval" / "modern_untuned_eval_results.json",
        out_dir=Path(__file__).parent.parent / "logs" / "plots" / "sapling",
        smooth_alpha=0.30
    )