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

_MODEL_PALETTE = [
    "#4a9eff", "#ff6b6b", "#34d399", "#a78bfa",
    "#f59e0b", "#f472b6", "#60a5fa", "#a3e635",
]

_BENCH_DISPLAY = {
    "lambada":       "LAMBADA",
    "arc_easy":      "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "hellaswag":     "HellaSwag",
    "average":       "Average",
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

def plot_model_comparison(results: dict[str, str | Path], out_path: Optional[str | Path] = None) -> plt.Figure:
    """Plot a vertical bar chart comparing final benchmark accuracies across multiple models."""
    if not results:
        raise ValueError("results dict is empty — nothing to plot.")

    model_names = list(results.keys())
    n_models = len(model_names)

    # Load and validate
    data: dict[str, dict[str, float]] = {}
    for label, path in results.items():
        raw = load_eval_results(path)
        # Normalise key names: accept both "hellaswag_acc" and "hellaswag"
        normalised: dict[str, float] = {}
        for key in _BENCH_KEYS:
            for candidate in (f"{key}_acc", key):
                if candidate in raw:
                    normalised[key] = raw[candidate]
                    break
        if "average_acc" in raw:
            normalised["average"] = raw["average_acc"]
        elif "average" in raw:
            normalised["average"] = raw["average"]
        else:
            present = [v for v in normalised.values()]
            normalised["average"] = float(np.mean(present)) if present else float("nan")
        data[label] = normalised

    # Benchmark order: individual first, average last (displayed left-to-right)
    bench_order = _BENCH_KEYS + ["average"]
    bench_labels = [_BENCH_DISPLAY[b] for b in bench_order]
    n_benches = len(bench_order)

    bar_w      = 0.65 / n_models   # bar width shrinks as models are added
    cluster_w  = 1.0               # spacing between benchmark clusters
    colors     = _MODEL_PALETTE[:n_models]

    # Adjust width dynamically based on the number of benchmarks and models
    fig_width = max(9, 1.25 * n_benches + 0.7 * n_models)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        # Centre positions for each benchmark cluster
        cluster_centres = np.arange(n_benches) * cluster_w

        for m_idx, (label, model_data) in enumerate(data.items()):
            # Offset each model's bars within the cluster
            offset = (m_idx - (n_models - 1) / 2) * bar_w
            x_positions = cluster_centres + offset
            values = [model_data.get(b, float("nan")) for b in bench_order]

            bars = ax.bar(
                x_positions, values,
                width=bar_w * 0.88,
                color=colors[m_idx],
                alpha=0.85,
                label=label,
            )

            # Value annotations
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.008, 
                        f"{val * 100:.1f}%",
                        va="bottom", ha="center",
                        fontsize=7.5,
                        color="#e0e0e0",  # Brighter off-white for cleaner dark-theme contrast
                        rotation=60,
                    )

        # Draw a subtle vertical separator to the left of the Average column
        sep_x = (cluster_centres[-1] + cluster_centres[-2]) / 2
        ax.axvline(sep_x, color="#333333", lw=0.8, ls="--")

        ax.set_xticks(cluster_centres)
        ax.set_xticklabels(bench_labels, fontsize=10, rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Comparison — Benchmark Accuracy")
        
        ax.set_ylim(0, min(1.0, ax.get_ylim()[1] + 0.12))
        
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
        ax.legend(loc="upper right", framealpha=0.8)
        ax.grid(True, axis="y")
        fig.tight_layout()

    _save(fig, out_path)
    return fig

def plot_sft_filter_stats(df: "pd.DataFrame", out_path: Optional[str | Path] = None) -> plt.Figure:
    """Stacked horizontal bar chart showing sample counts per source, split into kept vs. filtered samples."""
    counts = (
        df.groupby(["source", "filtered"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={False: "kept", True: "filtered_out"})
    )
    # Ensure both columns exist even if filtering removed nothing / everything
    for col in ("kept", "filtered_out"):
        if col not in counts.columns:
            counts[col] = 0

    # Sort sources by total descending so the longest bar is at the top
    counts["total"] = counts["kept"] + counts["filtered_out"]
    counts = counts.sort_values("total", ascending=True)

    sources    = counts.index.tolist()
    kept       = counts["kept"].values
    filtered   = counts["filtered_out"].values
    totals     = counts["total"].values
    n          = len(sources)

    bar_h     = 0.65  
    fig_h     = max(4, 0.7 * n + 1.2)

    _C_KEPT     = "#34d399"
    _C_FILTERED = "#ff6b6b"

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, fig_h))

        y = np.arange(n)

        ax.set_axisbelow(True)
        ax.grid(True, axis="x", color="#333333", linestyle=":", alpha=0.6)

        # Stacked: kept first (left), filtered on top (right)
        ax.barh(y, kept,     height=bar_h, color=_C_KEPT,     alpha=0.9, label="Kept")
        ax.barh(y, filtered, height=bar_h, color=_C_FILTERED, alpha=0.9,
                left=kept, label="Filtered out")

        max_total = totals.max()
        text_threshold = 0.05 * max_total

        for i, (k, f, t) in enumerate(zip(kept, filtered, totals)):
            # Kept count inside the kept bar (only if wide enough)
            if k > text_threshold:
                ax.text(k / 2, i, f"{k:,}",
                        va="center", ha="center", fontsize=9, color="#111111", fontweight="bold")
            
            if f > text_threshold:
                ax.text(k + f / 2, i, f"{f:,}",
                        va="center", ha="center", fontsize=9, color="#111111", fontweight="bold")
            
            pct = 100 * k / t if t > 0 else 0.0
            ax.text(t + max_total * 0.015, i, f"{t:,}  ({pct:.1f}% kept)",
                    va="center", ha="left", fontsize=9, color="#cccccc")

        ax.set_yticks(y)
        ax.set_yticklabels(sources, fontsize=10)
        ax.set_xlabel("Number of samples", fontsize=10, labelpad=8)
        ax.set_title("SFT Dataset — Samples per Source (before / after length filter)", fontsize=12, pad=16, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        
        # Adjust right margin dynamically to guarantee no text clipping
        ax.set_xlim(0, max_total * 1.32)
        
        # Subtle legend background blending
        ax.legend(loc="lower right", framealpha=0.2, edgecolor="none")
        fig.tight_layout()

    _save(fig, out_path)
    return fig

if __name__ == "__main__":
    plot_all(
        log_path=Path(__file__).parent.parent / "logs" / "train" / "sapling_training_log.json",
        eval_results_path=Path(__file__).parent.parent / "logs" / "eval" / "modern_sapling_eval_results.json",
        out_dir=Path(__file__).parent.parent / "logs" / "plots" / "sapling",
        smooth_alpha=0.30
    )
    plot_model_comparison(
        results={
            "GPT-2 Small": Path(__file__).parent.parent / "logs" / "eval" / "gpt2_eval_results.json",
            "Sprout Vanilla": Path(__file__).parent.parent / "logs" / "eval" / "vanilla_eval_results.json",
            "Sprout Modern (untuned)": Path(__file__).parent.parent / "logs" / "eval" / "modern_untuned_eval_results.json",
            "Sprout Modern (sapling)": Path(__file__).parent.parent / "logs" / "eval" / "modern_sapling_eval_results.json",
        },
        out_path=Path(__file__).parent.parent / "logs" / "plots" / "model_comparison.png",
    )