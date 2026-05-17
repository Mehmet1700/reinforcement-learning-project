"""Shared plotting helpers for training curves and result comparisons."""

import numpy as np
import matplotlib.pyplot as plt


def plot_training_curve(
    returns: list,
    label: str,
    window: int = 100,
    ax: plt.Axes = None,
    save_path: str = None,
) -> plt.Axes:
    """Plot a smoothed training curve with raw returns in the background."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    returns = np.array(returns)
    episodes = np.arange(1, len(returns) + 1)

    ax.plot(episodes, returns, alpha=0.2, color="tab:blue")

    if len(returns) >= window:
        smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1:], smoothed, label=label, color="tab:blue")
    else:
        ax.plot(episodes, returns, label=label, color="tab:blue")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)

    return ax


def plot_comparison(
    results_dict: dict,
    metric: str = "mean_return",
    title: str = "Algorithm Comparison",
    save_path: str = None,
) -> plt.Axes:
    """Bar chart comparing eval metrics across algorithms."""
    labels = list(results_dict.keys())
    values = [results_dict[k][metric] for k in labels]

    _, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color="tab:blue", alpha=0.8)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)

    return ax
