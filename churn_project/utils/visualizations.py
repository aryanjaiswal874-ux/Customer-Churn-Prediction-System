"""
Visualization Utilities for Churn Prediction App
All plot functions return Matplotlib Figure objects (Streamlit-compatible).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional, List

# --- Consistent color palette ---
CHURN_COLORS = {"Yes": "#EF4444", "No": "#22C55E"}
PALETTE = ["#3B82F6", "#EF4444", "#F59E0B", "#10B981", "#8B5CF6", "#EC4899"]


def set_plot_style():
    """Apply a clean, modern dark-ish style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#0F172A",
            "axes.facecolor": "#1E293B",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#CBD5E1",
            "xtick.color": "#94A3B8",
            "ytick.color": "#94A3B8",
            "text.color": "#E2E8F0",
            "grid.color": "#334155",
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "font.family": "DejaVu Sans",
        }
    )


def plot_churn_distribution(predictions: pd.Series) -> plt.Figure:
    """
    Bar chart showing count of Churn=Yes vs Churn=No.
    """
    set_plot_style()
    counts = predictions.value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))

    bars = ax.bar(
        counts.index,
        counts.values,
        color=[CHURN_COLORS.get(k, "#64748B") for k in counts.index],
        width=0.5,
        edgecolor="#1E293B",
        linewidth=1.5,
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + counts.max() * 0.02,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#E2E8F0",
        )

    total = len(predictions)
    churn_pct = (predictions == "Yes").sum() / total * 100
    ax.set_title(
        f"Churn Distribution  |  {churn_pct:.1f}% will churn",
        fontsize=13,
        fontweight="bold",
        color="#F1F5F9",
        pad=12,
    )
    ax.set_xlabel("Churn", fontsize=11)
    ax.set_ylabel("Customer Count", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_churn_probability_histogram(probabilities: np.ndarray) -> plt.Figure:
    """
    Histogram of churn probabilities across all customers.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    n, bins, patches = ax.hist(
        probabilities, bins=30, edgecolor="#0F172A", linewidth=0.5
    )

    # Color bars by risk zone
    for patch, left in zip(patches, bins[:-1]):
        if left < 0.33:
            patch.set_facecolor("#22C55E")
        elif left < 0.66:
            patch.set_facecolor("#F59E0B")
        else:
            patch.set_facecolor("#EF4444")

    ax.axvline(0.5, color="#F8FAFC", linestyle="--", linewidth=1.2, alpha=0.7, label="Decision boundary (0.5)")
    ax.set_title("Churn Probability Distribution", fontsize=13, fontweight="bold", color="#F1F5F9", pad=12)
    ax.set_xlabel("Predicted Churn Probability", fontsize=11)
    ax.set_ylabel("Number of Customers", fontsize=11)

    low = mpatches.Patch(color="#22C55E", label="Low risk (<0.33)")
    med = mpatches.Patch(color="#F59E0B", label="Medium risk (0.33–0.66)")
    high = mpatches.Patch(color="#EF4444", label="High risk (>0.66)")
    ax.legend(handles=[low, med, high], fontsize=8, framealpha=0.2, labelcolor="#CBD5E1")

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str], importances: np.ndarray, top_n: int = 15
) -> plt.Figure:
    """
    Horizontal bar chart of top-N feature importances.
    """
    set_plot_style()

    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx][::-1]
    vals = importances[idx][::-1]

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.38)))

    colors = [
        f"#{int(255*(1-v/vals.max())):02x}{int(180*v/vals.max()):02x}ff"
        for v in vals
    ]
    # Simpler: gradient from teal to orange
    norm_vals = vals / vals.max()
    bar_colors = [
        (1 - v) * np.array([0.13, 0.60, 0.95]) + v * np.array([0.94, 0.33, 0.23])
        for v in norm_vals
    ]

    bars = ax.barh(names, vals, color=bar_colors, edgecolor="#0F172A", linewidth=0.5)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold", color="#F1F5F9", pad=12)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, vals):
        ax.text(
            val + vals.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=8,
            color="#CBD5E1",
        )

    fig.tight_layout()
    return fig


def plot_model_comparison(results: dict) -> plt.Figure:
    """
    Grouped bar chart comparing model metrics.
    results = {model_name: {metric: value, ...}}
    """
    set_plot_style()
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (model, scores) in enumerate(results.items()):
        vals = [scores.get(m, 0) for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9, label=model, color=PALETTE[i % len(PALETTE)])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#E2E8F0",
            )

    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold", color="#F1F5F9", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="#CBD5E1")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_risk_segmentation(probabilities: np.ndarray) -> plt.Figure:
    """
    Pie chart showing Low / Medium / High risk segments.
    """
    set_plot_style()
    low = (probabilities < 0.33).sum()
    med = ((probabilities >= 0.33) & (probabilities < 0.66)).sum()
    high = (probabilities >= 0.66).sum()

    labels = [f"Low Risk\n{low:,}", f"Medium Risk\n{med:,}", f"High Risk\n{high:,}"]
    sizes = [low, med, high]
    colors = ["#22C55E", "#F59E0B", "#EF4444"]
    explode = [0.03, 0.03, 0.07]

    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        explode=explode,
        startangle=140,
        textprops={"color": "#E2E8F0", "fontsize": 9},
        wedgeprops={"edgecolor": "#0F172A", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("#0F172A")
        at.set_fontweight("bold")

    ax.set_title("Customer Risk Segmentation", fontsize=13, fontweight="bold", color="#F1F5F9", pad=12)
    fig.tight_layout()
    return fig
