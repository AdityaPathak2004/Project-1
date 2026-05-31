"""
dskit.viz — Publication-ready visualization helpers.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

sns.set_theme(style="whitegrid", palette="husl")


def plot_missing_values(df: pd.DataFrame, figsize=(12, 5)):
    """Bar chart of missing value percentages per column."""
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
        return None
    fig, ax = plt.subplots(figsize=figsize)
    missing.plot(kind="bar", color="salmon", edgecolor="black", ax=ax)
    ax.set_title("Missing Values by Column (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Missing %")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, figsize=(12, 9), method="pearson"):
    """Heatmap of the correlation matrix for numeric columns."""
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5, ax=ax,
    )
    ax.set_title(f"Correlation Matrix ({method.capitalize()})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_distributions(df: pd.DataFrame, cols: Optional[List[str]] = None, figsize_per=(4, 4)):
    """Grid of distribution plots for numeric columns."""
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    if not cols:
        return None
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
    plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_model_comparison(results_df: pd.DataFrame, metric_col="Mean Score", figsize=(10, 6)):
    """Horizontal bar chart comparing model performance."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(results_df))
    bars = ax.barh(results_df["Model"], results_df[metric_col], color=colors, edgecolor="black")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_xlabel(metric_col)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_feature_importance(importances_df: pd.DataFrame, top_n: int = 20, figsize=(10, 8)):
    """Horizontal bar chart of top feature importances."""
    top = importances_df.head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("viridis", len(top))
    bars = ax.barh(top["Feature"], top["Importance"], color=colors, edgecolor="black")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig
