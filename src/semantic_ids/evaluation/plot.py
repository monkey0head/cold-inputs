import math

import matplotlib.pyplot as plt
import numpy as np


def plot_codes_distribution(
    level_to_counts: dict[str, np.ndarray],
    figsize: tuple[float, float] = (15, 3),
    title: str | None = None,
):
    """Plot the usage histogram of codebook entries for each level in subplots.

    Args:
        level_to_counts: mapping from level name to count arrays, 
            as returned by `get_codes_distribution()`.
        figsize: base figure size for the grid (width, height per row).
        title: optional main title for the figure.
    """
    num_levels = len(level_to_counts)    
    fig, axes = plt.subplots(num_levels, 1, figsize=(figsize[0], figsize[1] * num_levels), squeeze=False)
    axes = axes.flatten()

    if title:
        fig.suptitle(title, fontsize=16)

    for i, (level, counts) in enumerate(level_to_counts.items()):
        ax = axes[i]
        ax.bar(range(len(counts)), counts)
        ax.set_title(f"Level: {level}")
        ax.set_xlabel("Code")
        ax.set_ylabel("Count")
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    return fig


def plot_level_histograms(
    level_to_values: dict[str, np.ndarray | list[float]],
    bins: int = 50,
    figsize: tuple[float, float] = (15, 5),
    cols: int = 4,
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    log_y: bool = False,
    exclude_lower_than: float | None = None,
    title: str | None = None,
):
    """Plot histograms for a metric computed per level in subplots.

    Args:
        level_to_values: mapping from level name to an array of values.
        bins: number of histogram bins.
        figsize: base figure size for the grid (width, height per row).
        cols: number of columns in the subplot grid.
        xlabel: label for the X axis.
        ylabel: label for the Y axis.
        log_y: if True, plots the Y axis on a logarithmic scale.
        exclude_lower_than: if set, values less than this threshold will be excluded from the plot.
        title: optional main title for the figure.
    """
    num_levels = len(level_to_values)
    rows = math.ceil(num_levels / cols)
    cols = min(cols, num_levels)

    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows), squeeze=False)
    axes = axes.flatten()

    if title:
        fig.suptitle(title, fontsize=16)

    for i, (level, values) in enumerate(level_to_values.items()):
        ax = axes[i]
        # Filter out NaNs if any exist
        clean_values = np.array(values)
        clean_values = clean_values[~np.isnan(clean_values)]
        if exclude_lower_than is not None:
            clean_values = clean_values[clean_values >= exclude_lower_than]

        if len(clean_values) > 0:
            ax.hist(clean_values, bins=bins, alpha=0.7, edgecolor='black', log=log_y)

        ax.set_title(f"Level: {level}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for j in range(num_levels, len(axes)):
        fig.delaxes(axes[j])
 
    plt.tight_layout()

    return fig
