# %%
# ============================================================
# Plot decoding time-series
# One figure per comparison
# Rows = bands
# Cols = contrasts
# Category = "All" only
# ============================================================

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
result_path = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/TFdecoding/Results/TFclassDecoding_svm.pkl"
save_dir = "/Users/woojaejeong/Desktop/Data/USC/DARPA-NEAT/Code/TFdecoding/Figures/TFclassDecoding_svm"

category_to_plot = "All"

use_percent = True
show_std = True
show_sem = False
same_ylim = True
ylim = None  # example: (30, 80)
dpi = 150

os.makedirs(save_dir, exist_ok=True)

# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------
with open(result_path, "rb") as f:
    Result = pickle.load(f)

meta = Result["meta"]
times = np.asarray(meta["times"]) * 1000  # ms
band_names = list(meta["band_names"])
contrasts = list(meta["contrasts"])
comparisons = [k for k in Result.keys() if k != "meta"]

# ------------------------------------------------------------
# Pretty names
# ------------------------------------------------------------
comparison_titles = {
    "control_vs_depressed": "Control vs Depressed",
    "depressed_vs_suicidal": "Depressed vs Suicidal",
    "control_vs_suicidal": "Control vs Suicidal",
    "control_vs_depressedsuicidal": "Control vs Depressed+Suicidal",
}

contrast_titles = {
    "positive": "Positive",
    "negative": "Negative",
    "neg_minus_pos": "Negative - Positive",
}

band_titles = {
    "theta": "Theta",
    "alpha": "Alpha",
    "beta": "Beta",
    "gamma": "Gamma",
    "alpha_low": "Alpha low",
    "alpha_high": "Alpha high",
    "beta_low": "Beta low",
    "beta_mid": "Beta mid",
    "beta_high": "Beta high",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def maybe_percent(x):
    return x * 100 if use_percent else x


def get_mean_and_err(arr):
    """
    arr shape:
        (n_times, n_folds) or (n_times,)
    returns:
        mean, err with shape (n_times,)
    """
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        return arr, np.zeros_like(arr)

    mean_ = np.nanmean(arr, axis=1)

    if show_sem:
        n = np.sum(np.isfinite(arr), axis=1)
        err_ = np.nanstd(arr, axis=1, ddof=1) / np.sqrt(np.maximum(n, 1))
    elif show_std:
        err_ = np.nanstd(arr, axis=1, ddof=1)
    else:
        err_ = np.zeros_like(mean_)

    return mean_, err_


def get_global_ylim():
    vals = []

    for comp in comparisons:
        for con in contrasts:
            entry = Result[comp][category_to_plot][con]

            if "fold_scores" in entry:
                x = entry["fold_scores"]  # bands x times x folds
                vals.append(maybe_percent(x[np.isfinite(x)]))
            elif "score" in entry:
                x = entry["score"]  # bands x times
                vals.append(maybe_percent(x[np.isfinite(x)]))

    if len(vals) == 0:
        return (0, 1)

    vals = np.concatenate(vals)
    lo = np.nanmin(vals)
    hi = np.nanmax(vals)
    pad = 0.08 * (hi - lo + 1e-12)
    return (lo - pad, hi + pad)


# ------------------------------------------------------------
# Global y-limits
# ------------------------------------------------------------
global_ylim = ylim if ylim is not None else (get_global_ylim() if same_ylim else None)

# ------------------------------------------------------------
# Main plotting
# One figure per comparison
# Rows = bands
# Cols = contrasts
# ------------------------------------------------------------
for comp in comparisons:
    n_rows = len(band_names)
    n_cols = len(contrasts)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 2.8 * n_rows),
        sharex=True,
        sharey=same_ylim,
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, band in enumerate(band_names):
        for j, con in enumerate(contrasts):
            ax = axes[i, j]
            entry = Result[comp][category_to_plot][con]

            if "fold_scores" in entry:
                # shape: bands x times x folds
                x = entry["fold_scores"][i, :, :]
                x = maybe_percent(x)
                m, e = get_mean_and_err(x)
            else:
                # shape: bands x times
                x = entry["score"][i, :]
                x = maybe_percent(x)
                m, e = get_mean_and_err(x)

            ax.plot(times, m, linewidth=2)
            ax.fill_between(times, m - e, m + e, alpha=0.35, linewidth=0)

            ax.axhline(
                50 if use_percent else 0.5,
                linestyle="--",
                linewidth=2,
                color="k",
            )
            ax.axvline(
                0,
                linestyle=":",
                linewidth=1.5,
                color="k",
                alpha=0.8,
            )

            if i == 0:
                ax.set_title(
                    contrast_titles.get(con, con),
                    fontsize=12,
                    fontweight="bold",
                )

            if j == 0:
                ylabel = (
                    f"{band_titles.get(band, band)}\nBalanced accuracy (%)"
                    if use_percent
                    else f"{band_titles.get(band, band)}\nBalanced accuracy"
                )
                ax.set_ylabel(ylabel)

            if i == n_rows - 1:
                ax.set_xlabel("Time (ms)")

            if global_ylim is not None:
                ax.set_ylim(global_ylim)

            ax.grid(False)

    fig.suptitle(
        f"{comparison_titles.get(comp, comp)} | Category: {category_to_plot}",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout()

    save_path = os.path.join(
        save_dir, f"{comp}_{category_to_plot}_bands_x_contrasts.png"
    )
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
