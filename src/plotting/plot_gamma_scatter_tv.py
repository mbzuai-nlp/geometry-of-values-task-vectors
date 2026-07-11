#!/usr/bin/env python3
"""
Scatter plot of gamma parameters for high-accuracy task-vector results.

Reads a combined task-vector CSV and plots:
  x = gamma_instr
  y = gamma_pref
  point size proportional to test accuracy (percent)
  color by model size (1B vs 3B)

Outputs:
  results_tv/gamma_scatter_tv.(png|pdf)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TV_PATH = "results_tv/combined_task_vector_results_datav2.csv"
OUT_DIR = "results_tv"
OUT_BASE = "gamma_scatter_tv"

TOTAL_TEST_ITEMS = 400.0
TOP_PERCENTILE = 20
SIZE_ORDER = ["1B", "3B"]
SIZE_COLORS = {"1B": "#2F5AFF", "3B": "#FF3B3B"}
SIZE_LABELS = {"1B": "1B", "3B": "3B"}
JITTER = 0.02  # set to 0.0 to disable jitter
POINT_ALPHA = 0.6


def scale_sizes(values, min_size=20, max_size=120):
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return [min_size] * len(values)
    scaled = (values - vmin) / (vmax - vmin)
    return min_size + scaled * (max_size - min_size)

def main():
    if not os.path.exists(TV_PATH):
        raise SystemExit(f"Missing input CSV: {TV_PATH}")

    df = pd.read_csv(TV_PATH)
    df = df.dropna(subset=["gamma_instr", "gamma_pref", "test_accuracy", "size"]).copy()
    df["test_accuracy"] = df["test_accuracy"].astype(float)
    df["test_accuracy_pct"] = (df["test_accuracy"] / TOTAL_TEST_ITEMS) * 100.0

    cutoff = np.percentile(df["test_accuracy_pct"], TOP_PERCENTILE)
    df = df[df["test_accuracy_pct"] >= cutoff].copy()
    if df.empty:
        raise SystemExit("No rows after filtering for top percentile.")

    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 7))

    size_vals = df["test_accuracy_pct"]
    point_sizes = scale_sizes(size_vals)
    df = df.assign(point_size=point_sizes)

    for sz in SIZE_ORDER:
        sub = df[df["size"] == sz]
        if sub.empty:
            continue
        x = sub["gamma_instr"].to_numpy()
        y = sub["gamma_pref"].to_numpy()
        if JITTER > 0:
            x = x + np.random.uniform(-JITTER, JITTER, size=len(x))
            y = y + np.random.uniform(-JITTER, JITTER, size=len(y))
        ax.scatter(
            x,
            y,
            s=sub["point_size"],
            c=SIZE_COLORS.get(sz, "#888888"),
            edgecolors="black",
            linewidths=0.5,
            alpha=POINT_ALPHA,
            label=f"{SIZE_LABELS.get(sz, sz)} (n={len(sub)})",
            zorder=3,
        )

    ax.set_title(
        "Scatter Plot: Gamma Parameters for High-Accuracy Experiments\n"
        f"(Top {TOP_PERCENTILE}th percentile, Point size ∝ Accuracy)"
    )
    ax.set_xlabel("Gamma Instruction (gamma_instr)")
    ax.set_ylabel("Gamma Preference (gamma_pref)")
    ax.grid(alpha=0.25)

    acc_min = df["test_accuracy_pct"].min()
    acc_max = df["test_accuracy_pct"].max()
    ax.text(
        0.02,
        0.98,
        "Point sizes scaled by test accuracy\n"
        f"(Range: {acc_min:.1f}% - {acc_max:.1f}%)",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#f2e6c9", edgecolor="#777777"),
    )

    ax.legend(loc="lower left", frameon=True)

    png = os.path.join(OUT_DIR, f"{OUT_BASE}.png")
    pdf = os.path.join(OUT_DIR, f"{OUT_BASE}.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {png}\nSaved: {pdf}")


if __name__ == "__main__":
    main()
