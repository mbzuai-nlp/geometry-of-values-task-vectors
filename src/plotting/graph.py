#!/usr/bin/env python3
# graph.py — Streamlined two-panel heatmap for 3B vs 1B accuracies

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- 1. Read & combine data ---
df3 = pd.read_csv("finetuning_results_v2_3B_5epochs.csv")
df1 = pd.read_csv("finetuning_results_v2_1B_5epochs.csv")

df = pd.concat([df3, df1], ignore_index=True)
df = df[df.task.isin(["AB", "BC", "CA"])]

# --- 2. Pivot ---
heat3 = df[df.model_size == "3B"].pivot(index="language", columns="task", values="accuracy")
heat1 = df[df.model_size == "1B"].pivot(index="language", columns="task", values="accuracy")

# --- 3. Order languages ---
lang_order = ["en", "es", "hi", "ar", "zh-cn"]
heat3 = heat3.loc[lang_order]
heat1 = heat1.loc[lang_order]

# --- 4. Plot setup ---
sns.set_theme(style="white", font_scale=1.2)
fig, (ax0, ax1) = plt.subplots(
    1, 2,
    figsize=(6, 3),
    gridspec_kw={"wspace": 0.02}
)

# --- 5. Heatmaps (auto aspect to fill width) ---
for ax, heat, title in [
    (ax0, heat1, "Llama-3.2-1B"),
    (ax1, heat3, "Llama-3.2-3B")
]:
    sns.heatmap(
        heat,
        ax=ax,
        cmap="YlGnBu",
        vmin=0.95,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 11},
        cbar=False,
        linewidths=0
    )
    ax.set_title(title, fontsize=14, pad=6)
    ax.set_xlabel("", fontsize=0)
    ax.tick_params(labelsize=12)
    ax.set_aspect('auto')  # let the axes fill the allotted width
# Add y-axis label to left plot
ax0.set_ylabel("Language", fontsize=12)
# Remove y-axis label and ticks on right plot
ax1.set_ylabel("")
ax1.set_yticklabels([])

# --- 6. Shared colorbar ---
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="2%", pad=0.01)
cb = fig.colorbar(ax1.collections[0], cax=cax)
cb.ax.tick_params(labelsize=11)
cb.set_label("Accuracy", fontsize=12, labelpad=4)

# --- 7. Trim margins ---
fig.subplots_adjust(
    left=0.07,
    right=0.95,
    top=0.90,
    bottom=0.12
)

# --- 8. Save figure ---
output_dir = os.path.join("plots", "ft")
os.makedirs(output_dir, exist_ok=True)
fig.savefig(
    os.path.join(output_dir, "accuracy_heatmaps.pdf"),
    bbox_inches="tight",
    pad_inches=0
)
fig.savefig(
    os.path.join(output_dir, "accuracy_heatmaps.png"),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0
)
plt.close(fig)
