#!/usr/bin/env python3
"""
AAAI-ready figures comparing Task-Vector vs Full Fine-Tuning (1B + 3B)

- Dumbbell (top-2 per task): two-row figure -> Row1=1B, Row2=3B; languages are columns.
  Clipped y-axis, larger fonts, wider spacing; legend once.
- Heatmap (top-2 per task): two-row figure -> Row1=1B, Row2=3B; languages are columns.
  Inside each language panel: 2 x (#tasks) (Top row = higher, Second row = lower);
  no "Top/Second" labels; cell text on two lines "model\nvalue"; BLUE = BETTER (YlGnBu).
- Saves both PNG and PDF.

Outputs:
  results_tv/tv_vs_ft_dumbbell_1B_3B.(png|pdf)
  results_tv/tv_vs_ft_pct_heatmap_top2_1B_3B.(png|pdf)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# -------------------- Paths --------------------
TV_PATH = "results_tv/combined_task_vector_results_datav2.csv"
FT_1B_PATH = "finetuning_results_v2_1B_5epochs.csv"
FT_3B_PATH = "finetuning_results_v2_3B_5epochs.csv"
OUT_DIR = "results_tv"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- Load & Merge --------------------
print("Loading combined task vector results...")
vec_df = pd.read_csv(TV_PATH)
vec_df = vec_df.dropna(subset=["test_accuracy"]).copy()
vec_df["test_accuracy"] = vec_df["test_accuracy"].astype(float)
vec_df["vector_accuracy"] = (vec_df["test_accuracy"] / 400.0) * 100.0
vec_df = vec_df.rename(columns={"task": "tv_task"})

def derive_ft_task(tv_task: str) -> str:
    # "AB_BA" -> "AB", "BC_CB" -> "BC", "CA_AC" -> "CA"
    if not isinstance(tv_task, str):
        return np.nan
    if "_" in tv_task:
        return tv_task.split("_", 1)[0]
    return tv_task[:2] if len(tv_task) >= 2 else np.nan

vec_df["ft_task"] = vec_df["tv_task"].apply(derive_ft_task)
vec_df = vec_df.dropna(subset=["ft_task"])

# Fine-tune results
ft_dfs = []
for sz, path in [("1B", FT_1B_PATH), ("3B", FT_3B_PATH)]:
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        continue
    ft = pd.read_csv(path).rename(
        columns={"model_size": "size", "language": "lang",
                 "task": "task", "accuracy": "ft_accuracy"}
    )
    ft["size"] = sz
    if ft["ft_accuracy"].max() <= 1.01:
        ft["ft_accuracy"] = ft["ft_accuracy"] * 100.0
    ft_dfs.append(ft)

if not ft_dfs:
    raise SystemExit("No fine-tuning files found. Aborting.")

ft_df = pd.concat(ft_dfs, ignore_index=True)

merged = pd.merge(
    vec_df, ft_df,
    left_on=["size", "lang", "ft_task"],
    right_on=["size", "lang", "task"],
    how="left", suffixes=("", "_ft"),
).dropna(subset=["ft_accuracy"]).copy()

result = merged[["size", "lang", "instruction_models", "ft_task",
                 "vector_accuracy", "ft_accuracy"]].copy()
result["vector_pct_of_ft"] = (result["vector_accuracy"] / result["ft_accuracy"]) * 100.0

# -------------------- Helpers --------------------
TASK_ORDER = ["AB", "BC", "CA"]
LANG_PREF  = ["en", "hi", "ar", "es", "zh-cn", "zh", "fr", "de"]

def order_by(seq, preferred):
    seen = set(); out = []
    for p in preferred:
        if p in seq and p not in seen:
            out.append(p); seen.add(p)
    for s in sorted(seq):
        if s not in seen:
            out.append(s)
    return out

def ordered_union_langs(df, sizes=("1B","3B")):
    langs = []
    for sz in sizes:
        langs += df[df["size"]==sz]["lang"].unique().tolist()
    return order_by(list(dict.fromkeys(langs)), LANG_PREF)

def pick_top2_models_per_task(df, size="1B"):
    """Rank models by mean % of FT (across languages) for this size."""
    d = df[df["size"] == size]
    top2 = {}
    for t in sorted(d["ft_task"].unique()):
        agg = (d[d["ft_task"] == t]
               .groupby("instruction_models")["vector_pct_of_ft"]
               .mean()
               .sort_values(ascending=False))
        top2[t] = agg.index.tolist()[:2]
    return top2

def pick_top2_per_task_for_lang_and_size(df, lang, size):
    """For a (lang,size), return dict: task -> [(name_top,val_top),(name_second,val_second)]."""
    out = {}
    d = df[(df["lang"]==lang) & (df["size"]==size)]
    for t in sorted(d["ft_task"].unique()):
        sub = d[d["ft_task"] == t]
        ranked = sub.sort_values("vector_pct_of_ft", ascending=False).head(2)
        pairs = [(row["instruction_models"], float(row["vector_pct_of_ft"])) for _, row in ranked.iterrows()]
        if len(pairs) == 1:
            pairs.append(("", np.nan))
        out[t] = pairs
    return out

# -------------------- Dumbbell (combined 1B+3B) --------------------
def plot_dumbbell_combined(df, sizes=("1B","3B"),
                           y_min=55, y_max=102, font_base=12,
                           outfile_base="tv_vs_ft_dumbbell_1B_3B"):
    d = df[df["size"].isin(sizes)].copy()
    if d.empty:
        print("No rows for requested sizes; skipping dumbbell.")
        return

    langs = ordered_union_langs(d, sizes)
    tasks_available = sorted(d["ft_task"].unique().tolist())
    tasks = [t for t in TASK_ORDER if t in tasks_available]
    if not tasks:
        print("No tasks (AB/BC/CA) found for dumbbell; skipping.")
        return

    # Precompute positions per size (top-2 per task for that size)
    def compute_positions_for_size(size):
        dd = d[d["size"]==size]
        top2 = pick_top2_models_per_task(dd, size=size)
        block_gap, slot_gap = 1.8, 1.6
        positions, x_ticks, x_ticklabels, task_centers = {}, [], [], {}
        x_cursor = 0.0
        for t in tasks:
            kept = [m for m in top2.get(t, [])
                    if not dd[(dd["ft_task"] == t) & (dd["instruction_models"] == m)].empty]
            if not kept:
                continue
            start = x_cursor
            for im in kept:
                positions[(t, im)] = x_cursor
                x_ticks.append(x_cursor)
                x_ticklabels.append(im)
                x_cursor += slot_gap
            end = x_cursor - slot_gap
            task_centers[t] = 0.5 * (start + end)
            x_cursor += block_gap
        return positions, x_ticks, x_ticklabels, task_centers

    pos_by_size = {sz: compute_positions_for_size(sz) for sz in sizes}

    # Figure: rows = sizes (1B top, 3B bottom), cols = languages
    nrows, ncols = len(sizes), len(langs)
    fig_w = max(8.0, 3.4 * ncols)
    fig_h = 4.3 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             sharey=True, constrained_layout=True)
    if nrows == 1:
        axes = [axes]
    axes = np.array(axes).reshape(nrows, ncols)

    lw, ms = 1.1, 36

    for r, sz in enumerate(sizes):
        positions, x_ticks, x_ticklabels, task_centers = pos_by_size[sz]
        if not positions:
            for c in range(ncols):
                axes[r, c].axis("off")
            continue

        for c, lang in enumerate(langs):
            ax = axes[r, c]
            sub = d[(d["size"]==sz) & (d["lang"]==lang)]
            if sub.empty:
                ax.axis("off")
                continue

            first = True
            for (t, im), x in positions.items():
                row = sub[(sub["ft_task"] == t) & (sub["instruction_models"] == im)]
                if row.empty:
                    continue
                va = float(row["vector_accuracy"].iloc[0])
                fa = float(row["ft_accuracy"].iloc[0])
                lo, hi = sorted([va, fa])
                ax.vlines(x, lo, hi, lw=lw, alpha=0.85)
                ax.scatter([x], [va], s=ms, marker="o",
                           label="Task-Vector" if first else None, zorder=3)
                ax.scatter([x], [fa], s=ms, marker="s",
                           label="Fine-Tune" if first else None, zorder=3)
                first = False

            title = f"{lang}"
            ax.set_title(title, fontsize=font_base+1, pad=8)
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(min(x_ticks) - 0.7, max(x_ticks) + 0.7)
            ax.grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.6)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels, rotation=60, ha="right", fontsize=font_base-1)
            ax.tick_params(axis='x', pad=10)
            ax.margins(x=0.02)

            trans = blended_transform_factory(ax.transData, ax.transAxes)
            for t, xc in task_centers.items():
                ax.text(xc, -0.12, t, ha="center", va="top",
                        fontsize=font_base, transform=trans)

            centers = list(task_centers.items()); centers.sort(key=lambda kv: kv[1])
            for i in range(len(centers) - 1):
                sep_x = 0.5 * (centers[i][1] + centers[i+1][1])
                ax.axvline(sep_x, color="#BBBBBB", lw=0.7, alpha=0.5)

            if c == 0:
                ax.set_ylabel(f"Accuracy (%) — {sz}", fontsize=font_base)

    handles, labels = axes[0,0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=font_base)

    fig.suptitle("Task Vector vs Fine-Tuning (Dumbbell, top-2 per task)\nRow 1: 1B  •  Row 2: 3B",
                 y=1.08, fontsize=font_base+3, fontweight="bold")

    png = os.path.join(OUT_DIR, f"{outfile_base}.png")
    pdf = os.path.join(OUT_DIR, f"{outfile_base}.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {png}\nSaved: {pdf}")
    plt.close(fig)

# -------------------- Heatmap (combined 1B+3B) --------------------
def plot_heatmap_top2_combined(df, sizes=("1B","3B"),
                               vmin=50, vmax=105, font_base=12,
                               outfile_base="tv_vs_ft_pct_heatmap_top2_1B_3B"):
    d = df[df["size"].isin(sizes)].copy()
    if d.empty:
        print("No rows for requested sizes; skipping heatmap.")
        return

    tasks_available = sorted(d["ft_task"].unique().tolist())
    tasks = [t for t in TASK_ORDER if t in tasks_available]
    if not tasks:
        print("No tasks (AB/BC/CA) found for heatmap; skipping.")
        return

    langs = ordered_union_langs(d, sizes)
    nrows, ncols = len(sizes), len(langs)
    fig_w = max(8.0, 3.2 * ncols)
    # EDIT HERE
    fig_h = 3.0 * nrows  # Increased from 3.8 to make cells taller
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    if nrows == 1:
        axes = [axes]
    axes = np.array(axes).reshape(nrows, ncols)

    cmap = plt.cm.YlGnBu  # BLUE = better
    im = None
    mid = (vmin + vmax) / 2.0

    for r, sz in enumerate(sizes):
        for c, lang in enumerate(langs):
            ax = axes[r, c]
            top2 = pick_top2_per_task_for_lang_and_size(d, lang, sz)
            if not top2:
                ax.axis("off")
                continue

            vals_mat, names_mat = [], []
            for t in tasks:
                if t in top2:
                    (nm1, v1), (nm2, v2) = top2[t]
                else:
                    nm1, v1, nm2, v2 = "", np.nan, "", np.nan
                vals_mat.append([v1, v2])    # [top, second]
                names_mat.append([nm1, nm2]) # [top, second]

            mat = np.array(vals_mat, dtype=float).T   # shape (2, n_tasks)
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, aspect="auto", cmap=cmap)

            ax.set_title(f"{lang}", fontsize=font_base+1, pad=6)  # Increased title font
            ax.set_xticks(range(len(tasks)))
            ax.set_xticklabels(tasks, fontsize=font_base)  # Increased x-axis labels
            ax.set_yticks([])  # no "Top/Second" labels

            names_arr = np.array(names_mat, dtype=object).T
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    nm  = names_arr[i, j] if isinstance(names_arr[i, j], str) else ""
                    if np.isnan(val):
                        txt, color = "", "black"
                    else:
                        txt = f"{nm}\n{val:.0f}"
                        color = "white" if val >= mid else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            # EDIT HERE 
                            fontsize=font_base, color=color)  # Increased from font_base-1

            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if c == 0:
                ax.set_ylabel(sz, fontsize=font_base+2, rotation=90)  # Increased y-axis labels

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, location="right", pad=0.02)
        # no label as requested

    fig.suptitle("Task-Vector Accuracy as % of Finetune Accuracy:   Row 1: 1B  •  Row 2: 3B",
                 y=1.03, fontsize=font_base+2, fontweight="bold")

    png = os.path.join(OUT_DIR, f"{outfile_base}.png")
    pdf = os.path.join(OUT_DIR, f"{outfile_base}.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {png}\nSaved: {pdf}")
    plt.close(fig)

# -------------------- Run --------------------
print("\nMerged summary (head):")
print(result.head().to_string(index=False, float_format=lambda x: f"{x:.1f}"))

plot_dumbbell_combined(result, sizes=("1B","3B"))
# plot_heatmap_top2_combined(result, sizes=("1B","3B"))

print("Done.")
