import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Read & combine data ---
df1 = pd.read_csv("dpo_training_results_1B_datav2.csv")
df3 = pd.read_csv("dpo_training_results_3B_datav2.csv")
df = pd.concat([df1, df3], ignore_index=True)

# Filter to key tasks
tasks = ["AB", "BC", "CA"]
df = df[df.task.isin(tasks)]

# --- 2. Order categories ---
lang_order = ["en", "es", "hi", "ar", "zh-cn"]
df["language"] = pd.Categorical(df["language"], categories=lang_order, ordered=True)
df["task"] = pd.Categorical(df["task"], categories=tasks, ordered=True)
# --- 2.5 Customize facet titles for model sizes ---
# Map original model_size values to custom labels for independent facet naming
title_map = {"1B": "Llama-3.2-1B", "3B": "Llama-3.2-3B"}
df["model_size"] = df["model_size"].map(title_map)

# --- 3. Compute statistics ---
stats = df.groupby(["model_size", "language", "task"], observed=False)['accuracy'].agg(['mean', 'std']).reset_index()

# --- 4. Prepare output directory ---
output_dir = os.path.join("plots", "dpo")
os.makedirs(output_dir, exist_ok=True)

# Set seaborn theme
# sns.set_theme(style="whitegrid", font_scale=1.2)
sns.set_theme(
    style="whitegrid",
    rc={
    "axes.labelsize": 15,   # axis titles
    "xtick.labelsize": 14,  # x‐axis tick labels
    "ytick.labelsize": 12,  # y‐axis tick labels
    "legend.fontsize": 14,  # legend text
    "legend.title_fontsize": 14,  # legend “Language” if you ever set a title
    "axes.titlesize": 14    # facet titles
    }
)

# --- 5. Bar plot with error bars (mean ± std) ---
g = sns.catplot(
    data=df, x="task", y="accuracy", hue="language", col="model_size",
    kind="bar", height=4, aspect=1, palette="tab10",
    errorbar="sd", capsize=0.1
)
g.set_axis_labels("", "Accuracy")
g.set_titles("{col_name}")
# Remove the facet legend
g._legend.remove()
# Adjust bottom margin to fit legend
g.fig.subplots_adjust(bottom=0.2)
# Add a compact horizontal legend at the bottom without title
ncol = df["language"].nunique()
legend = g.add_legend(ncol=ncol, loc="lower left", bbox_to_anchor=(0.4, -0.03),
                      fontsize='medium', handlelength=1, handletextpad=0.3, columnspacing=0.5)
# Place the 'Language' label next to the legend entries; adjust coords as needed
g.fig.text(0.35, 0.032, 'Language', ha='center', va='center', fontsize='medium')

for ax in g.axes.flatten():
    ax.set_ylim(0.95, 1.0)
    g.savefig(os.path.join(output_dir, "barplot_accuracy_sd.pdf"), bbox_inches="tight")
    g.savefig(os.path.join(output_dir, "barplot_accuracy_sd.png"), dpi=300, bbox_inches="tight")
plt.close()