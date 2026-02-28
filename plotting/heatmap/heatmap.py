import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_FILE = "heatmap.csv"
OUTPUT_FILE = "split_filtered_heatmap.svg"
METRICS_OF_INTEREST = ["AUC_PRC", "AUC_ROC", "JaccardSimilarity"]

# 1. Load and Rename
df = pd.read_csv(INPUT_FILE)


def rename_3x_groups(row):
    if "3x" in str(row["dataset"]):
        return f"{row['method']}-3x"
    return row["method"]


df["method"] = df.apply(rename_3x_groups, axis=1)

# 2. Pivot & Calculate LFC
pivot_df = df.pivot_table(
    index=["dataset", "step_setting", "metric", "method", "prc_threshold"],
    columns="time_type",
    values="result",
).reset_index()

pivot_df["LFC"] = np.log2(pivot_df["Pseudotime"] / pivot_df["Real Time"])

# 3. Filter Logic
pivot_df = pivot_df[
    (pivot_df["metric"].isin(METRICS_OF_INTEREST)) & (pivot_df["prc_threshold"] == True)
]

# 4. Get unique step_settings to determine number of subplots
step_settings = pivot_df["step_setting"].unique()
fig, axes = plt.subplots(len(step_settings), 1, figsize=(14, 8 * len(step_settings)))

# Ensure axes is an array even if there is only one setting
if len(step_settings) == 1:
    axes = [axes]

# Global max for a consistent colorbar across all subplots
clean_vals = pivot_df.replace([np.inf, -np.inf], np.nan)
max_abs = clean_vals["LFC"].abs().max()
if pd.isna(max_abs) or max_abs == 0:
    max_abs = 1

# 5. Plotting Loop
for i, setting in enumerate(step_settings):
    ax = axes[i]
    # Filter data for this specific setting
    setting_df = pivot_df[pivot_df["step_setting"] == setting]

    # Final Pivot for this subplot
    subplot_pivot = setting_df.pivot_table(
        index=["dataset", "metric"], columns="method", values="LFC"
    )

    # Set grey background for NaNs
    ax.set_facecolor("#E0E0E0")

    # Draw Heatmap
    sns.heatmap(
        subplot_pivot,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Log2 Fold Change"} if i == 0 else None,  # Only one legend
        ax=ax,
    )

    # 6. MANUALLY DRAW SLASHES for this subplot
    rows, cols = subplot_pivot.shape
    for y in range(rows):
        for x in range(cols):
            val = subplot_pivot.iloc[y, x]
            if pd.isna(val) or np.isinf(val):
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    "/",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=14,
                    fontweight="bold",
                )

    ax.set_title(f"Step Setting: {setting}", fontsize=16, fontweight="bold")
    ax.set_ylabel("Dataset / Metric")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, format="svg")
# plt.show()
