import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load your dataset
# Replace 'data.csv' with your actual file path
# Assuming columns: method, Dataset, setting, metric, TimeType, Value, Error
df = pd.read_csv("heatmap.csv")

# 2. Pivot to get Real Time and Pseudotime side-by-side to calculate LFC
# We pivot based on dataset, setting, metric, and method
pivot_df = df.pivot_table(
    index=["dataset", "step_setting", "metric", "method", "prc_threshold"],
    columns="time_type",
    values="result",
).reset_index()

# 3. Calculate Log2 Fold Change
# LFC = log2(Pseudotime / Real Time)
# This handles the ratio: positive = Pseudotime is higher, negative = Real Time is higher
pivot_df["LFC"] = np.log2(pivot_df["Pseudotime"] / pivot_df["Real Time"])

# 4. Iterate through each dataset and setting to create separate heatmaps
datasets = pivot_df["dataset"].unique()
path_types = pivot_df["step_setting"].unique()
prc_thresholds = pivot_df["prc_threshold"].unique()

for ds in datasets:
    for pt in path_types:
        print(f"Processing Dataset: {ds}, Step Setting: {pt}")

        # let's choose prc_threshold to true if step_setting is 'all_paths'
        # and false otherwise
        prc_threshold = True
        # Filter for the specific dataset and setting combination
        subset = pivot_df[
            (pivot_df["dataset"] == ds)
            & (pivot_df["step_setting"] == pt)
            & (pivot_df["prc_threshold"] == prc_threshold)
            & (pivot_df["metric"].isin(["AUC_PRC", "AUC_ROC", "JaccardSimilarity"]))
        ]

        # If no data exists for this specific combination, skip it
        if subset.empty:
            continue

        # Pivot for the heatmap: metrics as rows, methods as columns
        heatmap_data = subset.pivot(index="metric", columns="method", values="LFC")

        # 5. Plot the heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            heatmap_data,
            annot=True,  # Show the LFC value in the cell
            fmt=".3f",  # Format to 3 decimal places
            cmap="RdBu_r",  # Red (Positive) to Blue (Negative) diverging map
            center=0,  # Ensure 0 (no change) is the neutral color
            cbar_kws={"label": "Log2 Fold Change"},
        )

        plt.title(f"Log Fold Change (Pseudotime / Real Time): {ds}")
        plt.ylabel(f"{pt} Metric")
        plt.xlabel("Method")

        # Save and/or show
        filename = f"heatmap_{ds}_{pt.replace(' ', '_')}.svg"
        plt.savefig(filename, format="svg", bbox_inches="tight")
        # plt.show()
        plt.close()
