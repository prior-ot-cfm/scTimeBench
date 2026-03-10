import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
import numpy as np

df = pd.read_csv("data_v2.csv")

# 1. Filter Data
metrics_to_plot = ["JaccardSimilarity", "AUC_PRC", "AUC_ROC"]
plot_df = df[
    (df["prc_threshold"] == True)
    & (df["time_type"] == "Real Time")
    & (df["metric"].isin(metrics_to_plot))
].copy()

# Sort and prepare combined labels
plot_df["metric"] = pd.Categorical(
    plot_df["metric"], categories=metrics_to_plot, ordered=True
)
plot_df = plot_df.sort_values(["metric", "dataset"])

plot_df["x_axis_group"] = (
    plot_df["dataset"].astype(str) + " | " + plot_df["metric"].astype(str)
)
group_order = list(dict.fromkeys(plot_df["x_axis_group"]))

# 2. Setup global method order and custom palette
all_methods = list(plot_df["method"].unique())
custom_palette = {}
set2_colors = sns.color_palette("Set2", len(all_methods))
color_idx = 0

for m in all_methods:
    if m == "Correlation":
        custom_palette[m] = "#E63946"  # Red
    elif m == "Random":
        custom_palette[m] = "#457B9D"  # Blue
    else:
        custom_palette[m] = set2_colors[color_idx]
        color_idx += 1

# ==========================================
# THE FIX: INVISIBLE DUMMY POINTS
# ==========================================
# We add one point for every method at y = -1000.
# This guarantees Seaborn never encounters an "empty group" and avoids the crash.
dummy_rows = []
template_row = plot_df.iloc[
    0
].copy()  # Use a real row as a template to keep all columns intact

for step in plot_df["step_setting"].unique():
    for m in all_methods:
        new_row = template_row.copy()
        new_row["step_setting"] = step
        new_row["method"] = m
        new_row["result"] = -1000  # Will be hidden below the plot boundary
        dummy_rows.append(new_row)

plot_df_padded = pd.concat([plot_df, pd.DataFrame(dummy_rows)], ignore_index=True)


# 3. Setup the Grid (Using our padded dataframe)
g = sns.FacetGrid(
    plot_df_padded,
    row="step_setting",
    height=4,
    aspect=1.8,
    margin_titles=True,
    sharex=True,
)


def single_swarm_modified(data, **kwargs):
    ax = plt.gca()

    # Because of the dummy points, we can safely use hue and palette without crashing!
    sns.swarmplot(
        data=data,
        x="x_axis_group",
        y="result",
        # hue="method",
        # hue_order=all_methods,
        order=group_order,
        # palette=custom_palette,
        size=4.5,
        dodge=False,
        ax=ax,
    )

    corr_rgb = mcolors.to_rgb("#E63946")
    rand_rgb = mcolors.to_rgb("#457B9D")

    path_circle = (
        mmarkers.MarkerStyle("o")
        .get_path()
        .transformed(mmarkers.MarkerStyle("o").get_transform())
    )
    path_diamond = (
        mmarkers.MarkerStyle("D")
        .get_path()
        .transformed(mmarkers.MarkerStyle("D").get_transform())
    )
    path_x = (
        mmarkers.MarkerStyle("X")
        .get_path()
        .transformed(mmarkers.MarkerStyle("X").get_transform())
    )

    # Safely iterate and swap shapes based on exact assigned colors
    for collection in ax.collections:
        if not hasattr(collection, "get_facecolors"):
            continue

        facecolors = collection.get_facecolors()
        if len(facecolors) == 0:
            continue

        new_paths, sizes, edgecolors, linewidths = [], [], [], []
        orig_sizes = collection.get_sizes()
        default_size = (
            orig_sizes[0] if len(orig_sizes) > 0 else 20.25
        )  # Fallback to default area

        for fc in facecolors:
            rgb = fc[:3]
            if np.allclose(rgb, corr_rgb, atol=0.05):
                new_paths.append(path_diamond)
                sizes.append(40)
                edgecolors.append((0, 0, 0, 1))
                linewidths.append(0.8)
            elif np.allclose(rgb, rand_rgb, atol=0.05):
                new_paths.append(path_x)
                sizes.append(45)
                edgecolors.append((0, 0, 0, 1))
                linewidths.append(0.8)
            else:
                new_paths.append(path_circle)
                sizes.append(default_size)
                edgecolors.append(fc)
                linewidths.append(0)

        collection.set_paths(new_paths)
        collection.set_sizes(sizes)
        collection.set_edgecolors(edgecolors)
        collection.set_linewidths(linewidths)

    if ax.get_legend():
        ax.get_legend().remove()


# Map the function
g.map_dataframe(single_swarm_modified)

# 5. Final Polish
# *** The (0, 1.05) limit perfectly hides our -1000 dummy points ***
g.set(ylim=(0, 1.05))
g.set_axis_labels("", "Score (0.0 - 1.0)")

for ax in g.axes.flat:
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order, rotation=90, ha="center", fontsize=7)

    for i in range(len(group_order)):
        if (i + 1) % len(metrics_to_plot) == 0:
            ax.axvline(i + 0.5, color="black", linestyle="-", alpha=0.1, linewidth=1)

# 6. Rebuild Legend to match new shapes
legend_elements = []

legend_elements.append(
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        label="Correlation",
        markerfacecolor="#E63946",
        markeredgecolor="black",
        markersize=7,
    )
)
legend_elements.append(
    Line2D(
        [0],
        [0],
        marker="X",
        color="w",
        label="Random",
        markerfacecolor="#457B9D",
        markeredgecolor="black",
        markersize=7,
    )
)

other_methods = [m for m in all_methods if m not in ["Correlation", "Random"]]
for m in other_methods:
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=m,
            markerfacecolor=custom_palette[m],
            markersize=5,
        )
    )

g.fig.legend(
    handles=legend_elements,
    loc="center right",
    bbox_to_anchor=(1.05, 0.5),
    title="Methods",
    fontsize="small",
)

# Final layout compression
plt.subplots_adjust(top=0.9, right=0.85, bottom=0.35, hspace=0.4)

plt.savefig("dot_plot_v2.svg", format="svg", bbox_inches="tight")
# plt.show()
