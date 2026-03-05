import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv("data.csv")

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

# 2. Setup the Grid - REDUCED ASPECT TO SQUEEZE HORIZONTALLY
# Aspect is the width/height ratio. Reducing from 3.5 to 1.8 - 2.0 squeezes the X-axis.
g = sns.FacetGrid(
    plot_df,
    row="step_setting",
    height=4,  # Slightly shorter height
    aspect=1.8,  # Much lower aspect ratio = tighter horizontal spacing
    margin_titles=True,
    sharex=True,
)


# 3. Plotting Function
def layered_swarm(data, **kwargs):
    ax = plt.gca()
    highlight_palette = {"Correlation": "#E63946", "Random": "#457B9D"}

    # Layer 1: The Field
    sns.swarmplot(
        # data=data[~data["method"].isin(["Correlation", "Random"])],
        data=data,
        x="x_axis_group",
        y="result",
        hue="method",
        order=group_order,
        palette="Set2",
        size=4.5,  # Slightly smaller dots to prevent overlap in tight space
        dodge=False,
        ax=ax,
    )

    # # Layer 2: Highlights
    # sns.swarmplot(
    #     data=data[data["method"].isin(["Correlation", "Random"])],
    #     x="x_axis_group",
    #     y="result",
    #     hue="method",
    #     order=group_order,
    #     palette=highlight_palette,
    #     marker="D",
    #     size=5.5,
    #     linewidth=0.8,
    #     edgecolor="black",
    #     dodge=False,
    #     ax=ax,
    # )

    if ax.get_legend():
        ax.get_legend().remove()


# 4. Map the function
g.map_dataframe(layered_swarm)

# 5. Final Polish
g.set(ylim=(0, 1.05))
g.set_axis_labels("", "Score (0.0 - 1.0)")

for ax in g.axes.flat:
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(
        group_order, rotation=90, ha="center", fontsize=7
    )  # Smaller font for tight space

    # Visual separation between Dataset blocks
    for i in range(len(group_order)):
        if (i + 1) % len(metrics_to_plot) == 0:
            ax.axvline(i + 0.5, color="black", linestyle="-", alpha=0.1, linewidth=1)

# Legend adjustment
other_methods = [
    # m for m in plot_df["method"].unique() if m not in ["Correlation", "Random"]
    m
    for m in plot_df["method"].unique()
]
colors = sns.color_palette("Set2", len(other_methods))
# legend_elements = [
#     Line2D(
#         [0],
#         [0],
#         marker="D",
#         color="w",
#         label="Correlation",
#         markerfacecolor="#E63946",
#         markeredgecolor="black",
#     ),
#     Line2D(
#         [0],
#         [0],
#         marker="D",
#         color="w",
#         label="Random",
#         markerfacecolor="#457B9D",
#         markeredgecolor="black",
#     ),
# ]
legend_elements = []
for i, m in enumerate(other_methods):
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=m,
            markerfacecolor=colors[i],
            markersize=5,
        )
    )

# Move legend slightly closer
g.fig.legend(
    handles=legend_elements,
    loc="center right",
    bbox_to_anchor=(1.05, 0.5),
    title="Methods",
    fontsize="small",
)

# Final layout compression
plt.subplots_adjust(top=0.9, right=0.85, bottom=0.35, hspace=0.4)
# Optional: force a specific width for the whole figure
# g.fig.set_size_inches(12, 8)

plt.savefig("dot_plot.svg", format="svg", bbox_inches="tight")
plt.show()
