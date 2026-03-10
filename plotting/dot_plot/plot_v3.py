import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load data
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


# 2. Setup the Unified Palette and Explicit Order
unique_methods = plot_df["method"].unique()
special_methods = ["Correlation", "Random"]
other_methods = [m for m in unique_methods if m not in special_methods]

# Force Correlation and Random to the front of the list
ordered_methods = special_methods + other_methods

# Get enough colors from Set2 for the non-highlighted methods
set2_colors = sns.color_palette("Set2", len(other_methods))

# Build the combined palette dictionary using the ordered list
custom_palette = {}
color_index = 0

for method in ordered_methods:
    if method == "Correlation":
        custom_palette[method] = "#E63946"
    elif method == "Random":
        custom_palette[method] = "#457B9D"
    else:
        custom_palette[method] = set2_colors[color_index]
        color_index += 1

all_methods = (
    ordered_methods  # This is now in the desired order for plotting and legend
)

# 3. Setup the Grid
g = sns.FacetGrid(
    plot_df,
    row="step_setting",
    height=4,
    aspect=1.8,
    margin_titles=True,
    sharex=True,
)


# 4. Define Plotting Function
def layered_swarm(data, **kwargs):
    ax = plt.gca()

    # Now we just plot everything in one go using the unified palette
    sns.swarmplot(
        data=data,
        x="x_axis_group",
        y="result",
        hue="method",
        order=group_order,
        palette=custom_palette,
        size=5,
        dodge=False,
        ax=ax,
        alpha=1.0,
    )

    if ax.get_legend():
        ax.get_legend().remove()


# 5. Map the function
g.map_dataframe(layered_swarm)

# 6. Final Polish
g.set(ylim=(0, 1.05))
g.set_axis_labels("", "Score (0.0 - 1.0)")

for ax in g.axes.flat:
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order, rotation=90, ha="center", fontsize=7)

    for i in range(len(group_order)):
        if (i + 1) % len(metrics_to_plot) == 0:
            ax.axvline(i + 0.5, color="black", linestyle="-", alpha=0.1, linewidth=1)

# 7. Custom Legend
# 7. Custom Legend
legend_elements = []

# Iterate through all methods and use the custom_palette dictionary for the colors
for m in all_methods:
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=m,
            markerfacecolor=custom_palette[m],
            markersize=6,
        )
    )

# Calculate the number of columns needed to split the items evenly into 2 rows
num_columns = (len(all_methods) + 1) // 2

g.fig.legend(
    handles=legend_elements,
    loc="upper center",  # Anchor point on the legend box
    bbox_to_anchor=(0.5, 0.0),  # Position relative to the figure (x=center, y=bottom)
    title="Methods",
    fontsize="small",
    ncol=num_columns,  # Automatically splits into 2 rows
)

plt.subplots_adjust(top=0.9, right=0.88, bottom=0.35, hspace=0.4)
plt.savefig("dot_plot_custom.svg", format="svg", bbox_inches="tight")
