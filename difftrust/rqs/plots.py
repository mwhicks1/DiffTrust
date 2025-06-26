import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches, pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def truncate_colormap(cmap, minval=0.5, maxval=1.0, n=256):
    """
    Returns a truncated version of a colormap.
    """
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def save_log2_scatter_plot(
        results_dir,
        x_vals,
        y_vals,
        alternate_label,
        file_name,
        point_color="#278091",
        point_size=32,
        point_alpha=0.8,
        figsize=(8, 8),
        font_scale=1,
        dpi=600,
        include_points=True,
        show_diag_line=True,
        xlim=(2 ** -10, 1),
        ylim=(2 ** -10, 1),
        x_label="Incoherence",
        y_label="Error",
        title="Error vs. Incoherence Scatter Plot (log₂ scale)",
):
    sns.set_theme(style="whitegrid", font_scale=font_scale)

    # Prepare plot
    fig, ax = plt.subplots(figsize=figsize)

    # Apply log scale
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)

    # Clip values to avoid log(0)
    x_vals = np.clip(x_vals, xlim[0], xlim[1])
    y_vals = np.clip(y_vals, ylim[0], ylim[1])

    if include_points:
        ax.scatter(x_vals, y_vals, color=point_color, s=point_size, alpha=point_alpha)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(f"{x_label}", labelpad=0, fontsize=30)
    ax.set_ylabel(f"{y_label}", labelpad=0, fontsize=30)
    ax.set_title(title, fontsize=16, pad=10)

    if show_diag_line:
        diag_x = np.logspace(np.log2(xlim[0]), np.log2(xlim[1]), base=2, num=500)
        diag_y = diag_x * 0.5

        diag_line, = ax.plot(
            diag_x, diag_y,
            linestyle="dashed",
            color="black",
            linewidth=4,
            alpha=0.9,
            label="_nolegend_"  # <-- this suppresses legend entry
        )

    # # 🔽 Add invisible dummy line just to carry alternate text
    # alt_line = ax.plot([], [], color='none', label=alternate_label)[0]
    #
    # ax.legend(
    #     handles=[alt_line],
    #     loc="lower right",
    #     fontsize=24,
    #     frameon=True,
    #     labelspacing=.8
    # )

    # Add annotation box with alternate text (no legend involved)
    plt.gca().text(
        0.98, 0.02,  # bottom-right corner in axes coordinates
        alternate_label,
        transform=plt.gca().transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=28,
        color="black",
        bbox=dict(
            facecolor="white",
            edgecolor="gray",
            boxstyle="round,pad=0.3"
        )
    )

    sns.despine(ax=ax)
    ax.grid(True, which="both", linestyle="--", linewidth=1.5, alpha=0.6)

    # Save and show
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, file_name)
    plt.savefig(f"{base_path}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{base_path}.pdf", dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()


def save_grid_plot(
        results_dir,
        x_vals,
        y_vals,
        name,
        cmap="mako_r",
        grid_bins=10,
        rect_alpha=0.5,
        point_color="#278091",
        point_size=4,
        point_alpha=0.8,
        grid_line_color='#333333',
        grid_line_style="--",
        grid_line_alpha=0.8,
        grid_line_width=0.8,
        figsize=(8, 8),
        font_scale=1,
        dpi=600,
        include_points=True,
        include_pdf=False,
        show_diag_line=True,
        show_lm_fit=True,
        xlim=(0, 1),
        ylim=(0, 1),
        x_label="Incoherence",
        y_label="Error",
        title="Error vs. Incoherence Grid Plot for GPT-4o"
):
    """
    Publication-grade 2D grid plot with conditional bottom legend and proper seaborn colormap.
    """
    sns.set_theme(style="darkgrid", font_scale=font_scale, rc={
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "grid.linestyle": grid_line_style,
        "grid.linewidth": grid_line_width,
        "axes.grid": True
    })

    # Compute 2D histogram bin edges
    x_edges = np.linspace(xlim[0], xlim[1], grid_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], grid_bins + 1)

    # Compute 2D histogram counts
    counts, _, _ = np.histogram2d(x_vals, y_vals, bins=[x_edges, y_edges])

    # Normalize color range to observed counts
    max_count = counts.max()
    norm = plt.Normalize(vmin=0, vmax=max_count)

    # Determine number of discrete colors dynamically
    num_colors = max(10, min(256, int(max_count)))  # clamp between 10 and 256

    # Load and truncate the base colormap using the appropriate resolution
    base_cmap = sns.color_palette(cmap, as_cmap=True)

    # Truncate lower 50%
    cmap_fn = truncate_colormap(base_cmap, minval=0, maxval=.5, n=num_colors)

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(grid_bins):
        for j in range(grid_bins):
            x0, y0 = x_edges[i], y_edges[j]
            w = x_edges[i + 1] - x_edges[i]
            h = y_edges[j + 1] - y_edges[j]
            count = counts[i, j]
            color = cmap_fn(norm(count)) if count > 0 else (1, 1, 1, 0)
            rect = patches.Rectangle((x0, y0), w, h, facecolor=color, edgecolor="none",
                                     linewidth=0, alpha=rect_alpha)
            ax.add_patch(rect)

    if include_points:
        ax.scatter(x_vals, y_vals, color=point_color, s=point_size, alpha=point_alpha)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    buffer_x = 0.01 * (xlim[1] - xlim[0])
    buffer_y = 0.01 * (ylim[1] - ylim[0])

    ax.set_xlim(xlim[0] - buffer_x, xlim[1] + buffer_x)
    ax.set_ylim(ylim[0] - buffer_y, ylim[1] + buffer_y)

    ax.set_xticks(x_edges)
    ax.set_yticks(y_edges)
    ax.grid(True, linestyle=grid_line_style, color=grid_line_color, alpha=grid_line_alpha, linewidth=grid_line_width)
    sns.despine(ax=ax, top=True, right=True)

    ax.set_xlabel(f"{x_label}", labelpad=10, fontsize=16)
    ax.set_ylabel(f"{y_label}", labelpad=10, fontsize=16)
    ax.set_title(f"{title}", pad=15, fontsize=18)

    sm = plt.cm.ScalarMappable(cmap=cmap_fn, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Frequency", fontsize=12)

    legend_handles = []

    if show_diag_line:
        diag_x = np.linspace(*xlim, 500)
        diag_line, = ax.plot(
            diag_x, 0.5 * diag_x,
            linestyle="solid",
            color="#8cc63e",
            linewidth=2,
            alpha=0.9,
            label=r"$y = \frac{1}{2}x$"
        )
        legend_handles.append(diag_line)

    if show_lm_fit:
        df = pd.DataFrame({f"{x_label}": x_vals, f"{y_label}": y_vals})
        # Plot, then fetch the actual line for legend
        sns.regplot(data=df, x=x_label, y=y_label, scatter=False, ax=ax,
                    color="red", line_kws={"linewidth": 2})
        # Grab the last line on the axis
        lm_line = ax.lines[-1]
        lm_line.set_label("Linear fit")
        legend_handles.append(lm_line)

    # Draw legend below
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower right",
            ncol=len(legend_handles),
            frameon=True,
            bbox_to_anchor=(.8, 0.15),
            facecolor="white",

            fontsize=12
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        plt.tight_layout()

    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, name)
    plt.savefig(f"{base_path}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.show()
    if include_pdf:
        plt.savefig(f"{base_path}.pdf", bbox_inches="tight")
    plt.close()


def save_grid_plot_with_log(
        results_dir,
        x_vals,
        y_vals,
        label,
        name,
        cmap="mako_r",
        grid_bins=10,
        rect_alpha=0.5,
        point_color="#278091",
        point_size=4,
        point_alpha=0.8,
        grid_line_color='#333333',
        grid_line_style="--",
        grid_line_alpha=0.8,
        grid_line_width=0.8,
        figsize=(8, 8),
        font_scale=1,
        dpi=600,
        include_points=True,
        include_pdf=False,
        show_diag_line=True,
        show_lm_fit=True,
        xlim=None,
        ylim=None,
        x_label="Incoherence",
        y_label="Error",
        title="Error vs. Incoherence Grid Plot for GPT-4o",
        log_scaled=False,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib import patches
    import os
    import pandas as pd

    sns.set_theme(style="darkgrid", font_scale=font_scale, rc={
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "grid.linestyle": grid_line_style,
        "grid.linewidth": grid_line_width,
        "axes.grid": True
    })

    if log_scaled:
        x_vals = np.log2(np.clip(x_vals, 2 ** -10, None))  # or use 1e-8 still if more appropriate
        y_vals = np.log2(np.clip(y_vals, 2 ** -10, None))
        if xlim is None:
            xlim = (-10, 0)
        if ylim is None:
            ylim = (-10, 0)
    else:
        if xlim is None:
            xlim = (0, 1)
        if ylim is None:
            ylim = (0, 1)

    # Compute histogram edges
    x_edges = np.linspace(xlim[0], xlim[1], grid_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], grid_bins + 1)

    # Compute 2D histogram counts
    counts, _, _ = np.histogram2d(x_vals, y_vals, bins=[x_edges, y_edges])
    max_count = counts.max()
    norm = plt.Normalize(vmin=0, vmax=max_count)
    num_colors = max(10, min(256, int(max_count)))

    base_cmap = sns.color_palette(cmap, as_cmap=True)
    cmap_fn = truncate_colormap(base_cmap, minval=0, maxval=.5, n=num_colors)

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(grid_bins):
        for j in range(grid_bins):
            x0, y0 = x_edges[i], y_edges[j]
            w = x_edges[i + 1] - x_edges[i]
            h = y_edges[j + 1] - y_edges[j]
            count = counts[i, j]
            color = cmap_fn(norm(count)) if count > 0 else (1, 1, 1, 0)
            rect = patches.Rectangle((x0, y0), w, h, facecolor=color, edgecolor="none", linewidth=0, alpha=rect_alpha)
            ax.add_patch(rect)

    if include_points:
        ax.scatter(x_vals, y_vals, color=point_color, s=point_size, alpha=point_alpha)

    buffer_x = 0.01 * (xlim[1] - xlim[0])
    buffer_y = 0.01 * (ylim[1] - ylim[0])
    ax.set_xlim(xlim[0] - buffer_x, xlim[1] + buffer_x)
    ax.set_ylim(ylim[0] - buffer_y, ylim[1] + buffer_y)

    ax.set_xticks(x_edges)
    ax.set_yticks(y_edges)
    ax.grid(True, linestyle=grid_line_style, color=grid_line_color, alpha=grid_line_alpha, linewidth=grid_line_width)
    sns.despine(ax=ax, top=True, right=True)

    # Label axes (log-aware)
    x_label_final = f"log₂({x_label})" if log_scaled else x_label
    y_label_final = f"log₂({y_label})" if log_scaled else y_label
    ax.set_xlabel(x_label_final, labelpad=10, fontsize=16)
    ax.set_ylabel(y_label_final, labelpad=10, fontsize=16)

    log_note = " (log-log scale)" if log_scaled else ""
    ax.set_title(f"{title}{log_note}", pad=15, fontsize=18)

    sm = plt.cm.ScalarMappable(cmap=cmap_fn, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Frequency", fontsize=12)

    legend_handles = []

    if show_diag_line:
        diag_x = np.linspace(xlim[0], xlim[1], 500)
        diag_line, = ax.plot(
            diag_x, diag_x - 1,  # Adjust as needed — log(y) = log(x) + c
            linestyle="solid",
            color="#8cc63e",
            linewidth=2,
            alpha=0.9,
            label=r"$\log_2(y) = \log_2(x) - 1$"
        )
        legend_handles.append(diag_line)

    if show_lm_fit:
        df = pd.DataFrame({x_label_final: x_vals, y_label_final: y_vals})
        sns.regplot(data=df, x=x_label_final, y=y_label_final, scatter=False, ax=ax,
                    color="red", line_kws={"linewidth": 2})
        lm_line = ax.lines[-1]
        lm_line.set_label("Linear fit")
        legend_handles.append(lm_line)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower right",
            ncol=len(legend_handles),
            frameon=True,
            bbox_to_anchor=(.8, 0.15),
            facecolor="white",
            fontsize=12
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        plt.tight_layout()

    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, name)
    plt.savefig(f"{base_path}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    if include_pdf:
        plt.savefig(f"{base_path}.pdf", bbox_inches="tight")
    plt.show()
    plt.close()


def plot_rank_scatter(
        df,
        rank_col_1,
        rank_col_2,
        title_1,
        title_2,
        results_dir,
        image_dir,
        alternate_text,
        file_stub="rank_scatter",
        show_spearman=True,  # <-- NEW
        show_r2=False,  # <-- NEW
):
    unique_x = df[rank_col_1].nunique()
    unique_y = df[rank_col_2].nunique()

    aspect_ratio = unique_x / unique_y
    width = 8
    height = 8

    plt.figure(figsize=(width, height))
    plt.tight_layout()

    rank_counts = pd.crosstab(df[rank_col_1], df[rank_col_2])

    df["rank_pair_count"] = df.apply(
        lambda row: rank_counts.loc[row[rank_col_1], row[rank_col_2]], axis=1
    )

    min_size, max_size = 1000, 2000
    df["size"] = df["rank_pair_count"].apply(
        lambda count: min_size
                      + (max_size - min_size)
                      * (count - rank_counts.values.min())
                      / (rank_counts.values.max() - rank_counts.values.min())
    )

    scatter = sns.scatterplot(
        data=df,
        x=rank_col_1,
        y=rank_col_2,
        hue="rank_pair_count",
        palette="viridis",
        size="size",
        sizes=(min_size, max_size),
        legend=None,
    )

    plt.xlabel(title_1, fontsize=14)
    plt.ylabel(title_2, fontsize=14)
    plt.xticks(range(1, unique_x + 1), fontsize=17)
    plt.yticks(range(1, unique_y + 1), fontsize=17)

    max_entry = rank_counts.values.max()
    norm = plt.Normalize(vmin=0, vmax=max_entry)
    sm = plt.cm.ScalarMappable(cmap="mako", norm=norm)
    sm.set_array([])

    # Optional annotation
    stats_lines = []
    x_vals = df[rank_col_1].values
    y_vals = df[rank_col_2].values

    if show_spearman:
        rho, p_value = spearmanr(x_vals, y_vals)
        stats_lines.append(f"Spearman's $\\rho = {rho:.2f}, p\_value < {p_value:.3f}$")
    if show_r2:
        r2 = r2_score(x_vals, y_vals)
        stats_lines.append(f"Goodness of fit $R^2 = {r2:.2f}$")

    if stats_lines:
        stats_text = "\n".join(stats_lines)
        plt.gca().text(
            0.02, 0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=20,
            color="Black",
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3")
        )

    # Bottom-right annotation box
    alternate_text = alternate_text
    plt.gca().text(
        0.98, 0.02,  # bottom right
        alternate_text,
        transform=plt.gca().transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=24,
        color="black",
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3")
    )

    plt.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.1)

    os.makedirs(os.path.join(results_dir, image_dir), exist_ok=True)
    # file_name = f"{file_stub}_{rank_col_1}_vs_{rank_col_2}".replace(" ", "_").replace("/", "")[:146]
    file_name = f"{file_stub}"

    file_path = os.path.join(results_dir, image_dir, f"{file_name}.png")
    file_path_pdf = os.path.join(results_dir, image_dir, f"{file_name}.pdf")

    plt.savefig(file_path, dpi=600, bbox_inches="tight")
    plt.savefig(file_path_pdf, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
