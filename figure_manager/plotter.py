import matplotlib.pyplot as plt
import polars as pl
import numpy as np

def _print_verbose(message: str, warning: bool = False) -> None:
    """Prints a verbose message with optional warning."""
    if warning:
        print(f"WARNING: {message}")
    else:
        print(message)

def _get_min_count_info(data: pl.DataFrame, x_col: str, bins: int = None) -> tuple:
    """Calculates min count and position for verbose output."""
    if bins is not None:
        counts, bin_edges = np.histogram(data[x_col].to_numpy(), bins=bins)
        min_count = counts.min()
        min_position = bin_edges[np.argmin(counts)]
    else:
        grouped_counts = data.group_by(x_col).agg(pl.count())
        min_count = grouped_counts["count"].min()
        min_position = grouped_counts.filter(pl.col("count") == min_count).select(x_col).to_series().to_list()
    return min_count, min_position

def _setup_plot_axes(ax: plt.Axes, x_col: str, y_col: str, plot_type: str, ax_settings: dict) -> None:
    """Sets up plot axes with labels, title, and limits."""
    ax.set_xlabel(ax_settings.get("xlabel", x_col).capitalize())
    if y_col:
        ax.set_ylabel(ax_settings.get("ylabel", y_col).capitalize())
        ax.set_title(ax_settings.get("title", f"{plot_type.capitalize()} of {y_col} by {x_col}"))
    else:
        ax.set_title(ax_settings.get("title", f"{plot_type.capitalize()} of {x_col}"))

    ax.set_xlim(ax_settings.get("xlim")) if "xlim" in ax_settings else None
    ax.set_ylim(ax_settings.get("ylim")) if "ylim" in ax_settings else None

def generate_plot(
    data: pl.DataFrame,
    x_col: str,
    y_col: str = None,
    plot_type: str = "plot",
    group_by_cols: list = None,
    agg_fct=None,
    ax: plt.Axes = None,
    plot_settings: dict = None,
    verbose: bool = False,
    bins: int = 10,
    **ax_settings,
) -> plt.Axes:
    """Generates a plot from a Polars DataFrame."""

    # Input validation and basic setup
    if not isinstance(data, pl.DataFrame):
        raise TypeError("Data must be a Polars DataFrame.")
    if x_col not in data.columns or (y_col and y_col not in data.columns and plot_type != "hist"):
        raise ValueError("x_col and y_col must be valid column names.")

    group_by_cols = [group_by_cols] if isinstance(group_by_cols, str) else group_by_cols
    ax = ax or plt.subplots()[1]
    plot_settings = plot_settings or {}

    # Create a dictionary to hold the data, grouped if necessary, before aggregation
    group_data_dict = (
        {group_name: group_data for group_name, group_data in data.group_by(group_by_cols)}
        if group_by_cols
        else {None: data}
    )

    # Apply aggregation if specified
    if agg_fct and y_col:
        over_columns = [x_col] + (group_by_cols or [])
        data = data.group_by(over_columns).agg(agg_fct(y_col).alias(y_col)).sort(over_columns)

    # Iterate over groups (or the entire dataset if no grouping)
    for group_name, group_data in data.group_by(group_by_cols) if group_by_cols else [(None, data)]:
        x_values = group_data[x_col].to_numpy()
        y_values = group_data[y_col].to_numpy() if y_col else None
        group_label = ", ".join(map(str, group_name)) if isinstance(group_name, tuple) else str(group_name)

        # Generate the plot based on the plot_type
        if plot_type == "hist":
            ax.hist(x_values, bins=bins, label=group_label, **plot_settings)
            if verbose:
                # Calculate and print verbose output for histograms
                min_count, min_position = _get_min_count_info(group_data_dict[group_name], x_col, bins)
                _print_verbose(f"  Group ({group_label}) uses {len(group_data_dict[group_name])} observations with fewest ({min_count}) at '{x_col}'={min_position}.", min_count <= 5)
        else:
            plot_func = getattr(ax, plot_type)
            if y_col:
                plot_func(x_values, y_values, label=group_label, **plot_settings)
            else:
                plot_func(x_values, label=group_label, **plot_settings)
            if verbose:
                # Calculate and print verbose output for other plot types
                min_count, min_position = _get_min_count_info(group_data_dict[group_name], x_col)
                _print_verbose(f"  Group ({group_label}) uses {len(group_data_dict[group_name])} observations with fewest ({min_count}) at '{x_col}'={min_position}.", min_count <= 5)

    # Add legend if grouping was used
    if group_by_cols:
        ax.legend()

    # Print a newline for better readability of verbose output
    if verbose:
        print("")

    # Setup the plot axes with labels, title, and limits
    _setup_plot_axes(ax, x_col, y_col, plot_type, ax_settings)
    return ax