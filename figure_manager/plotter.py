import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Union

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

def _sort_groups(
    data: pl.DataFrame,
    group_by_cols: List[str] = None,
    sort_order: List[Union[Tuple, str, int]] = None,
) -> Dict[Tuple, pl.DataFrame]:
    """
    Sorts groups in a DataFrame based on a custom or default order, placing None groups last.

    Args:
        data: The Polars DataFrame to group and sort.
        group_by_cols: List of columns to group by.
        sort_order: List defining the desired order of groups.
            If provided, groups not in the order are appended after.

    Returns:
        A dictionary where keys are group names (tuples) and values are DataFrames.
    """

    grouped_data = {group_name: group_data for group_name, group_data in data.group_by(group_by_cols)}

    if sort_order:
        # Normalize sort_order to a list of tuples
        normalized_sort_order: List[Tuple] = [
            (item,) if not isinstance(item, tuple) else item for item in sort_order
        ]

        sorted_groups: Dict[Tuple, pl.DataFrame] = {}
        remaining_groups: Dict[Tuple, pl.DataFrame] = {}

        for group_name in normalized_sort_order:
            if group_name in grouped_data:
                sorted_groups[group_name] = grouped_data.pop(group_name)

        remaining_groups = {
            group_name: group_data
            for group_name, group_data in grouped_data.items()
            if group_name != (None,)
        }

        if (None,) in grouped_data:
            sorted_groups[(None,)] = grouped_data[(None,)]

        sorted_groups.update(remaining_groups)
        return sorted_groups

    else:
        def sort_key(item: Tuple[Tuple, pl.DataFrame]) -> Tuple[int, Union[Tuple, str, int, None]]:
            group_name = item[0]
            if group_name == (None,):
                return (2, None)
            elif isinstance(group_name, tuple):
                return (0, group_name)
            else:
                return (1, group_name)

        return dict(sorted(grouped_data.items(), key=sort_key))

def generate_plot(
    data: pl.DataFrame,
    x_col: str,
    y_col: str = None,
    plot_type: str = "plot",
    group_by_cols: list = None,
    agg_fct=None,
    ax: plt.Axes = None,
    label = None,
    plot_settings: dict = None,
    verbose: bool = False,
    bins: int = 10,
    sort_order: list= None,
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

    # Handle case where no grouping is specified
    if not group_by_cols:
        pre_agg_data = {(label,): data}
    else:
        pre_agg_data = _sort_groups(data, group_by_cols, sort_order=sort_order)

    # Apply aggregation if specified
    if agg_fct and y_col:
        over_columns = [x_col] + (group_by_cols or [])
        data = data.group_by(over_columns).agg(agg_fct(y_col).alias(y_col)).sort(over_columns)

    # Iterate over groups (or the entire dataset if no grouping)
    if not group_by_cols:
        sorted_groups = {(label,): data}
    else:
        sorted_groups = _sort_groups(data, group_by_cols, sort_order=sort_order)

    for group_name, group_data in sorted_groups.items():
        x_values = group_data[x_col].to_numpy()
        y_values = group_data[y_col].to_numpy() if y_col else None
        group_label = ", ".join(map(str, group_name)) if isinstance(group_name, tuple) else str(group_name)

        # Generate the plot based on the plot_type
        if plot_type == "hist":
            ax.hist(x_values, bins=bins, label=group_label, **plot_settings)
            if verbose:
                # Calculate and print verbose output for histograms
                min_count, min_position = _get_min_count_info(pre_agg_data[group_name], x_col, bins)
                _print_verbose(f"  Group ({group_label}) uses {len(pre_agg_data[group_name])} observations with fewest ({min_count}) at '{x_col}'={min_position}.", min_count <= 5)
        else:
            plot_func = getattr(ax, plot_type)
            if y_col:
                plot_func(x_values, y_values, label=group_label, **plot_settings)
            else:
                plot_func(x_values, label=group_label, **plot_settings)
            if verbose:
                # Calculate and print verbose output for other plot types
                min_count, min_position = _get_min_count_info(pre_agg_data[group_name], x_col)
                _print_verbose(f"  Group ({group_label}) uses {len(pre_agg_data[group_name])} observations with fewest ({min_count}) at '{x_col}'={min_position}.", min_count <= 5)

    # Add legend if grouping was used or label is provided
    if group_by_cols or label:
        ax.legend()

    # Print a newline for better readability of verbose output
    if verbose:
        print("")

    # Setup the plot axes with labels, title, and limits
    _setup_plot_axes(ax, x_col, y_col, plot_type, ax_settings)
    return ax

def generate_plot_with_error(
    x: pl.Series,
    y: pl.Series,
    y_err: Union[pl.Series, Tuple[pl.Series, pl.Series]] = None,
    plot_type: str = "plot",
    ax: plt.Axes = None,
    label = None,
    plot_settings: dict = None,
    **ax_settings,
) -> plt.Axes:
    """Generates a plot with error bars or confidence intervals from Polars Series."""
    
    # Input validation
    if not isinstance(x, pl.Series) or not isinstance(y, pl.Series):
        raise TypeError("x and y must be Polars Series.")
    if y_err is not None:
        if isinstance(y_err, pl.Series) and not isinstance(y_err, pl.Series):
            raise TypeError("y_err must be a Polars Series or a tuple of Polars Series.")
        if isinstance(y_err, tuple) and (len(y_err) != 2 or not all(isinstance(e, pl.Series) for e in y_err)):
            raise TypeError("y_err must be a tuple of two Polars Series.")

    ax = ax or plt.subplots()[1]
    plot_settings = plot_settings or {}

    # Generate the plot based on the plot_type
    plot_func = getattr(ax, plot_type)
    x_np, y_np = x.to_numpy(), y.to_numpy()

    if isinstance(y_err, pl.Series):
        # set default format for error bars
        plot_settings.setdefault("capsize", 3)
        ax.errorbar(x_np, y_np, yerr=y_err.to_numpy(), label=label, **plot_settings)
    elif isinstance(y_err, tuple):
        y_ci_low, y_ci_high = y_err
        plot_func(x_np, y_np, label=label, **plot_settings)
        ax.fill_between(x_np, y_ci_low.to_numpy(), y_ci_high.to_numpy(), alpha=0.3, **plot_settings)
    else:
        plot_func(x_np, y_np, label=label, **plot_settings)

    # Setup the plot axes with labels, title, and limits
    _setup_plot_axes(ax, x.name, y.name, plot_type, ax_settings)
    
    if label:
        ax.legend()
    return ax
