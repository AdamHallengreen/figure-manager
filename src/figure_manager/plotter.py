from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes


def _print_verbose(message: str, warning: bool = False) -> None:
    """Prints a verbose message with optional warning."""
    if warning:
        print(f"WARNING: {message}")
    else:
        print(message)


def _get_min_count_info(
    data: pl.DataFrame, x_col: str, bins: int | None = None
) -> tuple:
    """Calculates min count and position for verbose output."""
    if bins is not None:
        counts, bin_edges = np.histogram(data[x_col].to_numpy(), bins=bins)
        min_count = counts.min()
        min_position = bin_edges[np.argmin(counts)]
    else:
        grouped_counts = data.group_by(x_col).agg(pl.len())
        min_count = grouped_counts["len"].min()
        min_position = (
            grouped_counts.filter(pl.col("len") == min_count)
            .select(x_col)
            .to_series()
            .to_list()
        )
    return min_count, min_position


def _setup_plot_axes(
    ax: Axes, x_col: str, y_col: str | None, plot_type: str, ax_settings: dict
) -> None:
    """Sets up plot axes with labels, title, and limits."""
    ax.set_xlabel(ax_settings.get("xlabel", x_col).capitalize())
    if y_col:
        ax.set_ylabel(ax_settings.get("ylabel", y_col).capitalize())
        ax.set_title(
            ax_settings.get("title", f"{plot_type.capitalize()} of {y_col} by {x_col}")
        )
    else:
        ax.set_title(ax_settings.get("title", f"{plot_type.capitalize()} of {x_col}"))

    ax.set_xlim(ax_settings.get("xlim")) if "xlim" in ax_settings else None
    ax.set_ylim(ax_settings.get("ylim")) if "ylim" in ax_settings else None


def _sort_groups(
    data: pl.DataFrame,
    group_by_cols: list[str] | None = None,
    sort_order: list[tuple | str | int] | None = None,
) -> dict[tuple, pl.DataFrame]:
    """
    Sorts groups in a DataFrame based on a custom or default order,
    placing None groups last.

    Args:
        data: The Polars DataFrame to group and sort.
        group_by_cols: List of columns to group by.
        sort_order: List defining the desired order of groups.
            If provided, groups not in the order are appended after.

    Returns:
        A dictionary where keys are group names (tuples) and values are DataFrames.
    """

    grouped_data = {
        group_name: group_data
        for group_name, group_data in data.group_by(group_by_cols)
    }

    if sort_order:
        # Normalize sort_order to a list of tuples
        normalized_sort_order: list[tuple] = [
            (item,) if not isinstance(item, tuple) else item for item in sort_order
        ]

        sorted_groups: dict[tuple, pl.DataFrame] = {}
        remaining_groups: dict[tuple, pl.DataFrame] = {}

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

        def sort_key(
            item: tuple[tuple, pl.DataFrame],
        ) -> tuple[int, tuple | str | int | None]:
            group_name = item[0]
            if group_name == (None,):
                return (2, None)
            elif isinstance(group_name, tuple):
                return (0, group_name)
            else:
                return (1, group_name)

        return dict(sorted(grouped_data.items(), key=sort_key))


def _prepare_plot_data(
    data: pl.DataFrame,
    x_col: str | None = None,
    y_col: str | None = None,
    group_by_cols: list[str] | str | None = None,
    sort_order: list[tuple | str | int] | None = None,
    agg_fct: Callable | None = None,
    bins: int | None = None,
    label: str | None = None,
    verbose: bool = False,
) -> tuple[dict[tuple, pl.DataFrame], dict[tuple, pl.DataFrame]]:
    """Prepares and sorts data for plotting."""
    if isinstance(group_by_cols, str):
        group_by_cols = [group_by_cols]
    elif isinstance(group_by_cols, list):
        group_by_cols = group_by_cols
    else:
        group_by_cols = []

    if not group_by_cols:
        pre_agg_data = {(label,): data}
    else:
        pre_agg_data = _sort_groups(data, group_by_cols, sort_order=sort_order)

    if agg_fct and y_col:
        over_columns = [x_col] + group_by_cols
        data = (
            data.group_by(over_columns)
            .agg(agg_fct(y_col).alias(y_col))
            .sort(over_columns)
        )

    if not group_by_cols:
        sorted_groups = {(label,): data}
    else:
        sorted_groups = _sort_groups(data, group_by_cols, sort_order=sort_order)

    return pre_agg_data, sorted_groups


def generate_plot(
    data: pl.DataFrame,
    x: str,
    y: str | None = None,
    plot_type: str = "plot",
    group_by: list[str] | str | None = None,
    agg_fct: Callable | None = None,
    ax: Axes | None = None,
    label: str | None = None,
    plot_settings: dict | None = None,
    verbose: bool = False,
    bins: int | None = None,
    sort_order: list[tuple | str | int] | None = None,
    y_err: str | list[str] | tuple[str, str] | None = None,
    **ax_settings,  # pyright: ignore[reportMissingParameterType]
) -> Axes:
    """
    Generates a plot from a Polars DataFrame with optional error bars
    or confidence intervals.

    Parameters:
        data (pl.DataFrame):
            The input data as a Polars DataFrame.
        x (str):
            The column name to use for the x-axis.
        y (str, optional):
            The column name to use for the y-axis.
            Required for most plot types except histograms.
        plot_type (str, optional):
            The type of plot to generate (e.g., "plot", "scatter", "bar", "hist").
            Defaults to "plot".
        group_by (list, optional):
            A list of column names to group the data by before plotting.
        agg_fct (callable, optional):
            An aggregation function to apply to grouped data.
        ax (plt.Axes, optional):
            A Matplotlib Axes object to plot on. If not provided,
            a new Axes will be created.
        label (str, optional):
            A label for the plot. Used in the legend if provided.
        plot_settings (dict, optional):
            Additional keyword arguments to customize the plot (e.g., color, linestyle).
        verbose (bool, optional):
            If True, prints detailed information about the plotting process.
            Defaults to False.
        bins (int, optional):
            The number of bins to use for histograms. Defaults to 10.
        sort_order (list, optional):
            A list specifying the order of groups for plotting.
        y_err (str or list/tuple, optional):
            Column name(s) for error bars or confidence intervals.
            If a string, it specifies the column for symmetric error bars.
            If a tuple/list, it should contain two column names for
            lower and upper bounds of confidence intervals.
        **ax_settings:
            Additional keyword arguments for customizing the Axes
            (e.g., axis labels, limits).

    Returns:
        plt.Axes: The Matplotlib Axes object containing the plot.

    Raises:
        TypeError:
            If `data` is not a Polars DataFrame, or if `y_err` is not a valid type.
        ValueError:
            If `x`, `y`, or `y_err` are not valid column names in the DataFrame.

    Notes:
        - If `group_by` is provided, the data will be grouped, and a separate plot
            will be created for each group.
        - For histograms, only `x` is required, and `y` is ignored.
        - If `y_err` is provided, error bars or confidence intervals will be added to
            the plot.
        - The function supports verbose mode to provide detailed insights into the data
            and plotting process.

    Example:
        ```python
        import matplotlib.pyplot as plt

        # Example DataFrame
        df = pl.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "group": ["A", "A", "B", "B", "B"]
        })

        # Generate a grouped line plot
        fig, ax = plt.subplots()
        generate_plot(
            data=df,
            x="x",
            y="y",
            group_by=["group"],
            plot_type="plot",
            ax=ax,
            verbose=True
        )
        plt.show()
        ```
    """
    if not isinstance(data, pl.DataFrame):
        raise TypeError("Data must be a Polars DataFrame.")
    if x not in data.columns:
        raise ValueError("x must be a valid column name.")
    if y and y not in data.columns and plot_type != "hist":
        raise ValueError("y must be a valid column name.")
    if isinstance(y_err, str):
        if y_err not in data.columns:
            raise ValueError("y_err must be a valid column name if provided.")
    elif (isinstance(y_err, tuple | list)) and (
        not all(isinstance(col, str) and col in data.columns for col in y_err)
    ):
        raise ValueError("All entries in y_err must be valid column names if provided.")

    ax = ax or plt.subplots()[1]
    if ax is None:
        raise ValueError("ax must be a valid matplotlib Axes instance.")
    plot_settings = plot_settings or {}
    bins = bins or 10
    group_by = group_by or []
    y_err = y_err or ""

    pre_agg_data, sorted_groups = _prepare_plot_data(
        data, x, y, group_by, sort_order, agg_fct, bins, label, verbose
    )

    for group_name, group_data in sorted_groups.items():
        x_values = group_data[x].to_numpy()
        group_label = (
            ", ".join(map(str, group_name))
            if isinstance(group_name, tuple)
            else str(group_name)
        )

        if plot_type == "hist":
            ax.hist(x_values, bins=bins, label=group_label, **plot_settings)
            if verbose:
                min_count, min_position = _get_min_count_info(
                    pre_agg_data[group_name], x, bins
                )
                message = (
                    f"  Group ({group_label}) uses "
                    f"{len(pre_agg_data[group_name])} "
                    f"observations with fewest ({min_count}) "
                    f"at '{x}'={min_position}."
                )
                _print_verbose(
                    message,
                    min_count <= 5,
                )
        else:
            plot_func = getattr(ax, plot_type)
            if y:
                y_values = group_data[y].to_numpy()
                if y_err:
                    if isinstance(y_err, str):
                        y_err_values = group_data[y_err].to_numpy()
                        if y_err_values is None:
                            raise ValueError(
                                f"y_err column '{y_err}' contains None values."
                            )
                        plot_settings.setdefault("capsize", 3)
                        ax.errorbar(
                            x_values,
                            y_values,
                            yerr=y_err_values,
                            label=group_label,
                            **plot_settings,
                        )
                    elif isinstance(y_err, tuple | list):
                        y_ci_low = group_data[y_err[0]].to_numpy()
                        y_ci_high = group_data[y_err[1]].to_numpy()
                        plot_func(
                            x_values, y_values, label=group_label, **plot_settings
                        )
                        ax.fill_between(
                            x_values, y_ci_low, y_ci_high, alpha=0.3, **plot_settings
                        )
                    else:
                        message = (
                            "y_err must be a string, a tuple of strings, "
                            "or a Polars Series."
                        )
                        raise TypeError(message)
                else:
                    plot_func(x_values, y_values, label=group_label, **plot_settings)
            else:
                plot_func(x_values, label=group_label, **plot_settings)

            if verbose:
                min_count, min_position = _get_min_count_info(
                    pre_agg_data[group_name], x
                )
                message = (
                    f"  Group ({group_label}) uses "
                    f"{len(pre_agg_data[group_name])} "
                    f"observations with fewest ({min_count}) "
                    f"at '{x}'={min_position}."
                )
                _print_verbose(message, min_count <= 5)

    if group_by or label:
        ax.legend()

    if verbose:
        print("")

    _setup_plot_axes(ax, x, y, plot_type, ax_settings)
    return ax
