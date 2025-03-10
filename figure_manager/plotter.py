import matplotlib.pyplot as plt
import polars as pl

def _select_matplotlib_plot_function(ax: plt.Axes, plot_type: str):
    """Selects the Matplotlib plotting function based on the plot type."""
    if plot_type in ['plot', 'scatter', 'bar']:
        return getattr(ax, plot_type)
    elif plot_type == 'hist':
        return ax.hist
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

def _execute_plot_function(plot_function, plot_type: str, x_data: list, y_data: list, plot_options: dict = None, label: str = None):
    """Executes the selected plotting function with the appropriate data."""
    if plot_type == 'hist':
        plot_function(y_data, label=label, **(plot_options or {}))
    else:
        plot_function(x_data, y_data, label=label, **(plot_options or {}))

def generate_plot(data: pl.DataFrame,
                  x_col: str,
                  y_col: str,
                  plot_type: str = 'plot',
                  group_by_cols: list = None,
                  agg_fct=None,
                  ax: plt.Axes = None,
                  plot_settings: dict = None,
                  **ax_settings) -> plt.Axes:
    """Generates a plot from a Polars DataFrame with dynamic plot type selection.

    Args:
        data: Polars DataFrame containing the data.
        x_col: Name of the column for the x-ax.
        y_col: Name of the column for the y-ax.
        plot_type: Type of plot (e.g., 'plot', 'scatter', 'bar', 'hist').
        group_by_cols: List of columns to group by.
        agg_fct: Aggregation function to apply to the y-column.
        ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        plot_settings: Dictionary of keyword arguments to pass to the plot function.
        ax_settings: Additional keyword arguments for labels, title, and ax limits.

    Returns:
        Matplotlib Axes object.
    """

    # Input validation
    if not isinstance(data, pl.DataFrame):
        raise TypeError("Data must be a Polars DataFrame.")
    if x_col not in data.columns or y_col not in data.columns:
        raise ValueError("x_col and y_col must be valid column names.")

    # Convert group_by_cols to list if it is a string
    if isinstance(group_by_cols, str):
        group_by_cols = [group_by_cols]

    # Create a new figure and ax if ax is not provided
    if ax is None:
        _, ax = plt.subplots()

    # Group and aggregate data if group_by_cols and agg_fct are provided
    if agg_fct:
        over_columns = [x_col] + (group_by_cols or [])
        data = data.group_by(over_columns).agg(agg_fct(y_col).alias(y_col)).sort(over_columns)

    # Select plotting function based on plot_type
    plotting_function = _select_matplotlib_plot_function(ax, plot_type)

    # Extract data from the DataFrame and execute the plotting function
    if group_by_cols:
        for group_name, group_data in data.group_by(group_by_cols):
            x_values = group_data[x_col].to_numpy().tolist()
            y_values = group_data[y_col].to_numpy().tolist()
            # Convert tuple group_name to a string
            group_label = ', '.join(map(str, group_name)) if isinstance(group_name, tuple) else str(group_name)
            _execute_plot_function(plotting_function, plot_type, x_values, y_values, plot_settings, label=group_label)
        ax.legend()  # Let Matplotlib handle the legend.
    else:
        x_values = data[x_col].to_numpy().tolist()
        y_values = data[y_col].to_numpy().tolist()
        _execute_plot_function(plotting_function, plot_type, x_values, y_values, plot_settings)

    # Add labels and title
    ax.set_xlabel(ax_settings.get('xlabel', x_col))
    ax.set_ylabel(ax_settings.get('ylabel', y_col))
    ax.set_title(ax_settings.get('title', f'{plot_type.capitalize()} of {y_col} by {x_col}'))

    # Set ax limits if provided
    if 'xlim' in ax_settings:
        ax.set_xlim(ax_settings['xlim'])
    if 'ylim' in ax_settings:
        ax.set_ylim(ax_settings['ylim'])

    return ax