import polars as pl
import matplotlib.pyplot as plt

def create_plot(ax: plt.Axes = None, df: pl.DataFrame, x_col: str, y_col: str, plot_method: str = 'plot', group_cols: list = None, agg_func=None, **kwargs) -> plt.Axes:
    # Create a new figure and axis if ax is not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Grouping and aggregating data if group_cols and agg_func are provided
    if group_cols and agg_func:
        df = df.groupby(group_cols).agg(agg_func(df[y_col]))

    # Extracting data from the DataFrame
    x = df[x_col].to_list()
    y = df[y_col].to_list()

    # Plotting the data using the specified plot method
    plot_func = getattr(ax, plot_method)
    plot_func(x, y, **kwargs)

    # Adding labels and title
    ax.set_xlabel(kwargs.get('xlabel', 'X-axis'))
    ax.set_ylabel(kwargs.get('ylabel', 'Y-axis'))
    ax.set_title(kwargs.get('title', 'Example Plot'))
    if 'label' in kwargs:
        ax.legend()

    # Return the axis
    return ax
