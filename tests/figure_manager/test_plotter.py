import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")  # Use a non-interactive backend for testing
import matplotlib.pyplot as plt

from figure_manager.plotter import (
    _get_min_count_info,
    _prepare_plot_data,
    _sort_groups,
    generate_plot,
)


@pytest.fixture
def sample_data():
    """Fixture to create sample Polars DataFrame."""
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "group": ["A", "A", "B", "B", "B"],
        }
    )


def test_get_min_count_info(sample_data):
    """Test _get_min_count_info function."""
    min_count, min_position = _get_min_count_info(sample_data, "x", bins=5)
    assert min_count == 1
    assert min_position == pytest.approx(1.0)


def test_sort_groups(sample_data):
    """Test _sort_groups function."""
    grouped = _sort_groups(sample_data, group_by_cols=["group"], sort_order=["B", "A"])
    assert list(grouped.keys()) == [("B",), ("A",)]


def test_prepare_plot_data(sample_data):
    """Test _prepare_plot_data function."""
    pre_agg, sorted_groups = _prepare_plot_data(
        sample_data, x_col="x", y_col="y", group_by_cols=["group"]
    )
    assert len(pre_agg) == 2
    assert len(sorted_groups) == 2
    assert ("A",) in sorted_groups
    assert ("B",) in sorted_groups


def test_generate_plot(sample_data):
    """Test generate_plot function."""
    fig, ax = plt.subplots()
    ax = generate_plot(
        data=sample_data,
        x="x",
        y="y",
        group_by=["group"],
        plot_type="plot",
        ax=ax,
        verbose=True,
    )
    assert ax.has_data()


def test_generate_plot_invalid_data():
    """Test generate_plot with invalid data."""
    with pytest.raises(TypeError):
        generate_plot(data="invalid_data", x="x", y="y")
