
import pytest
from matplotlib.figure import Figure

from figure_manager.figure_manager import FigureManager


@pytest.fixture
def figure_manager():
    """Fixture to create a FigureManager instance."""
    return FigureManager(output_dir="test_figures", paper_size="A4", use_latex=False)


def test_initialization(figure_manager):
    """Test initialization of FigureManager."""
    assert figure_manager.output_dir == "test_figures"
    assert figure_manager.paper_size == "a4"
    assert figure_manager.file_ext == ".pdf"
    assert figure_manager.dpi == 300
    assert not figure_manager.use_latex


def test_create_figure(figure_manager):
    """Test creating a figure with subplots."""
    fig, axes = figure_manager.create_figure(2, 2, 3)
    assert isinstance(fig, Figure)
    assert len(axes) == 3
    assert figure_manager.n_rows == 2
    assert figure_manager.n_cols == 2
    assert figure_manager.n_subplots == 3


def test_save_figure(figure_manager, monkeypatch, tmp_path):
    """Test saving a figure."""
    # Use tmp_path for the output directory
    monkeypatch.setattr(figure_manager, "output_dir", tmp_path)

    figure_manager.create_figure(1, 1, 1)
    figure_manager.save_figure("test_figure")

    assert (tmp_path / "test_figure.pdf").exists()
    assert (tmp_path / "test_figure_subplot_1.pdf").exists()


def test_invalid_create_figure(figure_manager):
    """Test invalid inputs for create_figure."""
    with pytest.raises(ValueError):
        figure_manager.create_figure(2, 2, 5)  # More subplots than available slots


def test_save_without_create(figure_manager):
    """Test saving a figure without creating one."""
    with pytest.raises(RuntimeError):
        figure_manager.save_figure("test_figure")
