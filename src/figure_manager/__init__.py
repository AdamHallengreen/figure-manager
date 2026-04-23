# figure_manager/__init__.py
from .figure_manager import CycleConfig, FigureManager
from .plotter import generate_plot

__version__ = "0.1.0"

__all__ = [
    "FigureManager",
    "CycleConfig",
    "generate_plot",
]
