# matpublib/__init__.py
from .composer import CycleConfig, FigureComposer
from .plotter import generate_plot

__version__ = "0.1.0"

__all__ = [
    "FigureComposer",
    "CycleConfig",
    "generate_plot",
]
