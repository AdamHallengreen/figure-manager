# matpublib

Publication-quality Matplotlib figures with standardized sizing, LaTeX support, and subplot management.

## Installation

pip install matpublib

## Usage

from matpublib import FigureComposer

fc = FigureComposer(paper_size="A4", use_latex=False)
fig, axes = fc.create_figure(n_rows=1, n_cols=2)
fc.save_figure(fig, "output.pdf")