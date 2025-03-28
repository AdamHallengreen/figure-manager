import os

import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib import transforms


class FigureManager:
    def __init__(
        self,
        output_dir="figures/",
        paper_size="A4",
        file_ext=".pdf",
        dpi=300,
        use_latex=True,
    ):
        """Initialize figure manager with output and style parameters."""
        self.output_dir = output_dir
        self.paper_size = paper_size.lower()
        self.file_ext = file_ext
        self.dpi = dpi
        self.use_latex = use_latex

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Enable LaTeX rendering for text
        if use_latex:
            plt.rc("text", usetex=True)
        else:
            plt.rc("text", usetex=False)
        plt.rc("font", family="serif")

        # Set seaborn defaults
        sns.set_context("paper")  # Optimized for LaTeX documents
        sns.set_palette("deep")

        # Internal figure tracking
        self.fig = None
        self.axes = None
        self.n_rows = None
        self.n_cols = None
        self.n_subplots = None

    def _apply_custom_style(self):
        """Apply custom style settings."""
        # Axes properties
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 0.8
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.grid.axis"] = "y"
        plt.rcParams["axes.grid.which"] = "major"
        plt.rcParams["grid.linestyle"] = "dotted"
        plt.rcParams["grid.linewidth"] = 0.5
        plt.rcParams["grid.alpha"] = 1.0  # Corrected line
        plt.rcParams["axes.facecolor"] = "white"

        # Font properties
        plt.rcParams["font.size"] = 10
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"

        # Lines properties
        plt.rcParams["lines.linewidth"] = 1.0
        plt.rcParams["lines.markersize"] = 5
        plt.rcParams["lines.marker"] = "o"
        plt.rcParams["lines.markeredgewidth"] = 0.5

        # 12 colors for cycling
        colors = [
            "#EF476F",
            "#1082A8",
            "#FFD166",
            "#06EFB1",
            "#08485E",
            "#FF6F61",
            "#1B998B",
            "#C6C013",
            "#5A189A",
            "#A7C957",
            "#D4A5A5",
            "#3D348B",
        ]

        # Cycle through colors only
        # plt.rcParams['axes.prop_cycle'] = cycler('color', colors)
        # Cycle through colors and line styles and markers
        plt.rcParams["axes.prop_cycle"] = (
            cycler("color", colors)
            + cycler("linestyle", 3 * ["-", "--", "-.", ":"])
            + cycler("marker", 2 * ["o", "s", "D", "^", "v", "x"])
        )

        # Ticks properties
        plt.rcParams["xtick.major.size"] = 4
        plt.rcParams["xtick.major.width"] = 0.8
        plt.rcParams["xtick.minor.size"] = 2
        plt.rcParams["xtick.minor.width"] = 0.5
        plt.rcParams["ytick.major.size"] = 4
        plt.rcParams["ytick.major.width"] = 0.8
        plt.rcParams["ytick.minor.size"] = 2
        plt.rcParams["ytick.minor.width"] = 0.5

        # Legend properties
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.fontsize"] = 9

        # Figure properties
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["figure.edgecolor"] = "white"
        plt.rcParams["figure.dpi"] = 300

        # Save properties
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["savefig.format"] = "pdf"
        plt.rcParams["savefig.transparent"] = True

    def _get_axis_extent(self, ax, padding):
        """Get the full bounding box of an axis including labels, ticks, and titles."""
        self.fig.canvas.draw()
        elements = [ax, ax.xaxis.label, ax.yaxis.label, ax.title]
        bbox = transforms.Bbox.union([el.get_window_extent() for el in elements if el])
        return bbox.expanded(1.0 + padding, 1.0 + padding)

    def _save_subplot(self, ax, filename, padding=0.05):
        """Save individual subplot with precise cropping."""
        try:
            bbox = self._get_axis_extent(ax, padding).transformed(
                self.fig.dpi_scale_trans.inverted()
            )
            self.fig.savefig(
                filename,
                dpi=self.dpi,
                bbox_inches=bbox,
                format=self.file_ext.strip("."),
                transparent=True,
            )
            print(f"Saved subplot to {filename}")
        except Exception as e:
            print(f"Error saving subplot {filename}: {e}")

    def set_figure_size(self, fig, n_rows: int, n_cols: int):
        """Set figure dimensions based on standard paper sizes."""
        paper_dimensions = {"A4": (8.27, 11.69), "A3": (11.69, 16.54)}
        width, height = paper_dimensions.get(
            self.paper_size, paper_dimensions["A4"]
        )  # default to A4

        # Adjust for margins (1 inch total) and maintain aspect ratio
        margin = 0.5  # 0.5 inch margin on each side
        usable_width = width - 2 * margin
        subplot_width = usable_width / n_cols
        subplot_height = subplot_width * 0.75  # Adjusted for better LaTeX fit

        fig.set_size_inches(usable_width, subplot_height * n_rows)

    def create_figure(self, n_rows, n_cols, n_subplots):
        """Create a figure with subplots and apply formatting."""
        # Validate inputs
        if not (
            isinstance(n_rows, int)
            and isinstance(n_cols, int)
            and isinstance(n_subplots, int)
        ):
            raise ValueError("n_rows, n_cols, and n_subplots must be integers")
        if n_subplots > n_rows * n_cols:
            raise ValueError("n_subplots cannot exceed n_rows * n_cols")

        # Apply custom style settings
        self._apply_custom_style()

        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)
        axes = axes.flatten()

        # Deactivate unused subplots
        for i in range(n_subplots, len(axes)):
            axes[i].axis("off")

        # Set figure size
        self.set_figure_size(fig, n_rows, n_cols)

        # Apply styles to active subplots
        for ax in axes[:n_subplots]:
            sns.despine(ax=ax)

        # Store these for later use in save_figure
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_subplots = n_subplots
        self.fig = fig
        self.axes = axes[:n_subplots]  # store active axes.

        return fig, axes[:n_subplots]

    def save_figure(self, filename="figure"):
        """Save the full figure and individual subplots."""
        # Ensure create_figure has been called
        if self.fig is None or self.axes is None:
            raise RuntimeError("Call create_figure before saving the figure.")

        # Apply tight layout before saving
        self.fig.tight_layout()  # Add tight_layout here!

        # Save the full figure
        full_path = os.path.join(self.output_dir, f"{filename}{self.file_ext}")
        try:
            self.fig.savefig(
                full_path,
                dpi=self.dpi,
                bbox_inches="tight",
                format=self.file_ext.strip("."),
                transparent=True,
            )
            print(f"Saved full figure to {full_path}")
        except Exception as e:
            print(f"Error saving full figure: {e}")

        # Save each subplot separately
        for i, ax in enumerate(self.axes):
            subplot_path = os.path.join(
                self.output_dir, f"{filename}_subplot_{i + 1}{self.file_ext}"
            )
            self._save_subplot(ax, subplot_path)
