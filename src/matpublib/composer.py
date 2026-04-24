from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import Cycler, cycler
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox


@dataclass
class CycleConfig:
    cycle_linestyles: bool = False
    cycle_markers: bool = False
    mode: Literal["together", "after_colors"] = "after_colors"


_CYCLE_PRESETS: dict[str, CycleConfig] = {
    "linestyles": CycleConfig(cycle_linestyles=True, mode="together"),
    "markers": CycleConfig(cycle_markers=True, mode="together"),
    "full": CycleConfig(cycle_linestyles=True, cycle_markers=True, mode="together"),
    "linestyles_sequential": CycleConfig(cycle_linestyles=True),
    "markers_sequential": CycleConfig(cycle_markers=True),
    "full_sequential": CycleConfig(cycle_linestyles=True, cycle_markers=True),
}


def _load_style(source: str | Path | dict) -> dict:  # type: ignore[type-arg]
    if isinstance(source, dict):
        return source
    if isinstance(source, str):
        ref = files("matpublib") / "styles" / f"{source}.json"
        return json.loads(ref.read_text(encoding="utf-8"))
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _load_palette(source: str | Path | dict | list) -> dict:  # type: ignore[type-arg]
    if isinstance(source, dict):
        return source
    if isinstance(source, list):
        return {"colors": source}
    if isinstance(source, str):
        ref = files("matpublib") / "palettes" / f"{source}.json"
        return json.loads(ref.read_text(encoding="utf-8"))
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _build_rc_params(style: dict, use_latex: bool) -> dict:  # type: ignore[type-arg]
    rc: dict = dict(style.get("rc", {}))  # type: ignore[type-arg]
    rc["text.usetex"] = use_latex
    return rc


def _build_prop_cycle(palette: dict, cycle: CycleConfig | None) -> Cycler:  # type: ignore[type-arg]
    colors = palette.get("colors", [])
    linestyles = palette.get("linestyles", [])
    markers = palette.get("markers", [])

    if cycle is None:
        return cycler(color=colors)

    color_cycler = cycler(color=colors)

    if cycle.mode == "together":
        result = color_cycler
        if cycle.cycle_linestyles and linestyles:
            result = result + cycler(linestyle=linestyles)
        if cycle.cycle_markers and markers:
            result = result + cycler(marker=markers)
        return result

    # mode == "after_colors": style changes after all colors exhausted
    if cycle.cycle_linestyles and cycle.cycle_markers and linestyles and markers:
        style_cycler = cycler(linestyle=linestyles) + cycler(marker=markers)
        return style_cycler * color_cycler
    if cycle.cycle_linestyles and linestyles:
        return cycler(linestyle=linestyles) * color_cycler
    if cycle.cycle_markers and markers:
        return cycler(marker=markers) * color_cycler
    return color_cycler


def _get_figure_size(
    style: dict,  # type: ignore[type-arg]
    paper_size: str,
    n_rows: int,
    n_cols: int,
    landscape: bool,
) -> tuple[float, float]:
    layout = style.get("layout", {})
    paper_sizes = layout.get("paper_sizes", {"A4": [8.27, 11.69]})
    page_margin = layout.get("page_margin", 0.5)
    subplot_aspect = layout.get("subplot_aspect", 0.75)

    dims = paper_sizes.get(paper_size, paper_sizes.get("A4", [8.27, 11.69]))
    width = dims[1] if landscape else dims[0]

    usable_width = width - 2 * page_margin
    subplot_height = (usable_width / n_cols) * subplot_aspect
    return usable_width, subplot_height * n_rows


def _get_ax_extent(fig: Figure, ax: Axes, padding: float) -> Bbox:
    fig.canvas.draw()
    elements = [ax, ax.xaxis.label, ax.yaxis.label, ax.title]
    bbox = Bbox.union([el.get_window_extent() for el in elements if el.get_visible()])
    return bbox.expanded(1.0 + padding, 1.0 + padding).transformed(
        fig.dpi_scale_trans.inverted()
    )


def _resolve_path(filename: str | Path, output_dir: Path | None) -> Path:
    p = Path(filename)
    if p.is_absolute():
        return p
    base = output_dir if output_dir is not None else Path.cwd()
    return base / p


class FigureComposer:
    def __init__(
        self,
        output_dir: Path | str | None = None,
        paper_size: str = "A4",
        use_latex: bool = True,
        style: str | Path | dict = "default",  # type: ignore[type-arg]
        palette: str | Path | dict = "deep",  # type: ignore[type-arg]
    ) -> None:
        if output_dir is not None:
            logger.info(
                "output_dir is set to '{}' - figures will be saved relative to "
                "this path instead of the working directory.",
                output_dir,
            )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.paper_size = paper_size
        self.use_latex = use_latex
        self._style: dict = _load_style(style)  # type: ignore[type-arg]
        self._palette: dict = _load_palette(palette)  # type: ignore[type-arg]
        self._rc_params: dict = _build_rc_params(self._style, use_latex)  # type: ignore[type-arg]

    def create_figure(
        self,
        n_rows: int,
        n_cols: int,
        n_subplots: int | None = None,
        landscape: bool = False,
        sharex: bool = False,
        sharey: bool = False,
        subplot_kw: dict | None = None,  # type: ignore[type-arg]
        gridspec_kw: dict | None = None,  # type: ignore[type-arg]
        cycle: CycleConfig | str | None = None,
    ) -> tuple[Figure, list[Axes]]:
        if isinstance(cycle, str):
            if cycle not in _CYCLE_PRESETS:
                raise ValueError(
                    f"Unknown cycle preset {cycle!r}. "
                    f"Choose from: {list(_CYCLE_PRESETS)}"
                )
            cycle = _CYCLE_PRESETS[cycle]
        if n_subplots is None:
            n_subplots = n_rows * n_cols
        if n_subplots > n_rows * n_cols:
            raise ValueError(
                f"n_subplots ({n_subplots}) cannot exceed "
                f"n_rows * n_cols ({n_rows * n_cols})"
            )

        rc = {
            **self._rc_params,
            "axes.prop_cycle": _build_prop_cycle(self._palette, cycle),
        }
        fig_w, fig_h = _get_figure_size(
            self._style, self.paper_size, n_rows, n_cols, landscape
        )

        with mpl.rc_context(rc):
            fig, axes_array = plt.subplots(
                n_rows,
                n_cols,
                squeeze=False,
                layout="constrained",
                sharex=sharex,
                sharey=sharey,
                subplot_kw=subplot_kw or {},
                gridspec_kw=gridspec_kw or {},
            )

        axes: list[Axes] = axes_array.flatten().tolist()  # pyright: ignore[reportAssignmentType]
        fig.set_size_inches(fig_w, fig_h)

        for ax in axes[n_subplots:]:
            ax.axis("off")

        for ax in axes[:n_subplots]:
            sns.despine(ax=ax)

        return fig, axes[:n_subplots]

    def save_figure(
        self,
        fig: Figure,
        filename: str | Path = "figure.pdf",
        transparent: bool = False,
        save_subplots: bool = False,
        subplot_include_title: bool = True,
        subplot_suffix: str = "_subplot",
    ) -> Path:
        path = _resolve_path(filename, self.output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)

        ext = path.suffix
        if not ext:
            logger.warning("No file extension in '{}'; defaulting to .pdf", filename)
            path = path.with_suffix(".pdf")
            ext = ".pdf"
        fmt = ext.lstrip(".")

        dpi = self._style.get("layout", {}).get("dpi", 300)
        fig.savefig(
            path, dpi=dpi, bbox_inches="tight", format=fmt, transparent=transparent
        )
        logger.info("Saved figure to {}", path)

        if save_subplots:
            active_axes = [ax for ax in fig.axes if ax.axison]
            for i, ax in enumerate(active_axes):
                subplot_path = path.parent / f"{path.stem}{subplot_suffix}_{i + 1}{ext}"
                self.save_subplot(
                    fig,
                    ax,
                    subplot_path,
                    include_title=subplot_include_title,
                    transparent=transparent,
                )

        return path

    def save_subplot(
        self,
        fig: Figure,
        ax: Axes,
        filename: str | Path,
        padding: float = 0.05,
        include_title: bool = True,
        transparent: bool = False,
    ) -> Path:
        path = _resolve_path(filename, self.output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)

        ext = path.suffix
        fmt = ext.lstrip(".") if ext else "pdf"

        all_axes = fig.axes
        original_visibility = [a.get_visible() for a in all_axes]
        for other_ax in all_axes:
            if other_ax is not ax:
                other_ax.set_visible(False)

        original_title = ax.get_title()
        if not include_title:
            ax.set_title("")

        dpi = self._style.get("layout", {}).get("dpi", 300)
        bbox = _get_ax_extent(fig, ax, padding)
        fig.savefig(
            path, dpi=dpi, bbox_inches=bbox, format=fmt, transparent=transparent
        )

        if not include_title:
            ax.set_title(original_title)
        for other_ax, vis in zip(all_axes, original_visibility, strict=True):
            other_ax.set_visible(vis)

        logger.info("Saved subplot to {}", path)
        return path
