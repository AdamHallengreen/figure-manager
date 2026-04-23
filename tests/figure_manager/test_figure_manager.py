# ruff: noqa: I001 — matplotlib.use("Agg") must precede pyplot imports
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from figure_manager.figure_manager import (
    CycleConfig,
    FigureManager,
    _build_prop_cycle,
    _build_rc_params,
    _get_figure_size,
    _load_palette,
    _load_style,
    _resolve_path,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def default_style():
    return _load_style("default")


@pytest.fixture
def deep_palette():
    return _load_palette("deep")


@pytest.fixture
def fm(tmp_path):
    return FigureManager(output_dir=tmp_path, use_latex=False)


# --- _resolve_path ---


def test_resolve_path_absolute(tmp_path):
    p = tmp_path / "out.pdf"
    assert _resolve_path(p, None) == p


def test_resolve_path_relative_with_output_dir(tmp_path):
    result = _resolve_path("fig.pdf", tmp_path)
    assert result == tmp_path / "fig.pdf"


def test_resolve_path_relative_no_output_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = _resolve_path("fig.pdf", None)
    assert result == tmp_path / "fig.pdf"


# --- _load_style ---


def test_load_style_default_string():
    style = _load_style("default")
    assert "rc" in style
    assert "layout" in style


def test_load_style_from_path(tmp_path):
    data = {"rc": {"font.size": 9}, "layout": {}}
    p = tmp_path / "custom.json"
    p.write_text(json.dumps(data))
    assert _load_style(p) == data


def test_load_style_dict_identity():
    d = {"rc": {}, "layout": {}}
    assert _load_style(d) is d


def test_load_style_missing_name():
    with pytest.raises(FileNotFoundError):
        _load_style("nonexistent_style_xyz")


# --- _load_palette ---


def test_load_palette_deep_string():
    palette = _load_palette("deep")
    assert "colors" in palette
    assert "linestyles" in palette
    assert "markers" in palette
    assert len(palette["colors"]) == 12
    assert len(palette["linestyles"]) == 12
    assert len(palette["markers"]) == 12


def test_load_palette_dict_identity():
    d = {"colors": ["red"]}
    assert _load_palette(d) is d


def test_load_palette_list_convenience():
    colors = ["#ff0000", "#00ff00"]
    result = _load_palette(colors)
    assert result == {"colors": colors}


def test_load_palette_from_path(tmp_path):
    data = {"colors": ["#aabbcc"], "linestyles": ["-"], "markers": ["o"]}
    p = tmp_path / "pal.json"
    p.write_text(json.dumps(data))
    assert _load_palette(p) == data


# --- _build_rc_params ---


def test_build_rc_params_use_latex_true(default_style):
    rc = _build_rc_params(default_style, use_latex=True)
    assert rc["text.usetex"] is True


def test_build_rc_params_use_latex_false(default_style):
    rc = _build_rc_params(default_style, use_latex=False)
    assert rc["text.usetex"] is False


def test_build_rc_params_missing_rc_key():
    rc = _build_rc_params({}, use_latex=False)
    assert rc == {"text.usetex": False}


# --- _build_prop_cycle ---


def test_build_prop_cycle_none(deep_palette):
    pc = _build_prop_cycle(deep_palette, None)
    keys = pc.keys
    assert keys == {"color"}


def test_build_prop_cycle_together_neither(deep_palette):
    pc = _build_prop_cycle(deep_palette, CycleConfig(mode="together"))
    assert pc.keys == {"color"}


def test_build_prop_cycle_together_linestyles(deep_palette):
    pc = _build_prop_cycle(
        deep_palette, CycleConfig(cycle_linestyles=True, mode="together")
    )
    assert "color" in pc.keys
    assert "linestyle" in pc.keys
    assert "marker" not in pc.keys


def test_build_prop_cycle_together_markers(deep_palette):
    pc = _build_prop_cycle(
        deep_palette, CycleConfig(cycle_markers=True, mode="together")
    )
    assert "color" in pc.keys
    assert "marker" in pc.keys
    assert "linestyle" not in pc.keys


def test_build_prop_cycle_together_both(deep_palette):
    pc = _build_prop_cycle(
        deep_palette,
        CycleConfig(cycle_linestyles=True, cycle_markers=True, mode="together"),
    )
    assert pc.keys == {"color", "linestyle", "marker"}
    assert len(list(pc)) == 12


def test_build_prop_cycle_after_colors_neither(deep_palette):
    pc = _build_prop_cycle(deep_palette, CycleConfig(mode="after_colors"))
    assert pc.keys == {"color"}


def test_build_prop_cycle_after_colors_linestyles(deep_palette):
    pc = _build_prop_cycle(
        deep_palette, CycleConfig(cycle_linestyles=True, mode="after_colors")
    )
    n_colors = len(deep_palette["colors"])
    n_ls = len(deep_palette["linestyles"])
    assert len(list(pc)) == n_ls * n_colors


def test_build_prop_cycle_after_colors_both(deep_palette):
    pc = _build_prop_cycle(
        deep_palette,
        CycleConfig(cycle_linestyles=True, cycle_markers=True, mode="after_colors"),
    )
    n_colors = len(deep_palette["colors"])
    n_ls = len(deep_palette["linestyles"])
    assert len(list(pc)) == n_ls * n_colors


# --- _get_figure_size ---


def test_get_figure_size_a4_portrait(default_style):
    w, h = _get_figure_size(default_style, "A4", 1, 1, landscape=False)
    assert w < 11  # within A4 width
    assert h > 0


def test_get_figure_size_a4_landscape_wider(default_style):
    w_p, h_p = _get_figure_size(default_style, "A4", 1, 1, landscape=False)
    w_l, _ = _get_figure_size(default_style, "A4", 1, 1, landscape=True)
    assert w_l > w_p


def test_get_figure_size_a3_wider_than_a4(default_style):
    w_a4, _ = _get_figure_size(default_style, "A4", 1, 1, landscape=False)
    w_a3, _ = _get_figure_size(default_style, "A3", 1, 1, landscape=False)
    assert w_a3 > w_a4


def test_get_figure_size_unknown_paper_falls_back_to_a4(default_style):
    w_a4, h_a4 = _get_figure_size(default_style, "A4", 1, 1, landscape=False)
    w_unk, h_unk = _get_figure_size(default_style, "B5", 1, 1, landscape=False)
    assert w_unk == w_a4
    assert h_unk == h_a4


# --- FigureManager.__init__ ---


def test_init_no_output_dir_no_warning(capfd):
    FigureManager(use_latex=False)
    assert "output_dir" not in capfd.readouterr().err


def test_init_with_output_dir_sets_attr(tmp_path):
    fm = FigureManager(output_dir=tmp_path, use_latex=False)
    assert fm.output_dir == tmp_path


def test_init_stores_attrs():
    fm = FigureManager(
        paper_size="A3", use_latex=False, style="default", palette="deep"
    )  # noqa: E501
    assert fm.paper_size == "A3"
    assert not fm.use_latex
    assert isinstance(fm._style, dict)
    assert isinstance(fm._palette, dict)
    assert isinstance(fm._rc_params, dict)


# --- create_figure ---


def test_create_figure_1x1(fm):
    fig, axes = fm.create_figure(1, 1)
    assert isinstance(fig, Figure)
    assert len(axes) == 1
    assert isinstance(axes[0], Axes)


def test_create_figure_default_n_subplots(fm):
    _, axes = fm.create_figure(2, 2)
    assert len(axes) == 4


def test_create_figure_partial_subplots(fm):
    fig, axes = fm.create_figure(2, 2, n_subplots=3)
    assert len(axes) == 3
    inactive = [ax for ax in fig.axes if not ax.axison]
    assert len(inactive) == 1


def test_create_figure_invalid_n_subplots(fm):
    with pytest.raises(ValueError):
        fm.create_figure(2, 2, n_subplots=5)


def test_create_figure_landscape_wider(fm):
    fig_p, _ = fm.create_figure(1, 1, landscape=False)
    w_p, h_p = fig_p.get_size_inches()
    fig_l, _ = fm.create_figure(1, 1, landscape=True)
    w_l, _ = fig_l.get_size_inches()
    assert w_l > w_p


def test_create_figure_with_cycle_config(fm):
    _, _ = fm.create_figure(1, 1, cycle=CycleConfig(cycle_linestyles=True))
    # Verify rc_context applied prop_cycle with linestyle during figure creation
    # (no global state change — just confirm no error and figure returned cleanly)


def test_create_figure_no_global_rc_pollution():
    original_usetex = mpl.rcParams["text.usetex"]
    fm = FigureManager(use_latex=False)
    fm.create_figure(1, 1)
    assert mpl.rcParams["text.usetex"] == original_usetex


# --- cycle presets ---


@pytest.mark.parametrize(
    "preset",
    [
        "linestyles",
        "markers",
        "full",
        "linestyles_sequential",
        "markers_sequential",
        "full_sequential",
    ],
)
def test_cycle_preset_string(fm, preset):
    fig, axes = fm.create_figure(1, 1, cycle=preset)
    assert len(axes) == 1


def test_cycle_preset_unknown_raises(fm):
    with pytest.raises(ValueError, match="Unknown cycle preset"):
        fm.create_figure(1, 1, cycle="bad")


# --- save_figure ---


def test_save_figure_creates_file(fm, tmp_path):
    fig, _ = fm.create_figure(1, 1)
    path = fm.save_figure(fig, tmp_path / "out.pdf")
    assert path.exists()
    assert path.suffix == ".pdf"


def test_save_figure_returns_path(fm, tmp_path):
    fig, _ = fm.create_figure(1, 1)
    result = fm.save_figure(fig, tmp_path / "out.pdf")
    assert isinstance(result, type(tmp_path))


def test_save_figure_no_extension_defaults_to_pdf(fm, tmp_path):
    fig, _ = fm.create_figure(1, 1)
    path = fm.save_figure(fig, tmp_path / "out")
    assert path.suffix == ".pdf"
    assert path.exists()


def test_save_figure_absolute_path_ignores_output_dir(fm, tmp_path):
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    fig, _ = fm.create_figure(1, 1)
    path = fm.save_figure(fig, other_dir / "out.pdf")
    assert path.parent == other_dir


def test_save_figure_save_subplots(fm, tmp_path):
    fig, _ = fm.create_figure(1, 2, n_subplots=2)
    fm.save_figure(fig, tmp_path / "fig.pdf", save_subplots=True)
    assert (tmp_path / "fig_subplot_1.pdf").exists()
    assert (tmp_path / "fig_subplot_2.pdf").exists()


def test_save_figure_custom_subplot_suffix(fm, tmp_path):
    fig, _ = fm.create_figure(1, 2, n_subplots=2)
    fm.save_figure(
        fig, tmp_path / "fig.pdf", save_subplots=True, subplot_suffix="_panel"
    )
    assert (tmp_path / "fig_panel_1.pdf").exists()
    assert (tmp_path / "fig_panel_2.pdf").exists()


# --- save_subplot ---


def test_save_subplot_creates_file(fm, tmp_path):
    fig, axes = fm.create_figure(1, 1)
    path = fm.save_subplot(fig, axes[0], tmp_path / "sub.pdf")
    assert path.exists()


def test_save_subplot_restores_title(fm, tmp_path):
    fig, axes = fm.create_figure(1, 1)
    axes[0].set_title("My Title")
    fm.save_subplot(fig, axes[0], tmp_path / "sub.pdf", include_title=False)
    assert axes[0].get_title() == "My Title"


def test_save_subplot_restores_axis_visibility(fm, tmp_path):
    fig, axes = fm.create_figure(1, 2, n_subplots=2)
    fm.save_subplot(fig, axes[0], tmp_path / "sub.pdf")
    assert all(ax.get_visible() for ax in fig.axes if ax.axison)
