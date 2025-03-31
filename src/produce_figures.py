# type: ignore
import argparse
import json
from pathlib import Path

import polars as pl
import yaml
from polars import col as c

from figure_manager import FigureManager, generate_plot
from utils.find_project_root import find_project_root


def make_figures(
    EXTERNAL_DATA_PATH, FIGURES_DIR, PAPER_SIZE, FILE_EXT, USE_LATEX, VERBOSE
):
    # Load data
    data = pl.read_csv(PROJECT_ROOT / EXTERNAL_DATA_PATH / "Males.csv")

    # Make figure 1
    fm = FigureManager(
        output_dir=PROJECT_ROOT / FIGURES_DIR,
        paper_size=PAPER_SIZE,
        file_ext=FILE_EXT,
        use_latex=USE_LATEX,
    )
    make_figure_1(fm, data, VERBOSE)

    # make figure 2
    fm = FigureManager(
        output_dir=PROJECT_ROOT / FIGURES_DIR,
        paper_size=PAPER_SIZE,
        file_ext=FILE_EXT,
        use_latex=USE_LATEX,
    )
    make_figure_2(fm, data)

    # make figure 3
    fm = FigureManager(
        output_dir=PROJECT_ROOT / FIGURES_DIR,
        paper_size=PAPER_SIZE,
        file_ext=FILE_EXT,
        use_latex=USE_LATEX,
    )
    make_figure_3(fm, data, VERBOSE)


def make_figure_1(fm, data, VERBOSE):
    # Create figure and subplot managers with specified layout
    fig, axes = fm.create_figure(n_rows=2, n_cols=2, n_subplots=3)

    axes[0] = generate_plot(
        data,
        x="school",
        y="wage",
        plot_type="plot",
        x_bins=5,
        group_by="residence",
        agg_fct=pl.mean,
        ax=axes[0],
        verbose=VERBOSE,
        xlabel="School",
        ylabel="Wage",
        title="Average Wage by School and Residence",
    )
    axes[1] = generate_plot(
        data,
        x="exper",
        y="wage",
        plot_type="plot",
        x_bins=5,
        group_by="residence",
        agg_fct=pl.mean,
        ax=axes[1],
        verbose=VERBOSE,
        xlabel="Experience",
        ylabel="Wage",
        title="Average Wage by Experience and Residence",
    )
    axes[2] = generate_plot(
        data,
        x="wage",
        bins=30,
        plot_type="hist",
        group_by="maried",
        ax=axes[2],
        verbose=VERBOSE,
        plot_settings={"alpha": 0.5},
    )

    # Save the entire figure and subplots
    fm.save_figure(filename="three_small_plots")


def make_figure_2(fm, data):
    cd = (
        data.group_by("school")
        .agg(
            pl.mean("wage").alias("wage_mean"),
            pl.std("wage").alias("wage_std"),
            pl.mean("exper").alias("exper_mean"),
            pl.std("exper").alias("exper_std"),
        )
        .with_columns(
            (c.wage_mean - c.wage_std * 1.96).alias("wage_ci_low"),
            (c.wage_mean + c.wage_std * 1.96).alias("wage_ci_high"),
            (c.exper_mean - c.exper_std * 1.96).alias("exper_ci_low"),
            (c.exper_mean + c.exper_std * 1.96).alias("exper_ci_high"),
        )
        .sort("school")
    )

    # Create figure and subplot managers with specified layout
    fig, axes = fm.create_figure(n_rows=1, n_cols=2, n_subplots=2)

    axes[0] = generate_plot(
        cd,
        x="school",
        y="wage_mean",
        y_err="wage_std",
        ax=axes[0],
        xlabel="School years",
        ylabel="Wage",
        title="Average Wage by School and Residence",
        label="Wage",
    )
    axes[0] = generate_plot(
        cd,
        x="school",
        y="exper_mean",
        y_err="exper_std",
        ax=axes[0],
        xlabel="School years",
        ylabel="Experience",
        title="Average Experience by School and Residence",
        label="Experience",
    )
    axes[1] = generate_plot(
        cd,
        x="school",
        y="wage_mean",
        y_err=("wage_ci_low", "wage_ci_high"),
        ax=axes[1],
        xlabel="School years",
        ylabel="Wage",
        title="Average Wage by School and Residence",
        label="Wage",
    )
    axes[1] = generate_plot(
        cd,
        x="school",
        y="exper_mean",
        y_err=("exper_ci_low", "exper_ci_high"),
        ax=axes[1],
        xlabel="School years",
        ylabel="Experience",
        title="Average Experience by School and Residence",
        label="Experience",
    )

    # Save the entire figure and subplots
    fm.save_figure(filename="two_std_dev_plots")


def make_figure_3(fm, data, VERBOSE):
    # Create figure and subplot managers with specified layout
    fig, axes = fm.create_figure(n_rows=1, n_cols=1, n_subplots=1)

    axes[0] = generate_plot(
        data,
        x="school",
        y="wage",
        plot_type="plot",
        x_bins=5,
        group_by="residence",
        agg_fct=pl.mean,
        ax=axes[0],
        verbose=VERBOSE,
        xlabel="School",
        ylabel="Wage",
    )

    # Save the entire figure and subplots
    fm.save_figure(filename="one_big_plot")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()

    with open(args.params) as param_file:
        params = yaml.safe_load(param_file)

    PROJECT_ROOT = find_project_root(__file__)
    EXTERNAL_DATA_PATH = Path(params["data_etl"]["external_data_path"])
    FIGURES_DIR = Path(params["figure_manager"]["figures_dir"])
    PAPER_SIZE: str = params["figure_manager"]["paper_size"]
    FILE_EXT: str = params["figure_manager"]["file_ext"]
    USE_LATEX: bool = params["figure_manager"]["use_latex"]
    VERBOSE: bool = params["plotter"]["verbose"]

    # output 4 random key value pairs to a metrics.json file
    # to be used by the CI/CD pipeline
    metrics = {
        k: v
        for k, v in params["plotter"].items()
        if k in ["verbose", "figures_dir", "paper_size", "file_ext"]
    }
    metrics["external_data_path"] = str(EXTERNAL_DATA_PATH)
    metrics["use_latex"] = USE_LATEX
    with open(PROJECT_ROOT / "metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
    # Call the make_figures function to generate the figures

    make_figures(
        EXTERNAL_DATA_PATH, FIGURES_DIR, PAPER_SIZE, FILE_EXT, USE_LATEX, VERBOSE
    )
