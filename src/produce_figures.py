import argparse
import polars as pl
from polars import col as c
import yaml

from figure_manager import FigureManager
from figure_manager import generate_plot, generate_plot_with_error
from utils.find_project_root import find_project_root

def make_figures(EXTERNAL_DATA_PATH, FIGURES_DIR, PAPER_SIZE, FILE_EXT, USE_LATEX, VERBOSE):
    
    # Load data
    data = pl.read_csv(PROJECT_ROOT / EXTERNAL_DATA_PATH / 'Males.csv')
    
    # Make figure 1
    fm = FigureManager(output_dir=PROJECT_ROOT / FIGURES_DIR, paper_size=PAPER_SIZE, file_ext=FILE_EXT, use_latex=USE_LATEX)
    make_figure_1(fm, data, VERBOSE)
    
    # make figure 2
    fm = FigureManager(output_dir=PROJECT_ROOT / FIGURES_DIR, paper_size=PAPER_SIZE, file_ext=FILE_EXT, use_latex=USE_LATEX)
    make_figure_2(fm, data, VERBOSE)
    
def make_figure_1(fm, data, VERBOSE):
        # Create figure and subplot managers with specified layout
    fig, axes = fm.create_figure(n_rows=2, n_cols=2, n_subplots=3)

    axes[0] = generate_plot(data, x_col='school', y_col='wage', plot_type='plot', x_bins=5, group_by_cols='residence', agg_fct=pl.mean, ax = axes[0], verbose=VERBOSE, xlabel='School', ylabel='Wage', title='Average Wage by School and Residence')
    axes[1] = generate_plot(data, x_col='exper', y_col='wage', plot_type='plot', x_bins=5, group_by_cols='residence', agg_fct=pl.mean, ax = axes[1], verbose=VERBOSE, xlabel='Experience', ylabel='Wage', title='Average Wage by Experience and Residence')
    axes[2] = generate_plot(data, x_col='wage', bins=30, plot_type='hist', group_by_cols='maried', ax = axes[2], verbose=VERBOSE, plot_settings={'alpha': 0.5})

    # Save the entire figure and subplots
    fm.save_figure(filename="three_small_plots")
    
def make_figure_2(fm, data, VERBOSE):

    cd = collapsed_data = data.group_by('school').agg(
        pl.mean('wage').alias('wage_mean'),
        pl.std('wage').alias('wage_std'),
    ).with_columns(
        (c.wage_mean - c.wage_std*1.96).alias('wage_ci_low'),
        (c.wage_mean + c.wage_std*1.96).alias('wage_ci_high'),
    ).sort('school')

    # Create figure and subplot managers with specified layout
    fig, axes = fm.create_figure(n_rows=1, n_cols=2, n_subplots=2)

    axes[0] = generate_plot_with_error(x=cd.get_column("school"), y=cd.get_column("wage_mean"),   y_err=cd.get_column("wage_std"),   ax = axes[0], xlabel='School', ylabel='Wage', title='Average Wage by School and Residence', label='Wage')
    axes[0] = generate_plot_with_error(x=cd.get_column("school"), y=cd.get_column("wage_mean")*2, y_err=cd.get_column("wage_std"), ax = axes[0], xlabel='School', ylabel='Wage', title='Average Wage by School and Residence', label='Double Wage')
    axes[1] = generate_plot_with_error(x=cd.get_column("school"), y=cd.get_column("wage_mean"),   y_err=(cd.get_column("wage_ci_low"),   cd.get_column("wage_ci_high")),   ax = axes[1], xlabel='School', ylabel='Wage', title='Average Wage by School and Residence', label="Wage")
    axes[1] = generate_plot_with_error(x=cd.get_column("school"), y=cd.get_column("wage_mean")*2, y_err=(cd.get_column("wage_ci_low")*2, cd.get_column("wage_ci_high")*2), ax = axes[1], xlabel='School', ylabel='Wage', title='Average Wage by School and Residence', label="Double Wage")
    
    # Save the entire figure and subplots
    fm.save_figure(filename="two_std_dev_plots")
    
    
    
    
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--params', dest='params', required=True)
    args = args_parser.parse_args()

    with open(args.params) as param_file:
        params = yaml.safe_load(param_file)
        
    PROJECT_ROOT = find_project_root()
    EXTERNAL_DATA_PATH: str = params['data_etl']['external_data_path']
    FIGURES_DIR: str = params['figure_manager']['figures_dir']
    PAPER_SIZE: str = params['figure_manager']['paper_size']
    FILE_EXT: str = params['figure_manager']['file_ext']
    USE_LATEX: bool = params['figure_manager']['use_latex']
    VERBOSE: bool = params['plotter']['verbose']
    
    # output 4 random key value pairs to a metrics.json file
    # to be used by the CI/CD pipeline
    metrics = {k: v for k, v in params['plotter'].items() if k in ['verbose', 'figures_dir', 'paper_size', 'file_ext']}
    metrics['external_data_path'] = EXTERNAL_DATA_PATH
    metrics['use_latex'] = USE_LATEX
    with open(PROJECT_ROOT / 'metrics.json', 'w') as metrics_file:
        yaml.dump(metrics, metrics_file)
    # Call the make_figures function to generate the figures
    
    make_figures(EXTERNAL_DATA_PATH, FIGURES_DIR, PAPER_SIZE, FILE_EXT, USE_LATEX, VERBOSE)