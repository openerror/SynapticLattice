import dill
from typing import *
from astropy.visualization import hist as astrohist
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from src.RawRecord import RawRecord


def set_matplotlib_fontsize(DPI=100, SMALL_SIZE=17, MEDIUM_SIZE=19, BIGGER_SIZE=23):
    '''
        Convenience function for setting font sizes in figures
    '''
    plt.rc("text", usetex=True)
    plt.rc("figure", dpi=DPI)
    plt.rc("font", family="Times New Roman")
    plt.rc("mathtext", fontset="dejavuserif")
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def interp_synapse_occupancy(record: RawRecord) -> Callable:
    """ Create interpolator objects """
    return interp1d(record.times, record.synapse_occupancy[:, 0], kind="previous")


def plot_histogram(arr: List[int], ax, **hist_kwargs):
    astrohist(arr, bins="knuth", ax=ax, density=True, **hist_kwargs)
    return ax


def read_all_records(file_paths: List[str]) -> Iterable[RawRecord]:
    """ Iterator returning each RawRecord """
    for fp in file_paths:
        with open(fp, "rb") as f:
            yield dill.load(f)


def gather_sizes(interpolators: List[Callable], t: float) -> Iterable[int]:
    """ Obtain interpolated occupancies at independent variable t """
    for intp in interpolators:
        try:
            yield int(intp(t))
        except ValueError:
            yield -1  # an "illegal" value


def get_fig_ax(nrows: int = 1, ncols: int = 1, **kwargs):
    assert nrows >= 1
    assert ncols >= 1
    fig, axes = plt.subplots(nrows, ncols, **kwargs,
                             tight_layout=True, figsize=(8, 3.5*nrows))  # width, height
    return fig, axes