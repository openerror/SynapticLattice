import glob
from typing import *
import numpy as np
from scipy import stats
import seaborn as sns
from src.RawRecord import RawRecord
from src.analysis.common import *


def old_new_scatter(recs: List[RawRecord], times_of_interest: List[int], num_traces: int = 4):
    set_matplotlib_fontsize(DPI=100, SMALL_SIZE=12, MEDIUM_SIZE=14, BIGGER_SIZE=16)
    fig, axes = get_fig_ax(nrows=3, ncols=2)
    interpolated_sypo = [interp_synapse_occupancy(r) for r in recs]

    for ax, label in zip(axes.flatten(), ("(A)", "(B)", "(C)", "(D)", "(E)", "(F)")):
        ax.text(-0.13, 1.05, label, transform=ax.transAxes)

    """ A: sample trajectories """
    sorted_recs = sorted(recs, key=lambda r: -r.times[-1])
    for i in range(num_traces):
        axes[0, 0].plot(recs[i].times, recs[i].synapse_occupancy[:, 0])
    axes[0, 0].set_ylabel("Synaptic size")
    axes[0, 0].set_xlabel("Time (arbitrary units)")

    """ B: size change v.s. old size """
    sizes = [gather_sizes(interpolated_sypo, t)
             for t in times_of_interest]

    delta_sizes, old_sizes_for_plot = [], []
    for i, (os, ns) in enumerate(zip(sizes[0], sizes[-1])):
        if (os != -1) and (ns != -1):
            delta_sizes.append(ns - os)
            old_sizes_for_plot.append(os)

    slope, intercept, r_value, pv, se = stats.linregress(old_sizes_for_plot, delta_sizes)
    sns.regplot(x=old_sizes_for_plot, y=delta_sizes,
                ax=axes[0, 1], line_kws={"color": "red", "ls": "-"}, ci=None,
                label=f"y={slope:.2f}x+{intercept:.1f}\n$R^2 = {r_value ** 2:.2}$")
    axes[0, 1].set_xlabel("Synaptic size")
    axes[0, 1].set_ylabel("$\Delta$ Synaptic size")
    axes[0, 1].legend(frameon=False)
    print(len(old_sizes_for_plot))

    """ C: near new size v.s. old size """
    sizes = [gather_sizes(interpolated_sypo, times_of_interest[0]),
             gather_sizes(interpolated_sypo, times_of_interest[1])]

    new_sizes_for_plot, old_sizes_for_plot = [], []
    for i, (os, ns) in enumerate(zip(sizes[0], sizes[1])):
        if (os != -1) and (ns != -1):
            new_sizes_for_plot.append(ns)
            old_sizes_for_plot.append(os)

    slope, intercept, r_value, pv, se = stats.linregress(old_sizes_for_plot, new_sizes_for_plot)
    sns.regplot(x=old_sizes_for_plot, y=new_sizes_for_plot,
                ci=None, ax=axes[1, 0], line_kws={"color": "red"},
                label=f"y={slope:.2f}x+{intercept:.1f}\n$R^2 = {r_value ** 2:.2}$")

    delta_t = times_of_interest[1] - times_of_interest[0]
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("Synaptic size")
    axes[1, 0].set_ylabel(f"Synaptic size after $\Delta t = {delta_t:.1f}$")
    print(len(old_sizes_for_plot))

    """ D: near new size v.s. old size """
    sizes = [gather_sizes(interpolated_sypo, times_of_interest[0]),
             gather_sizes(interpolated_sypo, times_of_interest[-1])]

    new_sizes_for_plot, old_sizes_for_plot = [], []
    for i, (os, ns) in enumerate(zip(sizes[0], sizes[1])):
        if (os != -1) and (ns != -1):
            new_sizes_for_plot.append(ns)
            old_sizes_for_plot.append(os)

    slope, intercept, r_value, pv, se = stats.linregress(old_sizes_for_plot, new_sizes_for_plot)
    sns.regplot(x=old_sizes_for_plot, y=new_sizes_for_plot,
                ci=None, ax=axes[1, 1], line_kws={"color": "red"},
                label=f"y={slope:.2f}x+{intercept:.1f}\n$R^2 = {r_value ** 2:.2}$")

    delta_t = times_of_interest[-1] - times_of_interest[0]
    axes[1, 1].legend(frameon=False)
    axes[1, 1].set_xlabel("Synaptic size")
    axes[1, 1].set_ylabel(f"Synaptic size after $\Delta t = {delta_t:.1f}$")
    print(len(old_sizes_for_plot))

    """ D, E: slope, R^2 with time """
    slopes, R2, intercepts = [], [], []

    for i in range(0, len(times_of_interest)):
        sizes = [gather_sizes(interpolated_sypo, times_of_interest[0]),
                 gather_sizes(interpolated_sypo, times_of_interest[i])]

        new_sizes, old_sizes = [], []
        for i, (os, ns) in enumerate(zip(sizes[0], sizes[1])):
            if (os != -1) and (ns != -1):
                new_sizes.append(ns)
                old_sizes.append(os)

        slope, intercept, r_value, _, _ = stats.linregress(old_sizes, new_sizes)
        slopes.append(slope)
        R2.append(r_value ** 2)
        intercepts.append(intercept)

    times_elapsed = [t - times_of_interest[0] for t in times_of_interest]
    axes[2, 0].plot(times_elapsed, slopes, "-o", color="blue")
    axes[2, 0].set_ylabel("Slope", color="b")

    axes[2, 1].plot(times_elapsed, R2, "-o")
    axes[2, 1].set_ylabel("$R^2$")
    for ax in axes[2]:
        ax.set_xlabel(f"Time since $t = {times_of_interest[0]:.1f}$ (arbitrary units)")

    intercept_ax = axes[2, 0].twinx()
    intercept_ax.plot(times_elapsed, intercepts, "-o", color="orange")
    intercept_ax.set_ylabel("Intercept", color="orange")
    return fig, axes


if __name__ == "__main__":
    recs = list(read_all_records(glob.glob("data/sypdist/1/*.dill")))
    fig, axes = old_new_scatter(recs, times_of_interest = np.linspace(120, 170, 7))
    fig.savefig("img/shomar_size_dynamics.png")
