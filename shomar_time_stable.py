import glob
import numpy as np
import seaborn as sns
from src.analysis.common import *


if __name__ == "__main__":
    two_lambdas = ("1", "1.2")
    set_matplotlib_fontsize(DPI=100, SMALL_SIZE=13, MEDIUM_SIZE=15, BIGGER_SIZE=17)
    fig, axes = get_fig_ax(nrows=2, ncols=1)

    """ Top plot """
    recs = list(read_all_records(glob.glob(f"data/sypdist/{two_lambdas[0]}/*.dill")))
    interpolated_sypo = [interp_synapse_occupancy(r) for r in recs]
    sizes = list(filter(lambda s: s > 0, gather_sizes(interpolated_sypo, t=130)))
    sns.kdeplot(sizes, ax=axes[0], alpha=0.6, color="red", label=f"$t = 130, n = {len(sizes)}$")

    sizes = list(filter(lambda s: s > 0, gather_sizes(interpolated_sypo, t=150)))
    sns.kdeplot(sizes, ax=axes[0], alpha=0.6, color="green", label=f"$t = 150, n = {len(sizes)}$")

    sizes = list(filter(lambda s: s > 0, gather_sizes(interpolated_sypo, t=170)))
    sns.kdeplot(sizes, ax=axes[0], alpha=0.6, color="blue", label=f"$t = 170, n = {len(sizes)}$")

    """ Bottom plot """
    recs = list(read_all_records(glob.glob(f"data/sypdist/{two_lambdas[1]}/*.dill")))
    print(min((r.times[-1] for r in recs)), max((r.times[0] for r in recs)))
    interpolated_sypo = [interp_synapse_occupancy(r) for r in recs]
    sizes = list(filter(lambda s: s > 0, gather_sizes(interpolated_sypo, t=110)))
    sns.kdeplot(sizes, ax=axes[1], label=f"$t = 110, n = {len(sizes)}$", color="red")

    sizes = list(filter(lambda s: s > 0, gather_sizes(interpolated_sypo, t=130)))
    sns.kdeplot(sizes, ax=axes[1], label=f"$t = 130, n = {len(sizes)}$", color="green")

    sizes = list(filter(lambda s: s > 0, gather_sizes(interpolated_sypo, t=150)))
    sns.kdeplot(sizes, ax=axes[1], label=f"$t = 150, n = {len(sizes)}$", color="blue")

    for ax, lamb in zip(axes, two_lambdas):
        ax.set_title("$\lambda_{on} = $" + lamb)
        ax.set_xlabel("Synaptic size")
        ax.set_ylabel("Fitted kernel density")
        ax.legend(frameon=False)

    fig.savefig("img/shomar_time_stable.png")