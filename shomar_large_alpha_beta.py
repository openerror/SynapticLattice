import os.path, sys
from glob import glob
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from src.analysis.common import *
# plt.style.use("seaborn-white")


if __name__ == "__main__":
    DATA_ROOT = sys.argv[1]
    assert os.path.isdir(DATA_ROOT), "Please indicate correct data directory"
    set_matplotlib_fontsize(SMALL_SIZE=15, MEDIUM_SIZE=17, BIGGER_SIZE=20)
    fig, ax = get_fig_ax(nrows=1, ncols=1)

    get_recs_cmd = "read_all_records(glob(os.path.join(DATA_ROOT, 'LargeAlphaBeta', '*.dill')))"
    interpolators = list(map(interp_synapse_occupancy, eval(get_recs_cmd)))
    sizes = list(filter(lambda s: s > 0,
                        gather_sizes(interpolators, t=min((r.times[-1] for r in eval(get_recs_cmd))))))
    plot_histogram(sizes, ax, alpha=0.7,
                   label="$\\alpha, \\beta \\approx \lambda_{on}, \lambda_{off}$")

    ax.text(0.65, 0.75, f"normaltest p-val = {normaltest(sizes)[1]:.3f}", transform=ax.transAxes)
    ax.legend(frameon=False)
    ax.set_xlabel("Synaptic size")
    ax.set_ylabel("PDF")

    fig.savefig("img/shomar_large_alpha_beta.png")