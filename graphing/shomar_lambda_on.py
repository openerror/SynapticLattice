import os.path, sys
from glob import glob
import matplotlib.pyplot as plt
from src.analysis.common import *
# plt.style.use("seaborn-white")


if __name__ == "__main__":
    DATA_ROOT = sys.argv[1]
    assert os.path.isdir(DATA_ROOT), "Please indicate correct data directory"

    """ 
    Image layout
        LEFT: Similar/Identifical size distribution shape throughout time
        RIGHT: Switching between left-skewed and right-skewed distributions as lambda_on changes
    """
    set_matplotlib_fontsize(SMALL_SIZE=15, MEDIUM_SIZE=17, BIGGER_SIZE=20)
    fig, axes = get_fig_ax(nrows=1, ncols=2)
    colormap = {"0.9": "red", "1": "blue", "1.1": "orange", "1.2": "green", "1.5": "m"}

    """ LEFT & RIGHT """
    for lambda_on in os.listdir(DATA_ROOT):
        # print(len(glob(os.path.join(DATA_ROOT, lambda_on, "*.dill"))))
        if lambda_on in ("0.9", "1", "1.1", "1.2", "1.5"):
            print(lambda_on)
            get_recs_cmd = "read_all_records(glob(os.path.join(DATA_ROOT, lambda_on, '*.dill')))"
            interpolators = list(map(interp_synapse_occupancy, eval(get_recs_cmd)))

            sizes = list(filter(lambda s: s > 0,
                                gather_sizes(interpolators, t=min((r.times[-1] for r in eval(get_recs_cmd))))))

            if lambda_on in ("0.9", "1"):
                axes[0].set_xlim([-10, 450])
                plot_histogram(sizes, ax=axes[0], color=colormap[lambda_on],
                               label="$\lambda_{on}$ = " + lambda_on, alpha=0.5)
            else:
                #axes[1].set_xlim([1800, 2400])
                plot_histogram(sizes, ax=axes[1], color=colormap[lambda_on],
                               label="$\lambda_{on}$ = " + lambda_on, alpha=0.5)

    for ax in axes:
        ax.legend(frameon=False)
        ax.set_xlabel("Synaptic size")
        ax.set_ylabel("PDF")

    fig.savefig("img/shomar_lambda_on.png")