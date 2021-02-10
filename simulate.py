import argparse
import numpy as np
import ray
from single_trial import run_single_trial
from simulation_helpers import *

parser = argparse.ArgumentParser(description="Run one simulation instance")
parser.add_argument("rates", type=str,
                    default="rates.txt", nargs="?",
                    help="Path to rates file")
parser.add_argument("simulation_params", type=str,
                    default="sim_params.txt", nargs="?",
                    help="Path to simulation params file")
parser.add_argument("-d", "--outputdir", help="Directory storing simulation output",
                    type=str, default="data")

if __name__ == "__main__":
    args = parser.parse_args()
    # parser.print_help()

    """ Load parameters: rates and max time / number of steps """
    synaptic_rates, pool_rates = load_rates_file(args.rates)

    sim_params = load_simparams_file(args.simulation_params)
    max_steps, max_time = sim_params["max_steps"], sim_params["max_time"]
    # final_syp_sizes = np.zeros(shape=(n_trials:=2500,), dtype=np.uint32)

    """ Run simulation trials """
    n_trials: int = 4
    ray.init(num_cpus=4)
    records = ray.get([run_single_trial.remote(trial_i, max_steps=max_steps,
                       pool_rates=pool_rates, synaptic_rates=synaptic_rates, output_dir=args.outputdir)
                       for trial_i in range(n_trials)])

    # for i, rec in enumerate(records):
    #     final_syp_sizes[i] = rec.synapse_occupancy[-1, 0]
    # np.save(os.path.join(args.outputdir, "final_sizes"), final_syp_sizes)
