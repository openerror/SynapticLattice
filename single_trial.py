import dill, os.path
import ray
from src.MoleculePool import MoleculePool
from src.RawRecord import RawRecord
from src.Simulation import Simulation
from src.Synapse import Synapse
from src.SypInit import SynapseInitializer


@ray.remote
def run_single_trial(max_steps: int, trial_id: int, *args, **kwargs):
    """
    Modify returned objects to collect different kinds of data
    """
    pool_rates = kwargs.get("pool_rates")
    synaptic_rates = kwargs.get("synaptic_rates")
    output_dir = kwargs.get("output_dir")

    # Create synapses and pool
    # pool = MoleculePool(max_n=2500, init_n=1250, **pool_rates)
    synapses = [Synapse(side_length=50, pool_instance=None, **synaptic_rates)]

    # Create Simulation instance
    sim = Simulation(synapses=synapses, m_pool=None)
    record = RawRecord(max_steps, len(synapses), None, None,  #pool.max_n, pool.n,
                       [syp.side_length for syp in sim.synapses],
                       [syp.s.sum() for syp in sim.synapses],
                       **synaptic_rates, **pool_rates)

    # Initialize synaptic occupancies
    syp_init = SynapseInitializer()
    for syp in synapses:
        syp_init.random_init(syp, fill_frac=0.2)

    # Run the simulation
    for ns in range(max_steps):
        _ = sim.single_step()
        record.synapse_occupancy[ns, :] = sim.get_synapse_sizes()
        record.times[ns] = sim.T
        if sim.m_pool:
            record.pool_occupancy[ns] = sim.m_pool.n

    # Clean up, store/return results
    if trial_id % 500 == 0:
        print(f"Trial {trial_id} completed")
        with open(os.path.join(output_dir, f"rec-{trial_id}.dill"), "wb") as f:
            dill.dump(record, f)

    return record
