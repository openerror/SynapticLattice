import dill, os.path
import ray
from src.MoleculePool import MoleculePool
from src.RawRecord import RawRecord
from src.Simulation import Simulation
from src.Synapse import Synapse
from src.SypInit import SynapseInitializer


@ray.remote
def run_single_trial(trial_id: int, max_steps: int = None, *args, **kwargs):
    """
    Modify returned objects to collect different kinds of data
    """
    pool_rates = kwargs.get("pool_rates")
    synaptic_rates = kwargs.get("synaptic_rates")
    output_dir = kwargs.get("output_dir")

    # Create synapses and pool
    # pool = MoleculePool(max_n=2500, init_n=2500, **pool_rates)
    # synapses = [Synapse(side_length=50, pool_instance=pool, **synaptic_rates)]
    synapses = [Synapse(side_length=50, pool_instance=None, **synaptic_rates)]

    # Create Simulation instance
    # sim = Simulation(synapses=synapses, m_pool=pool)
    # record = RawRecord(max_steps, len(synapses), pool.max_n, pool.n,
    #                    [syp.side_length for syp in sim.synapses],
    #                    [syp.s.sum() for syp in sim.synapses],
    #                    **synaptic_rates, **pool_rates)
    sim = Simulation(synapses=synapses, m_pool=None)
    record = RawRecord(max_steps, len(synapses), None, None,
                       [syp.side_length for syp in sim.synapses],
                       [syp.s.sum() for syp in sim.synapses],
                       **synaptic_rates, **pool_rates)

    # Initialize synaptic occupancies
    syp_init = SynapseInitializer()
    for syp in synapses:
        syp_init.empty_init(syp)

    # Run the simulation: warm start, and then record
    for nw in range(max(2000, max_steps//3)):
        _ = sim.single_step()

    for ns in range(max_steps):
        _ = sim.single_step()
        record.synapse_occupancy[ns, :] = sim.get_synapse_sizes()
        record.times[ns] = sim.T
        if sim.m_pool:
            record.pool_occupancy[ns] = sim.m_pool.n

    # Clean up, store/return results
    with open(os.path.join(output_dir, f"rec-{trial_id}.dill"), "wb") as f:
        dill.dump(record, f)
    if trial_id % 50 == 0:
        print(f"Trial {trial_id} completed")
    return trial_id
