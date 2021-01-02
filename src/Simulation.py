from typing import *
import numpy as np
from src.locate_first_geq import locate_first_geq
from src.Synapse import Synapse
from src.MoleculePool import MoleculePool


class Simulation:
    """ One KMC realization assuming one set of rates """

    def __init__(self, synapses: List[Synapse], m_pool: MoleculePool,
                 init_T: float = 0., dt_random_state: int = None,
                 event_random_state: int = None, **kwargs):
        self.T = init_T
        self.synapses = synapses
        self.m_pool = m_pool
        self.dt_rng = np.random.default_rng(dt_random_state)
        self.event_rng = np.random.default_rng(event_random_state)

    def advance_time(self) -> float:
        total_p = sum((sp.total_p for sp in self.synapses))
        total_p = (total_p + self.m_pool.total_p) if self.m_pool else total_p
        self.T += (dt := self.dt_rng.exponential(scale=1./total_p))
        return dt

    def select_entity(self) -> Tuple[float, int]:
        cum_p = [sp.total_p for sp in self.synapses]
        if self.m_pool: cum_p += [self.m_pool.total_p]
        cum_p = np.cumsum(cum_p)
        # cum_p = np.cumsum([sp.total_p for sp in self.synapses] + [self.m_pool.total_p])
        selected_entity = locate_first_geq(cum_p, r := self.event_rng.uniform(low=0., high=cum_p[-1]))
        return r, selected_entity

    def single_step(self) -> int:
        """
        Put together other built-in functions to carry out one step of the
        simuatlion. Hard to unit test as a whole; test its individual steps
        instead.

        Record system state OUTSIDE of this function!
        """
        self.advance_time()
        _, entity = self.select_entity()

        if entity == len(self.synapses):  # pool selected
            self.m_pool.trigger()
            for syp in self.synapses: syp.update_all_p()
        else:
            self.synapses[entity].trigger()  # synapse selected
            if self.m_pool:
                self.m_pool.update_p()
                for si, syp in enumerate(self.synapses):
                    if si != entity: syp.update_all_p()
        return entity  # for debug purposes

    def get_synapse_sizes(self) -> List[int]:
        return [syp.s.sum() for syp in self.synapses]

    def print_system_state(self):
        print(f"T {self.T:.3f}; pool occupancy {self.m_pool.n}; synapse occupancy {self.synapses[0].s.sum()}")
