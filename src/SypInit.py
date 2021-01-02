import numpy as np
from src.Synapse import Synapse


class SynapseInitializer:
    def __init__(self, random_state: int = None):
        self.rng = np.random.default_rng(random_state)

    def helper(self, syp: Synapse) -> Synapse:
        """ Carries out repeated steps after syp.s is initialized """
        syp.ss[1:-1, 1:-1] = syp.s
        for site_ix in range(syp.s.size):
            syp.update_site_p(site_ix)
        syp.total_p = syp.p.sum()
        return syp

    def empty_init(self, syp: Synapse) -> Synapse:
        """ Initialize propensities but do not insert any molecules """
        return self.helper(syp)

    def full_init(self, syp: Synapse) -> Synapse:
        """ Fills up every lattice site """
        syp.s[:] = 1  # np.ones_like(syp.s)
        return self.helper(syp)

    def random_init(self, syp: Synapse, fill_frac: float = 0.5) -> Synapse:
        """ Fills up each lattice site randomly and independently """
        syp.s = self.rng.uniform(size=syp.s.shape)
        syp.s = np.where(syp.s <= fill_frac, 1, 0)
        return self.helper(syp)
