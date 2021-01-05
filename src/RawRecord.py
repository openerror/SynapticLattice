from dataclasses import dataclass
from typing import *
from numpy import zeros, float32, uint16


@dataclass
class RawRecord:
    """
        dataclass for one simulation instance
    """
    n_steps: int  # number of simulation steps
    n_synapses: int  # number of synapses in the simulation
    max_pool_n: int  # max pool occupancy allowed
    init_pool_n: int  # initial pool occupancy
    syp_side_length: List[int]  # linear dimension of square synapse
    syp_init_n: List[int]  # initial synaptic occupancy

    alpha: float  # rates used in the simulation
    beta: float
    lambda_on: float
    lambda_off: float
    gamma: float
    delta: float

    dt_random_state: int = None  # seeds of (some) random genereators
    event_random_state: int = None
    syp_init_random_state: int = None

    def __post_init__(self):
        self.times = zeros(shape=(self.n_steps,), dtype=float32)
        self.synapse_occupancy = zeros(shape=(self.n_steps, self.n_synapses), dtype=uint16)
        if self.max_pool_n and self.init_pool_n:
            # Only track pool occupancy when there is a pool
            self.pool_occupancy = zeros(shape=(self.n_steps,), dtype=uint16)
        else:
            self.pool_occupancy = None