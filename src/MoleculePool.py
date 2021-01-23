import numpy as np
from src.locate_first_geq import locate_first_geq


class MoleculePool:
    def __init__(self, delta: float, gamma: float,
                 max_n: int = np.inf, init_n: int = 0, random_state=None):
        assert init_n >= 0, "ERROR: specified a negative initial amount"
        assert max_n >= init_n, "ERROR: specified maximal amt smaller than initial amt"

        self.delta, self.gamma = delta, gamma
        self.max_n: int = max_n
        self.__n: int = init_n
        self.rng = np.random.default_rng(random_state)
        self.total_p: float = 0.
        self.p: np.ndarray = np.array([0., 0.])
        self.update_p()

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, new_n):
        assert new_n >= 0, "ERROR: encountered negative pool occupancy"
        assert new_n <= self.max_n, "ERROR: pool occupancy larger than max_n"
        self.__n = new_n

    def update_p(self):
        # self.p[0], self.p[1] = self.gamma, self.delta * self.get_filled_frac()
        self.p[0], self.p[1] = self.gamma, self.delta * self.__n
        self.total_p = self.p.sum()

    def trigger(self) -> int:
        r = self.rng.uniform(low=0., high=self.total_p, size=(1,))
        event_ix = locate_first_geq(self.p.cumsum(), r)
        assert event_ix != -1, "ERROR: failed to select any pool event"
        if event_ix == 0:
            self.n += 1
        elif event_ix == 1:
            self.n -= 1
        self.update_p()
        return event_ix

    def get_filled_frac(self) -> float:
        return self.__n / self.max_n
