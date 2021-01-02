from typing import *
from itertools import product
import warnings
import numpy as np
from src.im2col import convolute
from src.locate_first_geq import locate_first_geq


class Synapse:
    def __init__(self, side_length: int, alpha: float,
                 beta: float, lambda_on: float, lambda_off: float,
                 pool_instance=None, pool_factor_func: Callable = None,
                 random_state=None):
        self.s = np.zeros(shape=(side_length, side_length), dtype=np.int8)
        self.ss = np.zeros(shape=(side_length + 2, side_length + 2), dtype=np.int8)
        self.neighbor_mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=self.ss.dtype)
        self.p = np.zeros(shape=(side_length * side_length, 2), dtype=np.float64)
        self.total_p: float = 0.
        self.alpha, self.lambda_on, self.lambda_off = alpha, lambda_on, lambda_off
        self.side_length: int = side_length
        self.pool_instance = pool_instance
        self.pool_factor_func = pool_factor_func  # function postprocessing the raw filled fraction
        self.rng = np.random.default_rng(random_state)

    def trigger(self) -> int:
        """
            Trigger addition/removal of molecule to/from selected site
        """
        y, x = self._scalar_index_to_xy(site := self._select_site())
        if self.pool_instance:
            self.pool_instance.n += (1 if self.s[y, x] == 1 else -1)
        self.s[y, x] = (self.s[y, x] + 1) % 2
        self.ss[1:-1, 1:-1] = self.s

        if not self.pool_instance:
            # no pool, sufficient to only update neighbors
            for neighbor in self.get_neighbors(y, x):
                self.update_site_p(neighbor)
            self.update_site_p(site)
            self.update_total_p()
        else:
            # remember to update all other synapses
            # handle in external control flow
            self.update_all_p()

        return site

    def _select_site(self) -> int:
        """
            Sample one site where an event will happen
        """
        cp = np.cumsum(self.p.sum(axis=1))
        if np.all(cp[0] == cp) and np.isclose(cp[-1], 0.):
            warnings.warn("propensities are all identical \
            and sum to 0; absorption state reached?", category=RuntimeWarning)
        r = self.rng.uniform(low=0., high=cp[-1], size=1)
        selected_site_ix = locate_first_geq(cp, r)

        try:
            assert selected_site_ix != -1
        except AssertionError:
            print(r, cp)
            raise ValueError("ERROR: failed to select any site")

        # assert selected_site_ix != -1, "ERROR: failed to select any sites"
        return selected_site_ix

    def get_neighbors(self, y: int, x: int) -> Iterable[int]:
        for ny, nx in product((y-1, y, y+1), (x-1, x, x+1)):
            if ((ny, nx) != (y, x)) and (0 <= ny < self.side_length) and (0 <= nx < self.side_length):
                yield self._yx_to_scalar_index(ny, nx)

    def update_site_p(self, site_index: int, update_total: bool = False):
        """
            Update propensity at site_index (CRUX OF THE MODEL!)
        :param site_index: scalar referring to site to be updated
        """
        # fraction of filled neighbors
        v = self._count_occupied_neighbors(site_index) / 8.

        # pool factor [0...1]
        pf = self._pool_factor()

        # update propensity
        y, x = self._scalar_index_to_xy(site_index)
        assert self.s[y, x] in (0, 1), "ERROR: illegal occupancy value encountered"
        self.p[site_index] = 0.

        if self.s[y, x] == 1:  # removal to pool
            self.p[site_index, 1] = self.lambda_off * (1. - v)
            if self.pool_instance:
                self.p[site_index, 1] *= (self.pool_instance.max_n - self.pool_instance.n)
        elif self.s[y, x] == 0:  # uptake from pool
            self.p[site_index, 0] = self.alpha + self.lambda_on * v
            if self.pool_instance: self.p[site_index, 0] *= self.pool_instance.n

        if update_total:
            self.update_total_p()

    def update_total_p(self):
        self.total_p = self.p.sum()

    def update_all_p(self):
        v = convolute(self.ss, self.neighbor_mask).reshape(*self.s.shape)
        v = v / self.neighbor_mask.sum()  # normalize to fraction [0...1]
        on_mask, off_mask = (self.s == 1).flatten(), (self.s == 0).flatten()

        self.p[:, 1] = np.where(on_mask, self.lambda_off * (1. - v.flatten()), 0.)
        self.p[:, 0] = np.where(off_mask, self.alpha + self.lambda_on * v.flatten(), 0.)
        if self.pool_instance:
            self.p[on_mask, 1] *= (self.pool_instance.max_n - self.pool_instance.n)
            self.p[off_mask, 0] *= self.pool_instance.n
        self.update_total_p()
        # for site_index in range(self.s.size):
        #     self.update_site_p(site_index)
        # self.update_total_p()

    def _scalar_index_to_xy(self, index: int) -> Tuple[int, int]:
        # origin on top left corner, index first increases to the right across the row
        y = index // self.side_length
        x = (index - y * self.side_length)
        return y, x

    def _yx_to_scalar_index(self, y: int, x: int) -> int:
        # origin on top left corner, index first increases to the right across the row
        assert (y >= 0) and (x >= 0), "Error: y and/or x are negative"
        return y * self.side_length + x

    def _count_occupied_neighbors(self, site_scalar_index: int) -> int:
        center_y, center_x = self._scalar_index_to_xy(site_scalar_index)
        occupied_neighbors = 0  # translated from self.s to self.ss coordinates
        for y in (center_y, center_y + 1, center_y + 2):
            for x in (center_x, center_x + 1, center_x + 2):
                occupied_neighbors += (1 if self.ss[y, x] == 1 else 0)
        return (occupied_neighbors - 1) if self.s[center_y, center_x] == 1 else occupied_neighbors

    def _pool_factor(self) -> float:
        """
        :return:
        Compute multiplicative factor in k_on, regarding pool availability

        HARD requirement:
            - if no molecules in pool, return zero
        Examples:
            - the raw filled fraction [0...1]
            - sigmoidal function on the (scaled) number of pool particles

        :return: unitless floating point number
        """
        if self.pool_instance:
            return self.pool_instance.n
        # if self.pool_instance is None:
        #     return 1.  # infinite pool
        # frac = self.pool_instance.get_filled_frac()
        # return frac if (self.pool_factor_func is None) else self.pool_factor_func(frac)