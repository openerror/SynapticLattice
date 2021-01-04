from itertools import product
import numpy as np
import pytest
from src.MoleculePool import MoleculePool
from src.Simulation import Simulation
from src.Synapse import Synapse
from src.SypInit import SynapseInitializer


def test_scalar_index_to_xy():
    la = Synapse(side_length=(L := 3), alpha=1., beta=1., lambda_on=1., lambda_off=1.)
    scalar = 0
    for y in range(L):
        for x in range(L):
            assert la._yx_to_scalar_index(y, x) == scalar
            scalar += 1


def test_yx_to_scalar_index():
    la = Synapse(side_length=(L := 3), alpha=1., beta=1., lambda_on=1., lambda_off=1.)
    cartesian_indices = product(range(L), range(L))
    for cartesian, scalar in zip(cartesian_indices, range(L ** 2)):
        assert cartesian == la._scalar_index_to_xy(scalar)
        

def get_neighbors():
    la = Synapse(side_length=(L := 3), alpha=1., beta=1., lambda_on=1., lambda_off=1.)
    query_answers = [
        ((0, 0), [(0, 1), (1, 0), (1, 1)]),
        ((0, 1), [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]),
        ((1, 1), list(filter(lambda t: t[0] != t[1], product(range(L), range(L))))),
        [(2, 2), [(1, 1), (1, 2), (2, 1)]],
        [(2, 0), [(1, 0), (1, 1), (2, 1)]]
    ]
    
    for query, ans in query_answers:
        assert ans == la.get_neighbors(*query)


def test_count_occupied_neighbors():
    syp = Synapse(side_length=3, alpha=1, beta=1, lambda_on=1, lambda_off=1)

    # Fill lattice to at 4 arbitrary sites
    coordinates = [(0, 0), (0, 1), (1, 0), (1, 2)]
    true_amount = (2, 3, 2, 1)
    for y, x in coordinates: syp.s[y, x] = 1
    syp.ss[1:-1, 1:-1] = syp.s
    for (y, x), ta in zip(coordinates, true_amount):
        site_index = syp._yx_to_scalar_index(y, x)
        assert syp._count_occupied_neighbors(site_index) == ta

    # Now fill up the entire lattice
    syp.s = np.ones(shape=(syp.side_length, syp.side_length), dtype=np.int8)
    syp.ss[1:-1, 1:-1] = syp.s
    for y, x in product(range(syp.side_length), range(syp.side_length)):
        scalar_index = syp._yx_to_scalar_index(y, x)

        if (y, x) in ((0, 0), (0, syp.side_length-1),
                      (syp.side_length-1, 0), (syp.side_length-1, syp.side_length-1)):
            assert syp._count_occupied_neighbors(scalar_index) == 3
        elif (0 in (y, x)) or (syp.side_length-1 in (y, x)):
            assert syp._count_occupied_neighbors(scalar_index) == 5
        else:
            assert syp._count_occupied_neighbors(scalar_index) == 8


def test_select_site():
    syp = Synapse(side_length=3, alpha=1, beta=1, lambda_on=1, lambda_off=1)
    with pytest.warns(RuntimeWarning):
        for _ in range(30):
            # propensities are all zero, site 0 should always be selected
            # *no matter* the randomness involved
            assert syp._select_site() == 0


def test_update_site_p():
    syp = Synapse(side_length=2, alpha=1, beta=1, lambda_on=1, lambda_off=0.5)
    syp.s[:] = 1; syp.s[1, 1] = 0
    syp.ss[1:-1, 1:-1] = syp.s

    for y, x in product(range(syp.side_length), range(syp.side_length)):
        syp.update_site_p(syp._yx_to_scalar_index(y, x))

    true_p = np.zeros_like(syp.p)  # update below if dynamics model changes
    true_p[0, 1] = syp.beta + syp.lambda_off * (1. - 2/8)
    true_p[1, 1] = syp.beta + syp.lambda_off * (1. - 2/8)
    true_p[2, 1] = syp.beta + syp.lambda_off * (1. - 2/8)
    true_p[3, 0] = syp.alpha + syp.lambda_on * 3/8
    assert np.allclose(true_p, syp.p)


def test_update_all_p():
    """ Compare vectorized computation to for loop + update_site_p() """
    pool = MoleculePool(delta=1., gamma=2., max_n=1000, init_n=100)
    syp = Synapse(side_length=5, alpha=0.05, beta=0.05, lambda_on=1., lambda_off=0.8, pool_instance=pool)

    # Initialize synaptic occupancies and propensities
    # SynapseInitializer uses for loop + update_site_p()
    SynapseInitializer().full_init(syp)
    for_loop_p = syp.p.copy()

    # Perturb, update_all_p(), and compare
    syp.p = np.random.random(size=syp.p.shape)
    syp.update_all_p()
    assert np.allclose(for_loop_p, syp.p)


def test_trigger():
    """
    Check if changes are correctly propagated in synapse AND pool
    """
    pool = MoleculePool(delta=1., gamma=2., max_n=1000, init_n=100)
    syp = Synapse(side_length=3, alpha=0.05, beta=1, lambda_on=1., lambda_off=0.8, pool_instance=pool)

    # Initialize synapse
    syp.s = np.random.randint(low=0, high=2, size=syp.s.shape)
    syp.ss[1:-1, 1:-1] = syp.s
    for site_ix in range(syp.s.size):
        syp.update_site_p(site_ix)
    syp.total_p = syp.p.sum()

    # Trigger one exocytosis/endocytos event
    old_s, old_p_n = syp.s.copy(), syp.pool_instance.n
    y, x = syp._scalar_index_to_xy(syp.trigger())

    if old_s[y, x] == 1:
        assert syp.s[y, x] == 0
        assert syp.pool_instance.n == old_p_n+1
    elif old_s[y, x] == 0:
        assert syp.s[y, x] == 1
        assert syp.pool_instance.n == old_p_n-1