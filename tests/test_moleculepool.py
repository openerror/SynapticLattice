from src.MoleculePool import MoleculePool


def test_trigger():
    """
        Starting from an empty pool and running trigger should lead to
        the addiiton of 1 molecule into the pool. Confirm with different
        random seeds, and with checking event_ix returned
    """
    seeds = (42, 522, 32)  # arbitrary numbers
    for ss in seeds:
        pool = MoleculePool(delta=1., gamma=1., max_n=5, init_n=0, random_state=ss)
        event_ix = pool.trigger()
        assert event_ix == 0
        assert pool.n == 1

def test_get_filled_frac():
    pool = MoleculePool(1., 1., max_n=5, init_n=1)
    assert pool.get_filled_frac() == 0.2

    event_ix = pool.trigger()
    if event_ix == 0:
        assert pool.get_filled_frac() == 0.4
    elif event_ix == 1:
        assert pool.get_filled_frac() == 0.