from src.locate_first_geq import locate_first_geq


def test_empty():
    """ should return -1 because there's nothing to locate """
    assert (locate_first_geq([], 10) == -1)
    assert (locate_first_geq((), 10) == -1)


def test_small_inputs():
    """ small --- at most two elements """
    assert locate_first_geq([-1, 2], 1.5) == 1
    assert locate_first_geq([-1, 2], -4) == 0
    assert locate_first_geq([5], 1.5) == 0


def test_larger_input():
    """ larger input --- 10^1 elements or more """
    assert locate_first_geq(range(10), 3) == 3
    assert locate_first_geq(range(10), 5.5) == 6