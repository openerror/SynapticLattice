from collections import defaultdict
from typing import *
import numpy as np
from src.locate_first_geq import locate_first_geq
from src.MoleculePool import MoleculePool
from src.Synapse import Synapse


def load_rates_file(fpath: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    syp_rates, pool_rates = defaultdict(lambda: None), defaultdict(lambda: None)
    with open(fpath, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            name, val = line.split()
            if name in ("alpha", "beta", "lambda_on", "lambda_off"):
                syp_rates[name] = float(val)
            elif name in ("delta", "gamma"):
                pool_rates[name] = float(val)
    return syp_rates, pool_rates


def load_simparams_file(fpath: str) -> Dict[str, int]:
    sim_params = defaultdict(lambda: None)
    with open(fpath, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            name, val = line.split()
            sim_params[name] = int(val)
    return sim_params