# Self-Organizational Principles of Postsynaptic Membranes

Biophysics research boosted with data-era computational tools and algorithms.

Simulated molecular dynamics and organization on the postsynaptic membrane using Gillespie's
algorithm. Implemented and unit-tested in Python 3.8, parallelized using [Ray](https://ray.io/).
Accelerated counting of nearest neighbors using [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)-like vectorization; see `src/im2col.py` and `src/Synapse.py`.

## Usage
- Run from `simulate.py`
- "Content" of each simulation instance described in `single_trial.py`
