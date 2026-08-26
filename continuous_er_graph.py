"""Continuous sibling of er_graph.py: random ER-graph datasets from a
linear-Gaussian SEM instead of discrete CPTs, for the *_continuous.py test
scripts (mirrors discrete_estimator.py / continuous_estimator.py).

Thin wrapper around notears_util's simulate_dag / simulate_parameter /
simulate_linear_sem -- same building blocks er_graph.py itself builds on,
just skipping the discretisation step.
"""
from notears_util import simulate_dag, simulate_parameter, simulate_linear_sem, set_random_seed


def sample_dag(n_samples, d, s0, graph_type='ER', w_ranges=((-2.0, -0.5), (0.5, 2.0)),
               sem_type='gauss', noise_scale=1.0, seed=None, return_dag=False):
    """Generate a dataset by ancestral sampling from a random linear-Gaussian DAG.

    Args:
        n_samples (int): number of samples to generate
        d (int): number of nodes
        s0 (int): expected number of edges
        graph_type (str): ER, SF, BP, PATH, PATHPERM, or G2 (default: 'ER')
        w_ranges (tuple): disjoint edge-weight ranges passed to simulate_parameter
        sem_type (str): noise family passed to simulate_linear_sem (default: 'gauss')
        noise_scale (float or array): additive noise scale per node (default: 1.0)
        seed (int, optional): random seed for reproducibility
        return_dag (bool): if True, also return {'B': ..., 'W': ...}
            (default: False)

    Returns:
        X (np.ndarray): [n_samples, d] float array
        dag (dict, optional): {'B': binary adjacency, 'W': weighted adjacency},
            only when return_dag=True
    """
    if seed is not None:
        set_random_seed(seed)

    B = simulate_dag(d, s0, graph_type)
    W = simulate_parameter(B, w_ranges)
    X = simulate_linear_sem(W, n_samples, sem_type, noise_scale)

    if return_dag:
        return X, {'B': B, 'W': W}
    return X
