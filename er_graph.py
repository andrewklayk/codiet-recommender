import numpy as np
from notears_util import simulate_dag, set_random_seed


def _generate_cpt(n_parents, n_values, dominant_prob):
    """Generate a CPT with one dominant outcome per parent configuration.

    Each row corresponds to one parent configuration (lexicographic order).
    The dominant value is sampled randomly per row; remaining values share
    (1 - dominant_prob) / (n_values - 1) each.
    """
    n_configs = n_values ** n_parents if n_parents > 0 else 1
    other_prob = (1.0 - dominant_prob) / (n_values - 1) if n_values > 1 else 0.0
    cpt = np.full((n_configs, n_values), other_prob)
    dominant_indices = np.random.randint(0, n_values, size=n_configs)
    cpt[np.arange(n_configs), dominant_indices] = dominant_prob
    return cpt


def generate_dag(d, s0, graph_type='ER', n_values=3, dominant_prob=0.8, seed=None):
    """Generate a random DAG with discrete probabilistic CPTs.

    Calls simulate_dag (notears) to obtain the binary adjacency matrix, then
    attaches a Conditional Probability Table to every node.

    Args:
        d (int): number of nodes
        s0 (int): expected number of edges
        graph_type (str): ER, SF, BP, PATH, PATHPERM, or G2 (default: 'ER')
        n_values (int): number of discrete values per variable (default: 3)
        dominant_prob (float): probability assigned to the most likely outcome for
            each parent configuration; the remaining (1 - dominant_prob) is split
            equally among the other values. E.g. 0.8 gives [0.8, 0.1, 0.1] for
            n_values=3. (default: 0.8)
        seed (int, optional): random seed for reproducibility

    Returns:
        dict:
            'B'        : np.ndarray [d, d] binary adjacency matrix (B[i,j]=1 means i→j)
            'parents'  : list[list[int]] — parents[j] lists parent indices of node j
            'cpts'     : list[np.ndarray] — cpts[j] has shape
                         (n_values ** len(parents[j]), n_values); rows index parent
                         configurations in lexicographic order, columns are child values
            'n_values' : int
    """
    if dominant_prob <= 0 or dominant_prob > 1:
        raise ValueError('dominant_prob must be in (0, 1]')
    if n_values < 1:
        raise ValueError('n_values must be >= 1')

    if seed is not None:
        set_random_seed(seed)

    B = simulate_dag(d, s0, graph_type)

    parents = [sorted(np.where(B[:, j] == 1)[0].tolist()) for j in range(d)]
    cpts = [_generate_cpt(len(pa), n_values, dominant_prob) for pa in parents]

    return {'B': B, 'parents': parents, 'cpts': cpts, 'n_values': n_values}


def _topological_order(B):
    """Kahn's algorithm — returns nodes in topological order."""
    d = B.shape[0]
    in_degree = B.sum(axis=0).astype(int)
    queue = [j for j in range(d) if in_degree[j] == 0]
    order = []
    while queue:
        j = queue.pop(0)
        order.append(j)
        for child in np.where(B[j, :] == 1)[0]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(int(child))
    return order


def _sample_categorical(probs):
    """Vectorised categorical sampling. probs: (n, k) -> samples: (n,) int."""
    cumprobs = probs.cumsum(axis=1)
    u = np.random.uniform(size=(probs.shape[0], 1))
    return (u > cumprobs).sum(axis=1).astype(int)


def sample_dag(n_samples, d, s0, graph_type='ER', n_values=3, dominant_prob=0.8,
               seed=None, return_dag=False):
    """Generate a dataset by ancestral sampling from a randomly generated discrete DAG.

    Calls generate_dag to build the structure and CPTs, then samples each node
    in topological order conditioned on its already-sampled parents.

    Args:
        n_samples (int): number of samples to generate
        d (int): number of nodes
        s0 (int): expected number of edges
        graph_type (str): ER, SF, BP, PATH, PATHPERM, or G2 (default: 'ER')
        n_values (int): number of discrete values per variable (default: 3)
        dominant_prob (float): probability assigned to the most likely outcome
            per parent configuration (default: 0.8)
        seed (int, optional): random seed for reproducibility
        return_dag (bool): if True, also return the dag dict from generate_dag
            (default: False)

    Returns:
        X (np.ndarray): [n_samples, d] integer array, values in {0, …, n_values-1}
        dag (dict, optional): dag dict from generate_dag, only when return_dag=True
    """
    dag = generate_dag(d, s0, graph_type=graph_type, n_values=n_values,
                       dominant_prob=dominant_prob, seed=seed)

    B = dag['B']
    parents = dag['parents']
    cpts = dag['cpts']

    X = np.zeros((n_samples, d), dtype=int)

    for j in _topological_order(B):
        pa = parents[j]
        if not pa:
            probs = np.tile(cpts[j][0], (n_samples, 1))
        else:
            # lexicographic config index across parent values
            config_idx = np.zeros(n_samples, dtype=int)
            for k, p in enumerate(pa):
                config_idx += X[:, p] * (n_values ** (len(pa) - 1 - k))
            probs = cpts[j][config_idx]

        X[:, j] = _sample_categorical(probs)

    if return_dag:
        return X, dag
    return X