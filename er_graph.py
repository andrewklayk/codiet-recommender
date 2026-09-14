import numpy as np
from notears_util import simulate_dag, set_random_seed


def _generate_cpt(n_parents, n_values, dominant_prob=0.8, cpt_prior='dirichlet',
                  cpt_alpha=0.5):
    """Generate a CPT, one row per parent configuration (lexicographic order).

    Two priors over the rows:

    'dirichlet' (default) -- each row is an independent draw from
        Dirichlet(cpt_alpha, ..., cpt_alpha). Smaller cpt_alpha gives peakier
        rows; cpt_alpha = 1 is uniform over the simplex.

    'dominant' (legacy) -- every row is a permutation of
        (dominant_prob, q, ..., q) with q = (1 - dominant_prob)/(n_values - 1);
        the position of the dominant value is drawn per row.

    Why 'dirichlet' is the default: under 'dominant' EVERY conditional is the
    same distribution up to a relabelling, so the Bayes error of every node in
    every graph is pinned at 1 - dominant_prob. Measured by exact enumeration
    over 10 random 10-node ER graphs, raw Bayes error given all other variables
    was 0.169 +- 0.041 (hard-capped at 0.200) under 'dominant' vs 0.202 +- 0.116
    (range 0.020-0.540) under Dirichlet(0.5) -- i.e. 'dominant' leaves almost no
    variation in task difficulty for an experiment to resolve, which is why
    every dataset and every method came out looking alike.

    It also suppresses the very effect these experiments measure. On the
    majority-baseline-normalised scale, the gap between what a parents-only
    model can reach and what a full-information model can reach was 0.132 under
    'dominant' (dominant_prob 0.8) but 0.258 under Dirichlet(0.5), against a
    per-cell seed noise of ~0.203 -- so the signal goes from below the noise to
    above it. Lowering dominant_prob does NOT help (0.065 at 0.6, 0.036 at 0.5):
    it shrinks the learnable range faster than it adds variation.
    """
    n_configs = n_values ** n_parents if n_parents > 0 else 1
    if cpt_prior == 'dirichlet':
        if cpt_alpha <= 0:
            raise ValueError('cpt_alpha must be > 0')
        return np.random.dirichlet([cpt_alpha] * n_values, size=n_configs)
    if cpt_prior != 'dominant':
        raise ValueError(
            f"cpt_prior must be 'dirichlet' or 'dominant', got '{cpt_prior}'")
    other_prob = (1.0 - dominant_prob) / (n_values - 1) if n_values > 1 else 0.0
    cpt = np.full((n_configs, n_values), other_prob)
    dominant_indices = np.random.randint(0, n_values, size=n_configs)
    cpt[np.arange(n_configs), dominant_indices] = dominant_prob
    return cpt


def generate_dag(d, s0, graph_type='ER', n_values=3, dominant_prob=0.8, seed=None,
                 cpt_prior='dirichlet', cpt_alpha=0.5):
    """Generate a random DAG with discrete probabilistic CPTs.

    Calls simulate_dag (notears) to obtain the binary adjacency matrix, then
    attaches a Conditional Probability Table to every node.

    Args:
        d (int): number of nodes
        s0 (int): expected number of edges
        graph_type (str): ER, SF, BP, PATH, PATHPERM, or G2 (default: 'ER')
        n_values (int): number of discrete values per variable (default: 3)
        dominant_prob (float): only used when cpt_prior='dominant' -- probability
            assigned to the most likely outcome for each parent configuration;
            the remaining (1 - dominant_prob) is split equally among the other
            values. E.g. 0.8 gives [0.8, 0.1, 0.1] for n_values=3. (default: 0.8)
        seed (int, optional): random seed for reproducibility
        cpt_prior (str): 'dirichlet' (default) or 'dominant' -- how each CPT row
            is drawn; see _generate_cpt for why 'dominant' makes every node's
            Bayes error identical and is kept only to reproduce older results.
        cpt_alpha (float): Dirichlet concentration when cpt_prior='dirichlet'
            (default: 0.5)

    Returns:
        dict:
            'B'        : np.ndarray [d, d] binary adjacency matrix (B[i,j]=1 means i→j)
            'parents'  : list[list[int]] — parents[j] lists parent indices of node j
            'cpts'     : list[np.ndarray] — cpts[j] has shape
                         (n_values ** len(parents[j]), n_values); rows index parent
                         configurations in lexicographic order, columns are child values
            'n_values' : int
    """
    if cpt_prior == 'dominant' and (dominant_prob <= 0 or dominant_prob > 1):
        raise ValueError('dominant_prob must be in (0, 1]')
    if n_values < 1:
        raise ValueError('n_values must be >= 1')

    if seed is not None:
        set_random_seed(seed)

    B = simulate_dag(d, s0, graph_type)

    parents = [sorted(np.where(B[:, j] == 1)[0].tolist()) for j in range(d)]
    cpts = [_generate_cpt(len(pa), n_values, dominant_prob, cpt_prior, cpt_alpha)
            for pa in parents]

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
               seed=None, return_dag=False, cpt_prior='dirichlet', cpt_alpha=0.5):
    """Generate a dataset by ancestral sampling from a randomly generated discrete DAG.

    Calls generate_dag to build the structure and CPTs, then samples each node
    in topological order conditioned on its already-sampled parents.

    Args:
        n_samples (int): number of samples to generate
        d (int): number of nodes
        s0 (int): expected number of edges
        graph_type (str): ER, SF, BP, PATH, PATHPERM, or G2 (default: 'ER')
        n_values (int): number of discrete values per variable (default: 3)
        dominant_prob (float): only used when cpt_prior='dominant' (default: 0.8)
        seed (int, optional): random seed for reproducibility
        cpt_prior (str): 'dirichlet' (default) or 'dominant'; see _generate_cpt
        cpt_alpha (float): Dirichlet concentration (default: 0.5)
        return_dag (bool): if True, also return the dag dict from generate_dag
            (default: False)

    Returns:
        X (np.ndarray): [n_samples, d] integer array, values in {0, …, n_values-1}
        dag (dict, optional): dag dict from generate_dag, only when return_dag=True
    """
    dag = generate_dag(d, s0, graph_type=graph_type, n_values=n_values,
                       dominant_prob=dominant_prob, seed=seed,
                       cpt_prior=cpt_prior, cpt_alpha=cpt_alpha)

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