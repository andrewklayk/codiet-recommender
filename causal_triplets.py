"""Graph-only causal triplet logic: chain/fork/collider enumeration and the
parent / Markov-blanket feature-restriction baselines.

Everything here operates purely on the interaction graph (w_est) and variable
names -- no estimator, no data, no torch. This is what lets both the discrete
estimator (discrete_estimator.py) and the planned continuous estimator share
the same causal-structure logic instead of each re-deriving it.
"""
from dataclasses import dataclass
from enum import Enum

import networkx as nx


class TripletType(Enum):
    """Causal structure of a triplet centred on `centre`.

    CHAIN    — left → centre → right
    FORK     — left ← centre → right   (common cause)
    COLLIDER — left → centre ← right   (common effect / v-structure)
    """
    CHAIN = 'chain'
    FORK = 'fork'
    COLLIDER = 'collider'


@dataclass(frozen=True)
class Triplet:
    """A causal triplet over three named variables, centred on `centre`."""
    type: TripletType
    left: str
    centre: str
    right: str


def parent_features(w_est, names, target_col, feature_names):
    """Subset of `feature_names` that are direct causes (parents) of the target.

    Uses the interaction graph w_est (w_est[i, j] != 0 means i -> j): the
    parents of the target are exactly the variables with a directed edge
    into it. Order follows `feature_names`.
    """
    names = list(names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    t = name_to_idx[target_col]
    parents = {names[i] for i in range(len(names)) if w_est[i, t] != 0}
    return [f for f in feature_names if f in parents]


def markov_blanket_features(w_est, names, target_col, feature_names):
    """Subset of `feature_names` in the target's Markov blanket.

    The Markov blanket is parents + children + co-parents (the other parents
    of the target's children). Under the interaction graph w_est these are
    exactly the variables that carry predictive information about the target:
    everything outside the blanket is d-separated from the target given it.
    Order follows `feature_names`.
    """
    names = list(names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    t = name_to_idx[target_col]
    W = w_est
    parents = {i for i in range(len(names)) if W[i, t] != 0}
    children = {j for j in range(len(names)) if W[t, j] != 0}
    coparents = {i for c in children for i in range(len(names)) if W[i, c] != 0}
    mb = {names[i] for i in (parents | children | coparents) - {t}}
    return [f for f in feature_names if f in mb]


def filter_triplets(triplets, allowed):
    """Keep only triplets whose left/centre/right all lie in `allowed`.

    A constraint over an excluded variable is meaningless for a model that
    never sees it: used to restrict get_target_triplets() output to the
    triplets whose variables all survive a feature restriction (e.g. parents-
    only or Markov-blanket-only training).
    """
    allowed = set(allowed)
    return [tr for tr in triplets
            if {tr.left, tr.centre, tr.right} <= allowed]


def get_target_triplets(w_est, row_and_col_names, target_col):
    """Enumerate all chain / fork / collider triplets in w_est that involve target_col.

    w_est convention (confirmed from compute_tools.py / notears_util.py):
        w_est[i, j] != 0  means  i → j  (row = source, column = target)

    For each unordered triplet {target, i, j} the function tests every node as
    the potential centre and reports matching directed structures:
        chain    :  left → centre → right
        fork     :  left ← centre → right   (common cause)
        collider :  left → centre ← right   (common effect / v-structure)

    Returns:
        list[Triplet] — each with .type (a TripletType) and the
        .left / .centre / .right variable name strings.
    """
    names = list(row_and_col_names)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    t = name_to_idx[target_col]
    others = [i for i in range(len(names)) if i != t]

    # Directed graph over node indices for d-separation tests.
    # w_est[i, j] != 0 means edge i → j.
    G = nx.DiGraph()
    G.add_nodes_from(range(len(names)))
    G.add_edges_from((i, j) for i in range(len(names))
                     for j in range(len(names)) if w_est[i, j] != 0)

    triplets = []
    for k in range(len(others)):
        for l in range(k + 1, len(others)):
            i, j = others[k], others[l]
            # rotate through each node as centre; left/right are the remaining two
            for centre, left, right in [(t, i, j), (i, t, j), (j, t, i)]:
                lc = w_est[left, centre] != 0   # left  → centre
                cl = w_est[centre, left] != 0   # centre → left
                rc = w_est[right, centre] != 0  # right → centre
                cr = w_est[centre, right] != 0  # centre → right

                # The numeric (conditional) independence each constraint
                # encodes holds only when the conditioning set actually
                # d-separates the two endpoints in the *full* DAG — not just
                # when the direct left↔right edge is absent.  Indirect active
                # paths through other nodes would otherwise invalidate the fact.
                #   chain / fork : left ⊥ right | {centre}
                #   collider     : left ⊥ right | {}        (marginal)
                sep_given_centre = nx.is_d_separator(G, {left}, {right}, {centre})
                sep_marginal     = nx.is_d_separator(G, {left}, {right}, set())

                # chain:    left → centre → right
                if lc and cr and sep_given_centre:
                    triplets.append(Triplet(TripletType.CHAIN, names[left], names[centre], names[right]))
                # chain:    right → centre → left
                if rc and cl and sep_given_centre:
                    triplets.append(Triplet(TripletType.CHAIN, names[right], names[centre], names[left]))
                # fork:     left ← centre → right
                if cl and cr and sep_given_centre:
                    triplets.append(Triplet(TripletType.FORK, names[left], names[centre], names[right]))
                # collider: left → centre ← right
                if lc and rc and sep_marginal:
                    triplets.append(Triplet(TripletType.COLLIDER, names[left], names[centre], names[right]))

    return triplets
