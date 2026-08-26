"""Unit tests for causal_triplets: chain/fork/collider enumeration, the
parent / Markov-blanket feature-restriction helpers, and filter_triplets.

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import unittest

import numpy as np

from causal_triplets import (TripletType, Triplet, get_target_triplets,
                             parent_features, markov_blanket_features,
                             filter_triplets)


def _adjacency(names, edges):
    """Build a w_est adjacency matrix (w[i, j] != 0 means i -> j) from
    (src, dst) name pairs over `names`."""
    idx = {n: i for i, n in enumerate(names)}
    W = np.zeros((len(names), len(names)))
    for src, dst in edges:
        W[idx[src], idx[dst]] = 1
    return W


class TestChainForkCollider(unittest.TestCase):

    def test_chain(self):
        """A -> B -> C, target B: exactly one CHAIN(A, B, C) triplet."""
        names = ["A", "B", "C"]
        W = _adjacency(names, [("A", "B"), ("B", "C")])
        triplets = get_target_triplets(W, names, "B")
        self.assertEqual(triplets, [Triplet(TripletType.CHAIN, "A", "B", "C")])

    def test_fork(self):
        """A <- B -> C, target B: exactly one FORK(A, B, C) triplet."""
        names = ["A", "B", "C"]
        W = _adjacency(names, [("B", "A"), ("B", "C")])
        triplets = get_target_triplets(W, names, "B")
        self.assertEqual(triplets, [Triplet(TripletType.FORK, "A", "B", "C")])

    def test_collider(self):
        """A -> B <- C, target B: exactly one COLLIDER(A, B, C) triplet."""
        names = ["A", "B", "C"]
        W = _adjacency(names, [("A", "B"), ("C", "B")])
        triplets = get_target_triplets(W, names, "B")
        self.assertEqual(triplets, [Triplet(TripletType.COLLIDER, "A", "B", "C")])

    def test_indirect_active_path_blocks_chain(self):
        """A -> B -> C looks like a chain, but an extra A -> D -> C path means
        A and C are NOT d-separated given only {B} (the active path through D
        is never blocked). The CHAIN(A, B, C) triplet must therefore be
        REJECTED even though the raw A->B->C edges are present -- this is the
        indirect-active-path case get_target_triplets guards against (see its
        sep_given_centre / sep_marginal checks)."""
        names = ["A", "B", "C", "D"]
        W = _adjacency(names, [("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")])
        triplets = get_target_triplets(W, names, "B")
        self.assertNotIn(Triplet(TripletType.CHAIN, "A", "B", "C"), triplets)
        # sanity: the function still finds a real triplet elsewhere in this
        # graph (B <- A -> D is a genuine fork, d-separated by A), so the
        # rejection above is d-separation working, not the function silently
        # returning nothing.
        self.assertIn(Triplet(TripletType.FORK, "B", "A", "D"), triplets)


class TestFeatureRestriction(unittest.TestCase):

    def test_parent_features_is_direct_causes_only(self):
        """Collider A -> B <- C: B's parents are {A, C}, not the empty set a
        Markov-blanket view of some OTHER node might suggest."""
        names = ["A", "B", "C"]
        W = _adjacency(names, [("A", "B"), ("C", "B")])
        self.assertEqual(set(parent_features(W, names, "B", ["A", "C"])),
                         {"A", "C"})
        # a node with no parents (A is a root here) gets an empty parent set
        self.assertEqual(parent_features(W, names, "A", ["B", "C"]), [])

    def test_markov_blanket_includes_coparent(self):
        """Chain A -> B -> C plus D -> C: B's Markov blanket is {A, C, D} --
        parent A, child C, and co-parent D (D is also a parent of B's child C).
        C's Markov blanket is just its two parents {B, D}, since C has no
        children (and therefore no co-parents)."""
        names = ["A", "B", "C", "D"]
        W = _adjacency(names, [("A", "B"), ("B", "C"), ("D", "C")])
        self.assertEqual(set(markov_blanket_features(W, names, "B", ["A", "C", "D"])),
                         {"A", "C", "D"})
        self.assertEqual(set(markov_blanket_features(W, names, "C", ["A", "B", "D"])),
                         {"B", "D"})


class TestFilterTriplets(unittest.TestCase):

    def test_filter_drops_triplets_with_excluded_variables(self):
        triplets = [Triplet(TripletType.CHAIN, "A", "B", "C"),
                    Triplet(TripletType.FORK, "B", "D", "E")]
        kept = filter_triplets(triplets, {"A", "B", "C"})
        self.assertEqual(kept, [Triplet(TripletType.CHAIN, "A", "B", "C")])

    def test_filter_keeps_everything_when_all_allowed(self):
        triplets = [Triplet(TripletType.CHAIN, "A", "B", "C"),
                    Triplet(TripletType.FORK, "B", "D", "E")]
        kept = filter_triplets(triplets, {"A", "B", "C", "D", "E"})
        self.assertEqual(kept, triplets)


if __name__ == "__main__":
    unittest.main()
