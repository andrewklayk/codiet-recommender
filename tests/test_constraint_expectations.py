"""Unit tests for TripletConstraintsFn empty-cell handling.

Empty / unsatisfiable conditioning cells must be SKIPPED, not counted as a
phantom "conditional mean = 0" (which would add fake (0 - E[...])^2 violations
and their gradients). See discrete_estimator.TripletConstraintsFn.

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from discrete_estimator import (TripletConstraintsFn, Triplet, TripletType,
                                 DiscreteRecommenderPredictor)


def _chain_cfn(n_values=3):
    return TripletConstraintsFn([Triplet(TripletType.CHAIN, "A", "B", "C")],
                                ["A", "B"], "C", n_values)


class _LookupModel(nn.Module):
    """Fixed logits looked up by one feature column (col 0 = A, col 1 = B)."""

    def __init__(self, table, col):
        super().__init__()
        self.table = table
        self.col = col

    def forward(self, X):
        return self.table[X[:, self.col].long()]


class TestConstraintExpectations(unittest.TestCase):

    def test_bayes_optimal_chain_is_near_zero_with_empty_cells(self):
        """Bayes-optimal chain model (logits = log P(C|B), so it ignores A) must
        have ~0 chain violation even when some (a, b) cells are empty."""
        torch.manual_seed(0)
        nv = 3
        P = torch.softmax(torch.randn(nv, nv), dim=1)     # strictly positive
        logits = torch.log(P)                             # softmax(logits)==P(C|B)
        model = _LookupModel(logits, col=1)               # depends on B only
        # batch of 32 with A in {0,1} -> every (a=2, b) cell is empty
        A = torch.randint(0, 2, (32,)).float()
        B = torch.randint(0, nv, (32,)).float()
        X = torch.stack([A, B], dim=1)
        viol = _chain_cfn(nv)(model, X).item()
        self.assertLess(viol, 1e-4, f"Bayes-optimal model should be ~0, got {viol}")

    def test_missing_cell_no_phantom_gradient(self):
        """A batch missing (a, b) combinations must not inject a phantom (0 - E)^2
        term, nor its gradient onto the samples in the populated cell."""
        torch.manual_seed(0)
        nv = 3
        table = torch.zeros(nv, nv)
        table[0] = torch.tensor([0.1, 2.0, 0.5])          # E[C|B=0]!=0 -> phantom>0
        table = table.clone().detach().requires_grad_(True)
        model = _LookupModel(table, col=1)                # depends on B
        # all B=0, A in {0,1}: only (0,0),(1,0) populated; (2,0) & all b>0 empty
        A = torch.tensor([0, 1] * 8).float()
        B = torch.zeros(16)
        X = torch.stack([A, B], dim=1)
        viol = _chain_cfn(nv)(model, X)
        self.assertLess(viol.item(), 1e-6,
                        f"valid terms are all 0, got {viol.item()}")
        viol.sum().backward()
        g = 0.0 if table.grad is None else table.grad.abs().sum().item()
        self.assertLess(g, 1e-6,
                        f"no phantom term -> zero gradient expected, got {g}")

    def test_violating_model_is_positive(self):
        """Sanity: a model whose prediction depends on A (breaking A _||_ C | B)
        yields a strictly positive chain violation -- the skip logic did not just
        zero everything out."""
        torch.manual_seed(0)
        nv = 3
        table = torch.log(torch.softmax(torch.randn(nv, nv) * 3, dim=1))
        model = _LookupModel(table, col=0)                # depends on A
        A = torch.randint(0, nv, (300,)).float()
        B = torch.randint(0, nv, (300,)).float()
        X = torch.stack([A, B], dim=1)
        viol = _chain_cfn(nv)(model, X).item()
        self.assertGreater(viol, 1e-2, f"A-dependent model should violate, got {viol}")

    def test_public_api_intact(self):
        """fit() + constraint_violation() still work after the refactor."""
        torch.manual_seed(0)
        np.random.seed(0)
        names = ["A", "B", "C"]
        W = np.zeros((3, 3)); W[0, 1] = 1; W[1, 2] = 1
        data = pd.DataFrame(np.random.randint(0, 3, (120, 3)), columns=names)
        m = DiscreteRecommenderPredictor(
            W, "C", names, "none", None,
            {"network": "mlp", "n_epochs": 3, "use_alm": True})
        m.fit(data[["A", "B"]], data["C"])
        v = m.constraint_violation(data[["A", "B"]])
        self.assertEqual(set(v),
                         {"total", "by_type", "per_triplet", "n_terms", "n_triplets"})
        self.assertIsInstance(v["total"], float)
        self.assertGreaterEqual(v["total"], 0.0)
        self.assertEqual(len(v["n_terms"]), len(v["per_triplet"]))
        self.assertEqual(set(v["by_type"]), {"chain", "fork", "collider"})
        self.assertEqual(m.predict(data[["A", "B"]]).shape, (120,))


if __name__ == "__main__":
    unittest.main()