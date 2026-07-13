"""Unit tests for TripletConstraintsFn empty-cell handling.

Empty / unsatisfiable conditioning cells must be SKIPPED, not counted as a
phantom "conditional mean = 0" (which would add fake (0 - E[...])^2 violations
and their gradients). See discrete_estimator.TripletConstraintsFn.

Run:   python test_constraint_expectations.py
       (also collectable by pytest -- the checks are test_* functions.)
"""
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


def test_bayes_optimal_chain_is_near_zero_with_empty_cells():
    """Bayes-optimal chain model (logits = log P(C|B), so it ignores A) must
    have ~0 chain violation even when some (a, b) cells are empty in the batch."""
    torch.manual_seed(0)
    nv = 3
    P = torch.softmax(torch.randn(nv, nv), dim=1)     # strictly positive P(C|B)
    logits = torch.log(P)                             # softmax(logits) == P(C|B)
    model = _LookupModel(logits, col=1)               # depends on B only
    # batch of 32 with A in {0,1} -> every (a=2, b) cell is empty
    A = torch.randint(0, 2, (32,)).float()
    B = torch.randint(0, nv, (32,)).float()
    X = torch.stack([A, B], dim=1)
    viol = _chain_cfn(nv)(model, X).item()
    assert viol < 1e-4, f"Bayes-optimal model should be ~0, got {viol}"
    print(f"test_bayes_optimal_chain_is_near_zero_with_empty_cells: "
          f"viol={viol:.2e}  OK")


def test_missing_cell_no_phantom_gradient():
    """A batch missing (a, b) combinations must not inject a phantom (0 - E)^2
    term, nor its gradient onto the samples in the populated cell."""
    torch.manual_seed(0)
    nv = 3
    table = torch.zeros(nv, nv)
    table[0] = torch.tensor([0.1, 2.0, 0.5])          # E[C|B=0] != 0 -> phantom>0
    table = table.clone().detach().requires_grad_(True)
    model = _LookupModel(table, col=1)                # depends on B
    # all B=0, A in {0,1}: only cells (0,0),(1,0) populated; (2,0) & all b>0 empty
    A = torch.tensor([0, 1] * 8).float()
    B = torch.zeros(16)
    X = torch.stack([A, B], dim=1)
    viol = _chain_cfn(nv)(model, X)
    assert viol.item() < 1e-6, f"valid terms are all 0, got {viol.item()}"
    viol.sum().backward()
    g = 0.0 if table.grad is None else table.grad.abs().sum().item()
    assert g < 1e-6, f"no phantom term -> zero gradient expected, got {g}"
    print(f"test_missing_cell_no_phantom_gradient: "
          f"viol={viol.item():.2e} grad={g:.2e}  OK")


def test_violating_model_is_positive():
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
    assert viol > 1e-2, f"A-dependent model should violate, got {viol}"
    print(f"test_violating_model_is_positive: viol={viol:.4f}  OK")


def test_public_api_intact():
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
    assert set(v) == {"total", "per_triplet", "n_triplets"}
    assert isinstance(v["total"], float) and v["total"] >= 0.0
    assert m.predict(data[["A", "B"]]).shape == (120,)
    print(f"test_public_api_intact: violation={v['total']:.4f}  OK")


if __name__ == "__main__":
    test_bayes_optimal_chain_is_near_zero_with_empty_cells()
    test_missing_cell_no_phantom_gradient()
    test_violating_model_is_positive()
    test_public_api_intact()
    print("ALL TESTS PASSED")
