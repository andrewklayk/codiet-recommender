"""Unit tests for ContinuousTripletConstraintsFn: correctness of the
partial-correlation-based chain/fork/collider proxy, its two must-not-break
properties (correlation not covariance -> scale invariance; the n_terms
"could not compute" contract), and a public-API smoke test of the full
ContinuousRecommenderPredictor.

Data model for tests 1-6: a linear-Gaussian chain A -> B -> y (target y,
features {A, B}), built directly against ContinuousTripletConstraintsFn with a
hand-supplied CHAIN triplet, so each test isolates one specific behaviour
rather than depending on a trained model's convergence.

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from causal_triplets import Triplet, TripletType
from continuous_estimator import ContinuousTripletConstraintsFn, ContinuousRecommenderPredictor

FEATURES = ["A", "B"]
CHAIN_A_B_Y = Triplet(TripletType.CHAIN, "A", "B", "y")


def _chain_data(n=4000, seed=0):
    """A ~ N(0,1), B = 0.7*A + noise: a linear-Gaussian chain A -> B -> (y).

    Jointly Gaussian by construction, so A's residual after linearly
    conditioning on B is genuinely (not just linearly) independent of B --
    the property the "predicts only from B" test below relies on.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(n)
    B = 0.7 * A + rng.standard_normal(n) * 0.7
    return torch.tensor(A, dtype=torch.float32), torch.tensor(B, dtype=torch.float32)


def _cfn(ridge=1e-3):
    return ContinuousTripletConstraintsFn([CHAIN_A_B_Y], FEATURES, "y", ridge=ridge)


class _LinearOfColumn(nn.Module):
    """y_hat = weight * X[:, col] + bias (both learnable, for the gradient test)."""

    def __init__(self, col, weight=1.0, bias=0.0):
        super().__init__()
        self.col = col
        self.weight = nn.Parameter(torch.tensor(float(weight)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, X):
        return self.weight * X[:, self.col] + self.bias


class _MLPOfColumn(nn.Module):
    """y_hat = small nonlinear MLP(X[:, col]) -- deliberately NOT affine, so a
    "predicts only from B" model does not trivially zero out its own residual
    when regressed on the same linear basis used to condition on B (see the
    module-level test docstring)."""

    def __init__(self, col, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.col = col
        self.net = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, X):
        return self.net(X[:, self.col:self.col + 1]).squeeze(-1)


class _ConstantModel(nn.Module):
    def __init__(self, c=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(float(c)))

    def forward(self, X):
        return self.c + torch.zeros(X.shape[0])


class TestChainViolation(unittest.TestCase):

    def test_predicts_only_from_B_near_zero_violation(self):
        """A model that only reads B: A _||_ y_hat | B should hold (up to
        sampling noise), since y_hat is then a (nonlinear) function of B alone
        and A's linear-B residual is genuinely B-independent for jointly
        Gaussian A, B."""
        A, B = _chain_data()
        X = torch.stack([A, B], dim=1)
        model = _MLPOfColumn(col=1)
        cfn = _cfn()
        (viol, n_terms), = cfn.violations_with_terms(model, X)
        self.assertEqual(n_terms, 1)
        self.assertLess(viol.item(), 1e-2,
                        f"B-only model should be ~0, got {viol.item()}")

    def test_predicts_from_A_is_clearly_positive(self):
        """A model that reads A instead of B breaks A _||_ y_hat | B: A's own
        linear-B residual is correlated with A itself (A,B are dependent), so
        a model of A retains that dependence after conditioning on B."""
        A, B = _chain_data()
        X = torch.stack([A, B], dim=1)
        model = _LinearOfColumn(col=0, weight=1.0)
        cfn = _cfn()
        (viol, n_terms), = cfn.violations_with_terms(model, X)
        self.assertEqual(n_terms, 1)
        self.assertGreater(viol.item(), 0.05,
                           f"A-dependent model should violate, got {viol.item()}")

    def test_scale_invariance(self):
        """Correlation, not covariance: scaling the model's output by 10x must
        leave the violation unchanged (to 1e-5) -- this is what stops ALM from
        "cheating" by shrinking predictions (see class docstring point 1)."""
        A, B = _chain_data()
        X = torch.stack([A, B], dim=1)
        base = _LinearOfColumn(col=0, weight=1.0, bias=0.3)
        scaled = _LinearOfColumn(col=0, weight=10.0, bias=3.0)
        cfn = _cfn()
        (v1, n1), = cfn.violations_with_terms(base, X)
        (v2, n2), = cfn.violations_with_terms(scaled, X)
        self.assertEqual(n1, 1)
        self.assertEqual(n2, 1)
        self.assertAlmostEqual(v1.item(), v2.item(), places=5)

    def test_constant_predictor_reports_n_terms_zero(self):
        """A degenerate (constant) predictor must report n_terms == 0 --
        "could not be computed" -- not a falsely-satisfied ~0 violation (see
        class docstring point 2)."""
        A, B = _chain_data()
        X = torch.stack([A, B], dim=1)
        model = _ConstantModel(c=2.0)
        cfn = _cfn()
        (viol, n_terms), = cfn.violations_with_terms(model, X)
        self.assertEqual(n_terms, 0)
        self.assertEqual(viol.item(), 0.0)

    def test_violation_backward_gives_nonzero_gradient(self):
        """The violation must be differentiable w.r.t. the model's own
        parameters -- this is what lets ALM actually steer the predictions."""
        A, B = _chain_data()
        X = torch.stack([A, B], dim=1)
        model = _LinearOfColumn(col=0, weight=1.0)
        cfn = _cfn()
        viol = cfn(model, X)
        viol.sum().backward()
        self.assertIsNotNone(model.weight.grad)
        self.assertGreater(model.weight.grad.abs().item(), 0.0)


class TestColliderMarginCalibration(unittest.TestCase):

    def test_independent_endpoints_give_near_zero_margin(self):
        """A, B, C mutually independent (no real collider dependence): the
        calibrated margin should be ~0, same as the discrete version's
        "no phantom dependence demanded" guarantee."""
        rng = np.random.default_rng(1)
        n = 4000
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = rng.standard_normal(n)
        X = np.stack([A, C], axis=1)          # feature_names = ["A", "C"]
        t = Triplet(TripletType.COLLIDER, "A", "B", "C")
        cfn = ContinuousTripletConstraintsFn([t], ["A", "C"], "B")
        margins = cfn.calibrate_collider_margins(X, B, fraction=0.5)
        self.assertLess(margins[t], 0.05,
                        f"independent endpoints -> tiny margin expected, got {margins[t]}")
        self.assertGreaterEqual(margins[t], 0.0)


class TestPublicAPISmoke(unittest.TestCase):

    def test_fit_predict_constraint_violation(self):
        """fit() + predict() + constraint_violation() run end-to-end on a
        genuine chain A -> B -> y through the full ContinuousRecommenderPredictor
        (feature restriction, StandardScaler, ALM training, reporting)."""
        torch.manual_seed(0)
        np.random.seed(0)
        n = 300
        A = np.random.randn(n)
        B = 0.7 * A + np.random.randn(n) * 0.7
        y = 0.8 * B + np.random.randn(n) * 0.4
        data = pd.DataFrame({"A": A, "B": B, "y": y})

        names = ["A", "B", "y"]
        W = np.zeros((3, 3))
        W[0, 1] = 1   # A -> B
        W[1, 2] = 1   # B -> y

        cfg = {"network": "mlp", "n_epochs": 3, "use_alm": True}
        m = ContinuousRecommenderPredictor(W, "y", names, "none", None, cfg)
        m.fit(data[["A", "B"]], data["y"])

        preds = m.predict(data[["A", "B"]])
        self.assertEqual(preds.shape, (n,))

        v = m.constraint_violation(data[["A", "B"]])
        self.assertEqual(set(v),
                         {"total", "by_type", "per_triplet", "n_terms", "n_triplets"})
        self.assertGreaterEqual(v["n_triplets"], 1)
        self.assertIsInstance(v["total"], float)
        self.assertGreaterEqual(v["total"], 0.0)


if __name__ == "__main__":
    unittest.main()
