"""Unit tests for ContinuousTripletConstraintsFn: correctness of the
partial-correlation-based chain/fork/collider proxy, its two must-not-break
properties (correlation not covariance -> EXACT scale invariance; the n_terms
"could not compute" contract, now correctly scoped to only the two
irreducible cases -- see the class docstring's "Guards" section), and a
public-API smoke test of the full ContinuousRecommenderPredictor.

Data model for most tests: a linear-Gaussian chain A -> B -> y (target y,
features {A, B}), built directly against ContinuousTripletConstraintsFn with a
hand-supplied CHAIN or COLLIDER triplet, so each test isolates one specific
behaviour rather than depending on a trained model's convergence.

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import math
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from causal_triplets import Triplet, TripletType
from continuous_estimator import ContinuousTripletConstraintsFn, ContinuousRecommenderPredictor

FEATURES = ["A", "B"]
CHAIN_A_B_Y = Triplet(TripletType.CHAIN, "A", "B", "y")
COLLIDER_A_B_Y = Triplet(TripletType.COLLIDER, "A", "B", "y")


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


def _cfn(ridge=1e-4):
    return ContinuousTripletConstraintsFn([CHAIN_A_B_Y], FEATURES, "y", ridge=ridge)


def _cfn_collider(margin=0.30, ridge=1e-4):
    return ContinuousTripletConstraintsFn([COLLIDER_A_B_Y], FEATURES, "y",
                                          ridge=ridge, collider_margin=margin)


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

    def test_affine_in_B_near_zero_violation_and_computable(self):
        """A model that is an EXACT affine function of B (the Bayes-optimal
        predictor for a linear-Gaussian chain) has ~0 residual once you
        condition on B -- that is the CORRECT, computable answer (A _||_
        y_hat | B trivially holds, since y_hat has no variation left once B
        is known), not an "undecidable" case. Before the fix this wrongly
        reported n_terms == 0 (the hard eps guard mistook the model's own
        vanishing residual for "could not compute")."""
        A, B = _chain_data()
        X = torch.stack([A, B], dim=1)
        model = _LinearOfColumn(col=1, weight=1.0, bias=0.3)   # affine in B
        cfn = _cfn()
        (viol, n_terms), = cfn.violations_with_terms(model, X)
        self.assertEqual(n_terms, 1)
        self.assertLess(viol.item(), 1e-3,
                        f"affine-in-B model should be ~0, got {viol.item()}")

    def test_scale_invariance(self):
        """Correlation, not covariance: scaling the model's output leaves the
        violation EXACTLY unchanged (to high relative precision) -- this is
        what stops ALM from "cheating" by shrinking predictions (see class
        docstring point 1). Checked both up (x10) and down (x0.01, x1e-4) --
        the risky direction is down, since that is what would let a model
        escape a penalty by shrinking toward 0, and the old absolute "+ eps"
        denominator broke exactly that direction (measured 9.8e-3 deviation
        at scale=1e-4) while doing fine at x10."""
        A, B = _chain_data()
        X = torch.stack([A, B], dim=1)
        cfn = _cfn()
        base = _LinearOfColumn(col=0, weight=1.0, bias=0.3)
        (v0, n0), = cfn.violations_with_terms(base, X)
        self.assertEqual(n0, 1)
        for scale in [10.0, 0.01, 1e-4]:
            scaled = _LinearOfColumn(col=0, weight=1.0 * scale, bias=0.3 * scale)
            (v1, n1), = cfn.violations_with_terms(scaled, X)
            self.assertEqual(n1, 1)
            self.assertTrue(
                math.isclose(v0.item(), v1.item(), rel_tol=1e-6, abs_tol=1e-12),
                f"scale={scale}: base={v0.item()} scaled={v1.item()}")

    def test_constant_predictor_chain_near_zero_violation(self):
        """A degenerate (constant) predictor on a CHAIN is ALSO a genuinely
        computable, correct ~0 violation: a constant has no residual
        variation at all, so it is trivially uncorrelated with A given B.
        n_terms == 1 (computed), not 0 (see class docstring's Guards
        section: the target's own degeneracy is not a hard-guard case)."""
        A, B = _chain_data()
        X = torch.stack([A, B], dim=1)
        model = _ConstantModel(c=2.0)
        cfn = _cfn()
        (viol, n_terms), = cfn.violations_with_terms(model, X)
        self.assertEqual(n_terms, 1)
        self.assertLess(viol.item(), 1e-3,
                        f"constant model should be ~0 on CHAIN, got {viol.item()}")

    def test_constant_predictor_collider_full_violation(self):
        """A degenerate (constant) predictor on a COLLIDER is the WORST case,
        not the best: zero conditional dependence between the collider's
        endpoints is a full violation of the margin, not an escape from it.
        Before the fix this wrongly reported n_terms == 0 and violation == 0
        -- the hinge was silently bypassed by becoming unmeasurable."""
        A, B = _chain_data()
        X = torch.stack([A, B], dim=1)
        model = _ConstantModel(c=2.0)
        cfn = _cfn_collider(margin=0.30)
        (viol, n_terms), = cfn.violations_with_terms(model, X)
        self.assertEqual(n_terms, 1)
        self.assertAlmostEqual(viol.item(), 0.30, places=2,
                               msg=f"expected ~margin (0.30), got {viol.item()}")

    def test_constant_data_column_reports_n_terms_zero(self):
        """The one guard that MUST remain: a constant DATA (non-target)
        column carries no signal to correlate against, so it is genuinely
        unmeasurable -- n_terms == 0."""
        A, B = _chain_data()
        A_const = torch.zeros_like(A) + 5.0   # constant DATA column
        X = torch.stack([A_const, B], dim=1)
        model = _LinearOfColumn(col=1, weight=1.0)
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

    def test_null_bias_is_small_at_deployed_batch_size(self):
        """Small-batch bias check (see class docstring's "Small-batch bias"
        note): under the null (a genuinely B-dependent, non-degenerate model,
        so the TRUE violation is ~0), the mean measured violation should not
        be dominated by sampling noise.

        NOTE: at batch=32 the debiased *pre-clip* estimator is essentially
        exactly unbiased (empirically verified: mean ~= -0.001, from a much
        larger repeat count than used here), confirming nu = n - p is the
        right formula (nu = n - p - 1 was also checked and makes no material
        difference). But relu()'s clip-to-nonnegative -- required because
        rho^2 cannot be negative -- reintroduces a one-sided bias in the
        CLIPPED mean at small batch (empirically ~0.018 at batch=32, vs. the
        ~0.0359 pre-debias figure -- real, but not under ~0.005). This is why
        fix (b) bumps batch_size to 256 in every continuous*_constrained.yaml
        (only the constrained configs -- that's where per-minibatch primal
        gradient noise on this term actually matters): at batch=256 the
        clipped mean IS comfortably under the 0.005 target (empirically
        ~0.0016), so that is what this test checks, matching what constrained
        training actually uses.
        """
        cfn = _cfn()
        rng = np.random.default_rng(2)
        batch = 256
        vals = []
        for trial in range(200):
            A_np = rng.standard_normal(batch)
            B_np = 0.7 * A_np + rng.standard_normal(batch) * 0.7
            X = torch.tensor(np.stack([A_np, B_np], axis=1), dtype=torch.float32)
            model = _MLPOfColumn(col=1, seed=trial)   # genuinely B-only, nontrivial
            (viol, n_terms), = cfn.violations_with_terms(model, X)
            if n_terms == 1:
                vals.append(viol.item())
        vals = np.array(vals)
        self.assertGreater(len(vals), 150, "too many non-computable draws")
        self.assertLess(vals.mean(), 0.005,
                        f"null-bias mean at batch={batch} should be < 0.005, "
                        f"got {vals.mean():.5f}")


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
