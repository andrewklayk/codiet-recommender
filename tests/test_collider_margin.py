"""Unit tests for the two COLLIDER-branch fixes in TripletConstraintsFn:

1. When the collider centre is the target, the marginal equality term
   (E(AC) - E(A)E(C))^2 is data-only (constant w.r.t. the model) and must be
   SKIPPED so it neither inflates the violation nor feeds the ALM dual.
2. The hinge margin is calibrated per-triplet from the dependence achievable in
   the true labels (calibrate_collider_margins), not a fixed 0.1.

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import unittest

import numpy as np
import torch
import torch.nn as nn

from discrete_estimator import TripletConstraintsFn, Triplet, TripletType


class _Const(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.k = n_classes

    def forward(self, X):
        return torch.zeros(X.shape[0], self.k)


class TestColliderMargin(unittest.TestCase):

    def test_data_only_equality_skipped_when_centre_is_target(self):
        """Centre=target collider with margin 0: the only remaining piece would
        be the data-only marginal term. It must be skipped -> violation ~0 even
        though the endpoints A,C are strongly (marginally) dependent."""
        torch.manual_seed(0)
        nv = 3
        a = torch.randint(0, nv, (500,))
        X = torch.stack([a.float(), a.float()], dim=1)      # A == C (dependent)
        A, C = X[:, 0], X[:, 1]
        marg = abs((A * C).mean().item() - A.mean().item() * C.mean().item())
        self.assertGreater(marg, 0.3, f"setup: endpoints must be dependent, {marg}")
        cfn = TripletConstraintsFn([Triplet(TripletType.COLLIDER, "A", "B", "C")],
                                   ["A", "C"], "B", nv, collider_margin=0.0)
        viol = cfn(_Const(nv), X).item()
        self.assertLess(viol, 1e-6,
                        f"data-only equality not skipped: viol={viol} "
                        f"(would be ~{marg**2:.3f} if kept)")

    def test_margin_calibration_matches_empirical(self):
        """Genuine collider B=(A+C)%3: calibrated margin == 0.5 * empirical
        conditional dependence measured from the true labels (numpy reference)."""
        np.random.seed(0)
        nv = 3
        A = np.random.randint(0, nv, 3000)
        C = np.random.randint(0, nv, 3000)
        B = (A + C) % nv                                    # collider: A->B<-C
        X = np.stack([A, C], axis=1)
        t = Triplet(TripletType.COLLIDER, "A", "B", "C")
        cfn = TripletConstraintsFn([t], ["A", "C"], "B", nv)
        margins = cfn.calibrate_collider_margins(X, B, fraction=0.5)

        deps = []                                           # independent reference
        for b in range(nv):
            m = B == b
            if not m.any():
                continue
            dep = abs((A[m] * C[m]).mean() - A[m].mean() * C[m].mean())
            deps.append(dep)
        expected = 0.5 * (sum(deps) / len(deps))
        self.assertAlmostEqual(margins[t], expected, places=4)
        self.assertGreater(margins[t], 0.05,
                           f"real collider should give a sizeable margin, {margins[t]}")
        self.assertEqual(cfn._collider_margins[t], margins[t])   # stored on the fn

    def test_margin_small_for_independent_endpoints(self):
        """No real collider dependence (B independent of A,C) -> margin ~ 0
        (only sampling noise), so the hinge does not demand phantom dependence."""
        np.random.seed(1)
        nv = 3
        A = np.random.randint(0, nv, 3000)
        C = np.random.randint(0, nv, 3000)
        B = np.random.randint(0, nv, 3000)                  # independent of A, C
        X = np.stack([A, C], axis=1)
        t = Triplet(TripletType.COLLIDER, "A", "B", "C")
        cfn = TripletConstraintsFn([t], ["A", "C"], "B", nv)
        margins = cfn.calibrate_collider_margins(X, B, fraction=0.5)
        self.assertLess(margins[t], 0.05,
                        f"independent -> tiny margin expected, got {margins[t]}")
        self.assertGreaterEqual(margins[t], 0.0)


if __name__ == "__main__":
    unittest.main()