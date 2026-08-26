import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from causal_estimator_base import CausalConstrainedPredictor
from causal_triplets import TripletType, Triplet
from discrete_networks import build_network


def compute_discrete_predictor_errors_scikit(estimator, X, y):
    """Scoring for discrete estimator: returns classification error (1 - accuracy)."""
    return 1.0 - accuracy_score(y, estimator.predict(X))


class TripletConstraintsFn:
    """Constraint function for ALM training, one violation per causal triplet.

    The violations are computed from the model's PREDICTED target distribution
    (so they are differentiable w.r.t. the model and the ALM update genuinely
    drives the predictions toward the causal-independence facts) combined with
    the observed feature data. `feature_names` must list the columns of the
    X_batch the model is called on, in order.

    Args:
        triplets     : output of causal_triplets.get_target_triplets()
        feature_names: ordered column names of X_batch passed to __call__
        target_col   : name of the target variable (supplied by the model)
        n_values     : category cardinality (drives the conditioning loops)
    """

    def __init__(self, triplets, feature_names, target_col, n_values,
                 collider_margin=0.1):
        self.triplets = triplets
        self.feature_names = list(feature_names)
        self.target_col = target_col
        self.n_values = n_values
        self.collider_margin = collider_margin
        # per-triplet hinge margins, keyed by the (frozen) Triplet; set by
        # calibrate_collider_margins(). Empty -> fall back to collider_margin.
        self._collider_margins = {}
        self._col_idx = {name: i for i, name in enumerate(self.feature_names)}

    @property
    def n_constraints(self):
        return max(1, len(self.triplets))

    def _value(self, name, X, soft_val):
        """Per-sample value vector for `name`.

        Feature -> its data column. Target -> the model's expected category
        E[target] = Σ_k k·p_k(x) (differentiable). None if the variable is
        absent from the batch.
        """
        if name == self.target_col:
            return soft_val
        idx = self._col_idx.get(name)
        return X[:, idx].float() if idx is not None else None

    def _weight(self, cond_name, cond_val, X, soft_probs):
        """Per-sample conditioning weight for the event (cond_name == cond_val).

        Feature -> hard 0/1 indicator. Target -> predicted probability
        p(target=cond_val | x) (so conditioning on the target stays
        differentiable). None if the variable is absent.
        """
        if cond_name == self.target_col:
            if cond_val >= soft_probs.shape[1]:
                return torch.zeros(soft_probs.shape[0])
            return soft_probs[:, cond_val]
        idx = self._col_idx.get(cond_name)
        if idx is None:
            return None
        return (X[:, idx].long() == int(cond_val)).float()

    def _expect(self, value, given, X, soft_probs):
        """Soft-weighted (conditional) mean Σ w·value / Σ w of `value`.

        Hard 0/1 weights for feature conditioning, soft predicted probabilities
        for target conditioning; a plain mean when `given` is empty. Returns
        None when the conditioning cell is unsatisfiable -- a needed variable is
        absent from the batch, or no probability mass / no samples fall in the
        cell (denom <= 1e-8). Returning None (rather than a phantom zero) lets
        the caller SKIP empty cells instead of counting (0 - E[...])^2 as a fake
        violation.
        """
        w = torch.ones(value.shape[0])
        if given:
            for cond_name, cond_val in given.items():
                wi = self._weight(cond_name, cond_val, X, soft_probs)
                if wi is None:
                    return None
                w = w * wi
        denom = w.sum()
        if denom <= 1e-8:
            return None
        return ((w * value).sum() / denom).unsqueeze(0)

    def expectation(self, var_name, X, soft_probs, soft_val, given=None):
        """E(var_name) or E(var_name | given), target taken from the model.

        Returns None when the variable is absent or the conditioning cell is
        empty/unsatisfiable, so callers can skip it.
        """
        value = self._value(var_name, X, soft_val)
        if value is None:
            return None
        return self._expect(value, given, X, soft_probs)

    def expectation_product(self, var1, var2, X, soft_probs, soft_val, given=None):
        """E(var1 * var2 | given), same conventions (and None semantics) as
        expectation()."""
        v1 = self._value(var1, X, soft_val)
        v2 = self._value(var2, X, soft_val)
        if v1 is None or v2 is None:
            return None
        return self._expect(v1 * v2, given, X, soft_probs)

    def _constraint_for_triplet(self, triplet, X, soft_probs, soft_val):
        """(violation, n_terms) for one triplet (violation is zero when the
        constraint holds).

        Expectations involving the target use the model's predicted class
        distribution (soft_probs / soft_val); feature expectations use the data.

        Conditioning cells that are empty/unsatisfiable on this batch are
        SKIPPED (they contribute to neither the sum nor the term count) rather
        than counted as a phantom zero, so a missing (a, b) combination cannot
        inject a fake "(0 - E[...])^2" violation or push its gradient onto the
        samples in the other, populated cells. The average is taken over the
        number of VALID terms (>= 1 to avoid division by zero).

        n_terms is the number of those valid terms -- 0 means every
        conditioning cell was empty on this batch, so the returned violation is
        a meaningless zero rather than evidence the constraint holds. For
        COLLIDER triplets n_terms counts only the hinge (inequality) cells, not
        the (always data-only-or-skipped) marginal equality term.
        """
        A, B, C = triplet.left, triplet.centre, triplet.right
        if triplet.type == TripletType.CHAIN:
            # chain A→B→C:  E(C|A=a, B=b) = E(C|B=b)  for all (a,b)
            total = torch.zeros(1)
            n_terms = 0
            for b_val in range(self.n_values):
                e_c_b = self.expectation(C, X, soft_probs, soft_val, given={B: b_val})
                if e_c_b is None:                       # cell B=b empty -> skip
                    continue
                for a_val in range(self.n_values):
                    e_c_ab = self.expectation(C, X, soft_probs, soft_val,
                                              given={A: a_val, B: b_val})
                    if e_c_ab is None:                  # cell (a, b) empty -> skip
                        continue
                    total = total + (e_c_ab - e_c_b) ** 2
                    n_terms += 1
            return total / max(1, n_terms), n_terms

        if triplet.type == TripletType.FORK:
            # fork A←B→C:  E(A·C|B=b) = E(A|B=b)·E(C|B=b)  for all b
            total = torch.zeros(1)
            n_terms = 0
            for b_val in range(self.n_values):
                e_ac_b = self.expectation_product(A, C, X, soft_probs, soft_val,
                                                  given={B: b_val})
                e_a_b  = self.expectation(A, X, soft_probs, soft_val, given={B: b_val})
                e_c_b  = self.expectation(C, X, soft_probs, soft_val, given={B: b_val})
                if e_ac_b is None or e_a_b is None or e_c_b is None:
                    continue                            # cell B=b empty -> skip
                total = total + (e_ac_b - e_a_b * e_c_b) ** 2
                n_terms += 1
            return total / max(1, n_terms), n_terms

        if triplet.type == TripletType.COLLIDER:
            # collider A→B←C:
            #   equality  : E(A·C) = E(A)·E(C)          (marginal independence)
            #   inequality: E(A·C|B=b) ≠ E(A|B=b)·E(C|B=b)  for all b
            #               encoded as hinge: max(0, τ − |difference|)
            # --- equality part (marginal independence of the endpoints) ---
            # Only meaningful when it depends on the model, i.e. when an endpoint
            # is the target. If the centre is the target, both endpoints are
            # feature columns, so this term is a constant fact about the data
            # (zero gradient) -- skip it so it neither inflates the reported
            # violation nor feeds the ALM dual ratchet.
            if self.target_col in (A, C):
                e_ac = self.expectation_product(A, C, X, soft_probs, soft_val)
                e_a  = self.expectation(A, X, soft_probs, soft_val)
                e_c  = self.expectation(C, X, soft_probs, soft_val)
                eq = (torch.zeros(1) if (e_ac is None or e_a is None or e_c is None)
                      else (e_ac - e_a * e_c) ** 2)
            else:
                eq = torch.zeros(1)

            # --- inequality part: hinge over valid b cells only ---
            # Per-triplet margin (calibrated from the data's achievable
            # dependence) if available, else the fixed default.
            margin = self._collider_margins.get(triplet, self.collider_margin)
            hinge = torch.zeros(1)
            n_terms = 0
            for b_val in range(self.n_values):
                e_ac_b = self.expectation_product(A, C, X, soft_probs, soft_val,
                                                  given={B: b_val})
                e_a_b  = self.expectation(A, X, soft_probs, soft_val, given={B: b_val})
                e_c_b  = self.expectation(C, X, soft_probs, soft_val, given={B: b_val})
                if e_ac_b is None or e_a_b is None or e_c_b is None:
                    continue                            # cell B=b empty -> skip
                diff   = (e_ac_b - e_a_b * e_c_b).abs()
                hinge  = hinge + torch.relu(margin - diff)
                n_terms += 1
            hinge = hinge / max(1, n_terms)

            return eq + hinge, n_terms

        return torch.zeros(1), 0

    def calibrate_collider_margins(self, X_full, y_full, fraction=0.5):
        """Set each collider triplet's hinge margin from the ACHIEVABLE dependence.

        The hinge max(0, margin - |dep|) asks the model to induce conditional
        dependence between a collider's endpoints of at least `margin`. A fixed
        margin can exceed the dependence the true collider actually produces
        (which depends on the CPTs and may be smaller than sampling noise),
        making the constraint unsatisfiable and the ALM dual grow without bound.

        We therefore estimate the achievable dependence from the TRUE labels --
        |E[A·C|B=b] - E[A|B=b]·E[C|B=b]| averaged over the populated b cells,
        computed by feeding the one-hot true target through the same expectation
        machinery -- and set each collider's margin to `fraction` of it (floored
        at 0). Data-only: call once on the full training set. Returns the
        {triplet: margin} map (also stored on self).
        """
        X = torch.as_tensor(np.asarray(X_full), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(y_full)).long()
        n = y.shape[0]
        probs = torch.zeros(n, self.n_values)                 # true label one-hot
        probs[torch.arange(n), y.clamp(0, self.n_values - 1)] = 1.0
        val = y.float()                                       # true target values
        margins = {}
        for t in self.triplets:
            if t.type != TripletType.COLLIDER:
                continue
            A, B, C = t.left, t.centre, t.right
            deps = []
            for b_val in range(self.n_values):
                e_ac = self.expectation_product(A, C, X, probs, val, given={B: b_val})
                e_a  = self.expectation(A, X, probs, val, given={B: b_val})
                e_c  = self.expectation(C, X, probs, val, given={B: b_val})
                if e_ac is None or e_a is None or e_c is None:
                    continue
                deps.append((e_ac - e_a * e_c).abs().item())
            emp = sum(deps) / len(deps) if deps else 0.0
            margins[t] = max(0.0, fraction * emp)
        self._collider_margins = margins
        return margins

    def violations_with_terms(self, model, X_batch):
        """[(violation, n_terms), ...] for every triplet, in self.triplets order.

        Same underlying computation as __call__, but also exposes each
        triplet's n_terms (see _constraint_for_triplet) so callers -- notably
        DiscreteRecommenderPredictor.constraint_violation -- can tell a
        genuinely satisfied constraint (n_terms > 0, violation ~ 0) apart from
        one where every conditioning cell was empty on this data (n_terms == 0).
        """
        if not self.triplets:
            return []
        soft_probs = torch.softmax(model(X_batch), dim=1)
        k = torch.arange(soft_probs.shape[1], dtype=soft_probs.dtype)
        soft_val = (soft_probs * k).sum(dim=1)
        return [self._constraint_for_triplet(t, X_batch, soft_probs, soft_val)
                for t in self.triplets]

    def __call__(self, model, X_batch, y_batch=None):
        """Per-triplet constraint violations of `model` on the batch.

        The target variable is taken from the model's predicted class
        distribution (softmax of model(X_batch)) rather than the labels, so the
        violations are differentiable w.r.t. the model parameters and ALM can
        actually steer the predictions. Feature variables come from the data.
        `y_batch` is ignored (kept for call-site compatibility).
        """
        if not self.triplets:
            return torch.zeros(1)
        return torch.cat([v for v, _n in self.violations_with_terms(model, X_batch)])


class DiscreteRecommenderPredictor(CausalConstrainedPredictor):
    """Scikit-learn compatible neural classifier for discrete variable prediction.

    Implements the CausalConstrainedPredictor hooks for the discrete case;
    fit(), constraint_violation(), the feature-restriction baseline, and the
    whole ALM / Moreau-envelope / plain training loop are inherited unchanged
    from the base class (see causal_estimator_base.py for the full hyper-
    parameter reference -- network architecture knobs below are the only ones
    specific to this subclass).

    Training is driven by CrossEntropyLoss. When cfg.use_alm=True and no
    constraints_fn is supplied to the constructor, one is auto-built as a
    TripletConstraintsFn (see _build_constraints_fn).

    NN hyper-parameters (read from cfg, all optional):
        network      (str,   default 'mlp') backbone to use; one of
                     discrete_networks.NETWORK_REGISTRY (mlp, deep_mlp,
                     onehot_mlp, embedding_mlp, transformer). Each backbone
                     reads its own architecture knobs (hidden_dim, n_layers,
                     emb_dim, d_model, ...) from cfg.
    """

    # ------------------------------------------------------------------
    # discrete-specific helper
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_n_values(X_sel, n_classes):
        """Single source of truth for category cardinality.

        The data are ordinal with a cardinality shared across all variables
        (see er_graph.generate_dag), so one integer serves every consumer: the
        feature encoders/embeddings AND the constraint conditioning loops. It is
        the larger of the max category index present in the selected features
        (+1) and the number of target classes, so neither features nor target
        can exceed it. Both fit() and constraint_violation() call this so the
        constraint optimised during training and the one reported afterward loop
        over exactly the same value range.
        """
        X_sel = np.asarray(X_sel)
        feat_max = int(X_sel.max(initial=0)) + 1 if X_sel.size else 1
        return max(feat_max, n_classes)

    # ------------------------------------------------------------------
    # CausalConstrainedPredictor hooks
    # ------------------------------------------------------------------

    def _prepare_target(self, y):
        """Integer-encode the raw labels and remember the class order for
        predict() (self._classes)."""
        y_np = np.asarray(y)
        self._classes = np.unique(y_np)
        label_to_idx = {c: i for i, c in enumerate(self._classes)}
        y_enc = np.array([label_to_idx[c] for c in y_np], dtype=np.int64)
        return torch.tensor(y_enc), len(self._classes)

    def _build_model(self, n_features, n_outputs, X_sel):
        n_values = self._infer_n_values(X_sel, n_outputs)
        network_name = self.cfg.get('network', 'mlp')
        return build_network(network_name, n_features, n_outputs, n_values, self.cfg)

    def _build_criterion(self):
        return nn.CrossEntropyLoss()

    def _build_constraints_fn(self, triplets, kept_features, X_sel, y_t):
        n_values = self._infer_n_values(X_sel, len(self._classes))
        return TripletConstraintsFn(triplets, kept_features, self.target_col, n_values)

    def _eval_constraints_fn(self, triplets, feature_names, X_sel):
        """Rebuild the reporting constraints fn, reusing the fitted
        self._constraints_fn_'s calibrated collider margins when it is a
        TripletConstraintsFn (so training and reporting score the exact same
        constraint); unconstrained models (self._constraints_fn_ is None or a
        custom, uncalibrated constraints_fn) fall back to the default margin.
        """
        n_values = self._infer_n_values(X_sel, len(self._classes))
        fitted_cfn = self._constraints_fn_
        if isinstance(fitted_cfn, TripletConstraintsFn):
            cfn = TripletConstraintsFn(triplets, feature_names, self.target_col,
                                       n_values,
                                       collider_margin=fitted_cfn.collider_margin)
            cfn._collider_margins = dict(fitted_cfn._collider_margins)
        else:
            cfn = TripletConstraintsFn(triplets, feature_names, self.target_col, n_values)
        return cfn

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def predict(self, X):
        X_t = torch.tensor(self._select_features(np.asarray(X)),
                           dtype=torch.float32)
        with torch.no_grad():
            indices = self._rf_model_(X_t).argmax(dim=1).numpy()
        return self._classes[indices]
