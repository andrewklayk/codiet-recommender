import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from causal_triplets import (TripletType, Triplet, get_target_triplets,
                             parent_features, markov_blanket_features,
                             filter_triplets)
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


class DiscreteRecommenderPredictor(BaseEstimator):
    """Scikit-learn compatible neural classifier for discrete variable prediction.

    Accepts the same constructor signature as the other RecommenderPredictor
    classes so it plugs into the existing create_model / run_feature_selection_scikit
    pipeline without modification to those call sites.

    The DAG adjacency matrix (w_est) is stored as _w_est and passed through
    unchanged; DAG recalculation is not performed for discrete data.

    Training is driven by CrossEntropyLoss.  When cfg.use_alm=True and
    constraints_fn is provided, the loop uses ALM from
    humancompatible.train.dual_optim (pip install humancompatible-train).

    Args:
        constraints_fn (callable | None): called as
            constraints_fn(model, X_batch, y_batch) -> 1-D Tensor of constraint
            violations.  Required when cfg.use_alm=True.  This is the hook for
            plugging in humancompatible/train-style constrained training.

    NN hyper-parameters (read from cfg, all optional):
        network      (str,   default 'mlp') backbone to use; one of
                     discrete_networks.NETWORK_REGISTRY (mlp, deep_mlp,
                     onehot_mlp, embedding_mlp, transformer). Each backbone
                     reads its own architecture knobs (hidden_dim, n_layers,
                     emb_dim, d_model, ...) from cfg.
        n_epochs     (int,   default 50)   training epochs
        batch_size   (int,   default 32)   mini-batch size
        learning_rate(float, default 1e-3) Adam learning rate

    Feature-restriction baseline (read from cfg):
        restrict_to_parents (bool, default False) when True, the model is fed
            only the direct causes (parents) of the target in the interaction
            graph w_est; all other columns are dropped.
        restrict_to_markov_blanket (bool, default False) when True, the model is
            fed only the target's Markov blanket (parents + children +
            co-parents) -- the variables with predictive power under
            d-separation. Takes precedence over restrict_to_parents if both set.
        Either restriction combines with use_alm: constraints are then limited to
            triplets whose variables all survive. Orthogonal to use_alm, so all
            (vanilla / constrained) x (all / parents / markov_blanket)
            combinations are available.

    ALM hyper-parameters (read from cfg, used only when use_alm=True):
        use_alm      (bool,  default False)
        n_constraints(int,   default 1)    number of constraints returned by constraints_fn
        alm_lr       (float, default 5.0)  dual variable learning rate. The duals
            are updated ONCE PER EPOCH on the full training set (not per
            minibatch), so with honestly-measured violations the classic small
            step is far too weak; 5.0 matches the Markov-blanket oracle here.
        alm_penalty  (float, default 1.0)  ALM quadratic-penalty coefficient
            (the `penalty` arg of humancompatible ALM).
        alm_slack    (float, default 0.01) tolerance subtracted from the
            full-train violation before the dual update, i.e. duals step on
            (violation - alm_slack). This lets duals DECREASE when violations are
            within estimation noise, breaking the monotone ratchet that arises
            because constraints are nonnegative by construction.
        alm_momentum (float, default 0.5)  dual variable momentum
        collider_margin_fraction (float, default 0.5) sets each collider's hinge
            margin to this fraction of the endpoint dependence achievable in the
            true training labels (see
            TripletConstraintsFn.calibrate_collider_margins), so the constraint
            is not asked to manufacture more dependence than the data holds.
        alm_selection_weight (float, default 1.0) weight on the mean constraint
            violation when picking the best epoch under constrained training;
            the selection score is CE + alm_selection_weight * mean_violation,
            with the violation measured on the full training set (the same
            evaluation used for the dual update). Plain training selects on CE
            alone.

    Moreau-envelope-only hyper-parameters (read from cfg, used only when
    use_moreau=True; ignored if use_alm=True, which already wraps Adam in
    MoreauEnvelope alongside the ALM duals):
        use_moreau   (bool,  default False) wrap Adam in
            humancompatible.train.dual_optim.MoreauEnvelope with no dual
            variables or constraints -- isolates the effect of the optimizer
            alone from the effect of the causal constraint (the "uncon_ME"
            ablation in test_all_networks.py / test_constraints.py /
            test_shift.py).
        moreau_mu    (float, default 2.0)  smoothing multiplier
        moreau_beta  (float, default 0.5)  smoothing multiplier update rate

    Dual-update scheme: per minibatch we backprop the Lagrangian via
    dual.forward(loss, constraints) WITHOUT stepping the duals; once per epoch we
    evaluate the constraints on the whole training set and call
    dual.update(violation - alm_slack). This decouples the (noisy, per-batch)
    primal gradient from the (stable, per-epoch) dual dynamics.
    """

    def __init__(self, w_est, target_col, row_and_col_names, custom_objective,
                 prep_data, cfg, constraints_fn=None):
        self.w_est = w_est
        self.target_col = target_col
        self.row_and_col_names = row_and_col_names
        self.custom_objective = custom_objective
        self.prep_data = prep_data
        self.cfg = cfg
        self.constraints_fn = constraints_fn
        self._rf_model_ = None
        self._constraints_fn_ = None
        self._w_est = w_est
        self._classes = None
        self._feature_mask = None
        self._kept_features = None

    # ------------------------------------------------------------------
    # feature restriction (interaction-graph baseline)
    # ------------------------------------------------------------------

    def _select_features(self, X_arr):
        """Apply the fitted column mask; fall back to a constant column.

        When the target has no retained features (e.g. a root node under
        restrict_to_parents) the mask is empty and we feed a single constant
        feature so the network simply learns the class prior.
        """
        if self._feature_mask:
            return X_arr[:, self._feature_mask]
        return np.zeros((X_arr.shape[0], 1), dtype=X_arr.dtype)

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
    # training
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        # --- resolve hyper-parameters from cfg ---
        # (network architecture knobs like hidden_dim/n_layers are read by the
        #  selected backbone itself in discrete_networks.build_network)
        n_epochs      = self.cfg.get('n_epochs',      50)
        batch_size    = self.cfg.get('batch_size',    32)
        lr            = self.cfg.get('learning_rate', 1e-3)
        use_alm       = self.cfg.get('use_alm',       False)
        alm_lr        = self.cfg.get('alm_lr',        5.0)
        alm_penalty   = self.cfg.get('alm_penalty',   1.0)
        alm_slack     = self.cfg.get('alm_slack',     0.01)
        alm_momentum  = self.cfg.get('alm_momentum',  0.5)
        # Moreau-envelope smoothing on its own (no ALM dual/constraints): isolates
        # the effect of the optimizer alone from the effect of the causal
        # constraints, for the uncon_ME ablation.
        use_moreau    = self.cfg.get('use_moreau',    False)
        moreau_mu     = self.cfg.get('moreau_mu',     2.0)
        moreau_beta   = self.cfg.get('moreau_beta',   0.5)

        # --- resolve the feature-restriction mode (interaction-graph baseline) ---
        # 'markov_blanket' keeps parents + children + co-parents; 'parents' keeps
        # only the direct causes; 'none' keeps everything. Markov blanket takes
        # precedence if both flags are set.
        if self.cfg.get('restrict_to_markov_blanket', False):
            restrict_mode = 'markov_blanket'
        elif self.cfg.get('restrict_to_parents', False):
            restrict_mode = 'parents'
        else:
            restrict_mode = 'none'

        # --- feature restriction ---
        feature_names = (list(X.columns) if hasattr(X, 'columns')
                         else [n for n in self.row_and_col_names
                               if n != self.target_col])
        if restrict_mode == 'markov_blanket':
            kept = markov_blanket_features(self.w_est, self.row_and_col_names,
                                           self.target_col, feature_names)
        elif restrict_mode == 'parents':
            kept = parent_features(self.w_est, self.row_and_col_names,
                                   self.target_col, feature_names)
        else:
            kept = list(feature_names)

        if restrict_mode != 'none':
            kept_set = set(kept)
            self._feature_mask = [i for i, f in enumerate(feature_names)
                                  if f in kept_set]
        else:
            self._feature_mask = list(range(len(feature_names)))
        self._kept_features = kept

        # --- label encoding ---
        y_np = np.asarray(y)
        self._classes = np.unique(y_np)
        label_to_idx = {c: i for i, c in enumerate(self._classes)}
        y_enc = np.array([label_to_idx[c] for c in y_np], dtype=np.int64)

        # --- tensors & dataloader (features restricted to self._feature_mask) ---
        X_sel = self._select_features(np.asarray(X))
        X_t = torch.tensor(X_sel, dtype=torch.float32)
        y_t = torch.tensor(y_enc)
        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=batch_size, shuffle=True)

        n_features = X_t.shape[1]
        n_classes  = len(self._classes)
        # --- single source of truth for category cardinality ---
        # Inferred once here and reused by constraint_violation via the same
        # _infer_n_values helper, so the encoders/embeddings and the constraint
        # conditioning loops all agree (and training matches reporting).
        n_values = self._infer_n_values(X_sel, n_classes)

        # --- resolve the constraints function (never mutate the ctor param) ---
        # When ALM is on and the user passed none, auto-build a
        # TripletConstraintsFn. The result lives in the fitted attribute
        # self._constraints_fn_, leaving the constructor argument
        # self.constraints_fn untouched, so sklearn clone()/get_params and any
        # re-fit keep re-deriving from the original input.
        constraints_fn = self.constraints_fn
        if use_alm and constraints_fn is None:
            triplets = get_target_triplets(self.w_est, self.row_and_col_names,
                                           self.target_col)
            if restrict_mode != 'none':
                # A constraint over an excluded variable is meaningless for a
                # model that never sees it: keep only triplets whose variables
                # all remain (the target plus the retained features).
                allowed = set(kept) | {self.target_col}
                triplets = filter_triplets(triplets, allowed)
            constraints_fn = TripletConstraintsFn(
                triplets, kept, self.target_col, n_values
            )
        self._constraints_fn_ = constraints_fn

        # Calibrate collider hinge margins from the achievable dependence in the
        # TRUE training labels, so the constraint never asks for more conditional
        # dependence than the data actually contains (else the dual diverges).
        if use_alm and isinstance(self._constraints_fn_, TripletConstraintsFn):
            margin_fraction = self.cfg.get('collider_margin_fraction', 0.5)
            self._constraints_fn_.calibrate_collider_margins(
                X_sel, y_enc, margin_fraction)

        # Determine the dual dimension AFTER the constraints function exists, so
        # every constraint it returns gets its own dual variable in the ALM.
        n_constraints = getattr(constraints_fn, 'n_constraints',
                                self.cfg.get('n_constraints', 1))

        # --- build network ---
        network_name = self.cfg.get('network', 'mlp')
        model = build_network(network_name, n_features, n_classes, n_values, self.cfg)
        criterion = nn.CrossEntropyLoss()

        # --- optimiser (plain, Moreau-only, or ALM-wrapped) ---
        if use_alm:
            # humancompatible/train integration:
            # pip install humancompatible-train
            from humancompatible.train.dual_optim import ALM, MoreauEnvelope
            optimizer = MoreauEnvelope(torch.optim.Adam(model.parameters(), lr=lr))
            dual = ALM(m=n_constraints, lr=alm_lr, momentum=alm_momentum,
                       penalty=alm_penalty)
        elif use_moreau:
            # Same optimizer wrapper as ALM training, but with no dual variables
            # or constraints: measures what the Moreau-envelope smoothing alone
            # contributes, decoupled from the causal-constraint penalty.
            from humancompatible.train.dual_optim import MoreauEnvelope
            optimizer = MoreauEnvelope(torch.optim.Adam(model.parameters(), lr=lr),
                                       mu=moreau_mu, beta=moreau_beta)
            dual = None
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            dual = None

        # --- training loop ---
        # We keep the *best* model seen during training rather than the last
        # one: after each epoch the model's cross-entropy on the training set is
        # evaluated (no_grad) and the parameters with the lowest CE are cached.
        # CE is used as the selection criterion for both the plain and the
        # constrained variants because it is the underlying prediction
        # objective and is comparable across epochs (unlike the Lagrangian,
        # whose value shifts as the dual variables update).
        # Constrained training selects on CE + sel_weight * mean violation so the
        # returned epoch is not the one that fits the data hardest while breaking
        # the causal facts. A fixed sel_weight (rather than the shifting dual
        # variables) keeps the score comparable across epochs; plain training
        # falls back to CE alone.
        sel_weight = self.cfg.get('alm_selection_weight', 1.0)
        constrained = dual is not None and self._constraints_fn_ is not None
        best_score = float('inf')
        best_state = copy.deepcopy(model.state_dict())
        for _ in range(n_epochs):
            model.train()
            for X_batch, y_batch in loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                if constrained:
                    # Backprop the Lagrangian with the duals held FIXED for the
                    # epoch (forward, not forward_update): the per-minibatch
                    # constraint estimate is noisy and, being nonnegative, would
                    # ratchet the duals upward every step.
                    constraints = self._constraints_fn_(model, X_batch, y_batch)
                    lagrangian = dual.forward(loss, constraints)
                    lagrangian.backward()
                else:
                    loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            # --- per-epoch: full-train violation drives BOTH the dual update
            #     and the best-epoch selection score ---
            model.eval()
            with torch.no_grad():
                score = criterion(model(X_t), y_t).item()
                if constrained:
                    c_full = self._constraints_fn_(model, X_t)   # full-train
                    # Step the duals on (violation - slack) so they can decrease
                    # when the violation is within estimation noise.
                    dual.update(c_full - alm_slack)
                    score += sel_weight * c_full.mean().item()
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        model.eval()
        self._rf_model_ = model
        self._w_est = self.w_est
        return self

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def predict(self, X):
        X_t = torch.tensor(self._select_features(np.asarray(X)),
                           dtype=torch.float32)
        with torch.no_grad():
            indices = self._rf_model_(X_t).argmax(dim=1).numpy()
        return self._classes[indices]

    def constraint_violation(self, X, aggregate='mean'):
        """How much the fitted model's predictions violate the causal constraints.

        Reuses the very same triplet enumeration (get_target_triplets) as
        constrained training, but substitutes the model's PREDICTED target for
        the true label, so the result measures whether the trained predictor
        respects the conditional-independence constraints implied by w_est.
        Works for any fitted model regardless of cfg.use_alm, enabling a
        constrained-vs-unconstrained comparison.

        If the model was actually trained with ALM, self._constraints_fn_ is
        the TripletConstraintsFn that training optimised, complete with its
        calibrated per-triplet collider margins (see
        TripletConstraintsFn.calibrate_collider_margins). Those SAME margins
        are reused here, so this reports the exact constraint training saw
        rather than silently re-deriving a fresh, uncalibrated one (the fixed
        default collider_margin). Unconstrained models (self._constraints_fn_
        is None) fall back to that default, since there is nothing calibrated
        to reuse.

        X : the same feature frame/array passed to fit/predict (the estimator
            applies its own feature mask internally for prediction; constraint
            expectations use the feature columns present in X).
        aggregate : 'mean' (default) or 'sum', applied to 'total' and to each
            entry of 'by_type'.

        Returns dict:
            total      : float, `aggregate` over ALL triplets combined
                        (pd.NA when no triplet involves the target).
            by_type    : {'chain': float|pd.NA, 'fork': float|pd.NA,
                         'collider': float|pd.NA} -- same `aggregate`, computed
                         separately per causal structure so one structure's
                         violations can't hide in an overall mean; pd.NA for a
                         structure with no triplets here (e.g. after feature
                         restriction).
            per_triplet: list[float], one violation per triplet in `triplets`
                        order (get_target_triplets(), filtered to the model's
                        kept features).
            n_terms    : list[int], parallel to per_triplet -- the number of
                        valid (non-empty) conditioning cells that violation was
                        averaged over. n_terms == 0 means every conditioning
                        cell was empty for this X, so that triplet's near-zero
                        violation reflects missing data, not a satisfied
                        constraint.
            n_triplets : int, len(per_triplet).
        """
        if self._rf_model_ is None:
            raise RuntimeError("call fit() before constraint_violation()")
        # Evaluate over exactly the features the model was trained on (the same
        # construction as fit): triplets whose variables all survive, feature
        # names = the kept columns, X selected through the fitted mask.
        feature_names = list(self._kept_features or [])
        allowed = set(feature_names) | {self.target_col}
        triplets = filter_triplets(
            get_target_triplets(self.w_est, self.row_and_col_names, self.target_col),
            allowed)
        empty_by_type = {tt.value: pd.NA for tt in TripletType}
        if not triplets:
            return {'total': np.nan, 'by_type': empty_by_type,
                    'per_triplet': [], 'n_terms': [], 'n_triplets': 0}
        X_sel = self._select_features(np.asarray(X))
        n_values = self._infer_n_values(X_sel, len(self._classes))

        fitted_cfn = self._constraints_fn_
        if isinstance(fitted_cfn, TripletConstraintsFn):
            cfn = TripletConstraintsFn(triplets, feature_names, self.target_col,
                                       n_values,
                                       collider_margin=fitted_cfn.collider_margin)
            cfn._collider_margins = dict(fitted_cfn._collider_margins)
        else:
            cfn = TripletConstraintsFn(triplets, feature_names, self.target_col, n_values)

        with torch.no_grad():
            pairs = cfn.violations_with_terms(
                self._rf_model_, torch.tensor(X_sel, dtype=torch.float32))
        per_triplet = [float(v.item()) for v, _n in pairs]
        n_terms = [int(n) for _v, n in pairs]

        agg = np.mean if aggregate == 'mean' else np.sum
        total = float(agg(per_triplet))
        by_type = {}
        for tt in TripletType:
            vals = [v for v, t in zip(per_triplet, triplets) if t.type == tt]
            by_type[tt.value] = float(agg(vals)) if vals else pd.NA

        return {'total': total, 'by_type': by_type, 'per_triplet': per_triplet,
                'n_terms': n_terms, 'n_triplets': len(triplets)}