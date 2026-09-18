"""Estimator-independent ALM / Moreau-envelope constrained training loop.

CausalConstrainedPredictor factors out everything about fitting a causally
constrained neural predictor that does NOT depend on the concrete backbone,
loss, or target encoding: hyper-parameter resolution, the parent / Markov-
blanket feature-restriction baseline, the DataLoader, auto-building and
calibrating the constraints function, choosing the optimiser (plain Adam /
Moreau-envelope-only / ALM), the training loop itself, and constraint_violation
reporting. Subclasses (discrete_estimator.DiscreteRecommenderPredictor now, a
planned continuous estimator later) plug in a handful of hooks -- target
encoding, model/criterion/constraints-fn construction, and predict() -- and
get all of the above for free.

Builds on causal_triplets.py for the graph-only triplet logic (this module
adds the estimator/data side: fitting a model against those triplets).
"""
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator

from causal_triplets import (TripletType, get_target_triplets, parent_features,
                             markov_blanket_features, filter_triplets)


class CausalConstrainedPredictor(BaseEstimator):
    """Base class for scikit-learn compatible, causally constrained predictors.

    Accepts the same constructor signature as the other RecommenderPredictor
    classes so subclasses plug into the existing create_model /
    run_feature_selection_scikit pipeline without modification to those call
    sites.

    The DAG adjacency matrix (w_est) is stored as _w_est and passed through
    unchanged; DAG recalculation is not performed here.

    Training is driven by whatever criterion _build_criterion() returns. When
    cfg.use_alm=True and constraints_fn is provided (or auto-built via
    _build_constraints_fn), the loop uses ALM from
    humancompatible.train.dual_optim (pip install humancompatible-train).

    Args:
        constraints_fn (callable | None): called as
            constraints_fn(model, X_batch, y_batch) -> 1-D Tensor of constraint
            violations.  Required when cfg.use_alm=True.  This is the hook for
            plugging in humancompatible/train-style constrained training.

    NN hyper-parameters (read from cfg, all optional):
        n_epochs     (int,   default 50)   training epochs
        batch_size   (int,   default 32)   mini-batch size
        learning_rate(float, default 1e-3) Adam learning rate
        val_fraction (float, default 0.0)  opt-in: fraction of the training
            data held out (once, before the DataLoader/constraints_fn/dual are
            built) and used ONLY for the per-epoch best_epoch_ selection score
            -- never for gradient steps, constraints-function calibration, or
            the ALM dual update, which all keep operating on the remaining
            training split exactly as before. 0.0 (default) selects on the
            training split itself, as before. Off by default so existing
            configs keep behaving exactly as before.

            Why this matters: without it, an unconstrained model selects
            purely on training loss (which keeps improving as it overfits),
            while a constrained model selects on training loss PLUS the
            (ALM-driven, shrinking) violation term -- two different signals,
            so comparing con_* against uncon_* conflates "the causal
            constraint helps generalization" with "the violation term
            happened to nudge the selected epoch earlier." Setting
            val_fraction > 0 puts both arms' best_epoch_ selection on the same
            footing: held-out predictive loss.

            Known limitation: the split happens AFTER _prepare_features/
            _prepare_target's own fitting (StandardScaler stats, target
            mean/std for the continuous estimator), so those statistics are
            still computed over train+val combined, not train-only -- a
            small leak of feature/target *scale* (not labels) into the
            validation split. Splitting cleanly would need those hooks to
            support a separate fit-on-train/transform-on-val step; not worth
            the extra interface surface unless it turns out to matter.

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
            violation BOTH in the per-minibatch primal Lagrangian (dual.forward)
            and in the once-per-epoch dual update, i.e. every constraint value
            either ever sees is (violation - alm_slack), never the raw
            violation. This lets duals DECREASE when violations are within
            estimation noise, breaking the monotone ratchet that arises because
            constraints are nonnegative by construction -- and, just as
            importantly, it is what makes alm_slack actually mean "this much
            violation is tolerated for free": under the "hpr" augmentation
            (alm_penalty>0) the primal step's weight on d(violation)/d(theta) is
            max(0, lambda + alm_penalty*c), which for an un-slacked c is > 0 on
            almost every batch (violations are one-sided) even while lambda == 0
            -- i.e. without this offset the primal step pulls toward zero
            violation regardless of alm_slack, and only the mostly-idle dual
            update would ever see the configured tolerance.
        alm_momentum (float, default 0.5)  dual variable momentum
        collider_margin_fraction (float, default 0.5) forwarded to the
            constraints function's calibrate_collider_margins, when it has one
            (see the _build_constraints_fn hook docstring).
        calibrate_slacks (bool, default False) opt-in: when True AND the
            constraints function implements calibrate_slacks (duck-typed via
            hasattr, like calibrate_collider_margins above), replace the
            single global alm_slack with a PER-TRIPLET slack derived from
            each triplet's achievable (true-label) violation floor on the
            training set -- see TripletConstraintsFn.calibrate_slacks and
            calibrate_slacks.ipynb, which found some triplets (discrete
            chains in particular) with a floor well above the global
            default, leaving their dual with no fixed point. Off by default
            so existing configs keep behaving exactly as before.
        slack_calibration_fraction (float, default 1.2) forwarded to
            calibrate_slacks as `fraction`: each triplet's slack is set to
            `fraction * floor` (>= 1.0, unlike collider_margin_fraction,
            since slack should sit above the floor as headroom rather than
            below it as a minimum ask).
        alm_selection_weight (float, default 1.0) weight on the mean constraint
            violation when picking the best epoch under constrained training;
            the selection score is _selection_loss(...) + alm_selection_weight *
            mean_violation. The violation term is always measured on the full
            TRAINING split (the same evaluation used for the dual update,
            unaffected by val_fraction) -- constraint satisfaction isn't a
            generalization question the way predictive loss is. Only the
            _selection_loss(...) part switches to the held-out validation
            split when val_fraction > 0. Plain training uses _selection_loss(...)
            alone (also on the validation split when val_fraction > 0).

    Moreau-envelope hyper-parameters (read from cfg; moreau_mu / moreau_beta
    apply to BOTH the use_moreau=True arm and the use_alm=True arm, which wraps
    Adam in the same MoreauEnvelope alongside the ALM duals -- keeping the two
    arms on one optimizer is what makes uncon_ME a valid control for con_*):
        use_moreau   (bool,  default False) wrap Adam in
            humancompatible.train.dual_optim.MoreauEnvelope with no dual
            variables or constraints -- isolates the effect of the optimizer
            alone from the effect of the causal constraint (the "uncon_ME"
            ablation in test_all_networks.py / test_constraints.py /
            test_shift.py).
        moreau_mu    (float, default 2.0)  smoothing multiplier
        moreau_beta  (float, default 0.5)  smoothing multiplier update rate

    Dual-update scheme: per minibatch we backprop the Lagrangian via
    dual.forward(loss, constraints - alm_slack) WITHOUT stepping the duals; once
    per epoch we evaluate the constraints on the whole training set and call
    dual.update(violation - alm_slack). This decouples the (noisy, per-batch)
    primal gradient from the (stable, per-epoch) dual dynamics, while keeping
    both estimates of the SAME slack-adjusted quantity -- only the data they're
    computed over (one minibatch vs. the whole training set) differs.

    Fitted diagnostics attributes (set by fit(), read-only afterwards):
        train_history_    (list[dict]) one entry per epoch: always
            {'epoch', 'selection_loss'}; when use_alm=True (constrained)
            each entry also carries 'violation_mean', 'dual_mean', 'dual_max'
            and 'n_duals_saturated' -- the full-train mean constraint
            violation and the ALM dual variables' state right after that
            epoch's dual.update(). 'n_duals_saturated' counts duals sitting
            at the optimizer's own clamp (ALM's dual_range upper bound,
            100.0 by default): a dual pinned there for many consecutive
            epochs means that constraint's achievable violation floor sits
            above alm_slack, so the plain/HPR dual update has no fixed point
            below the clamp -- it will keep climbing for as long as training
            runs, and the resulting Σλc penalty term can swamp the
            prediction loss entirely (see CONSTRAINTS.md's worked example).
        best_epoch_       (int) the epoch (0-indexed) whose state_dict was
            kept as best_state -- if this equals n_epochs_run_ - 1, training
            was still improving when it stopped and more epochs might help;
            if it is much earlier, later epochs were actively getting worse
            (e.g. a still-climbing dual dragging the model away from a fit
            it already had).
        n_epochs_run_     (int) the n_epochs actually used this fit() call
            (after any cfg override), for comparing against best_epoch_.
        alm_slack_        (Tensor[n_constraints] | float | None) the actual
            slack used in this fit() call: per-triplet when calibrate_slacks
            was on and the constraints function supports it, else the plain
            cfg.alm_slack scalar broadcast; None when unconstrained.
        n_val_            (int) number of samples held out for best_epoch_
            selection (0 when val_fraction is 0 -- selection then reused the
            training split itself, as before).
        _train_row_mask_  (bool ndarray, len == the X this fit() call
            received) True for rows actually used for gradient training,
            False for any val_fraction held-out rows (all-True when
            val_fraction is 0). Not a "diagnostic" to read directly --
            consumed by run_feature_selection_scikit to score train_error
            only on the rows this model actually trained on, rather than on
            the whole cross-validation fold (which would otherwise silently
            include the held-out rows and understate any train-test
            comparison, e.g. shift = test_error - train_error).
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
        self._feature_mask = None
        self._kept_features = None

    # ------------------------------------------------------------------
    # feature restriction (interaction-graph baseline)
    # ------------------------------------------------------------------

    def _select_features(self, X_arr, fitting=False):
        """Apply the fitted column mask, then the subclass's own feature
        preparation hook; fall back to a constant column when the mask is empty.

        When the target has no retained features (e.g. a root node under
        restrict_to_parents) the mask is empty and we feed a single constant
        feature so the network simply learns the class prior.

        fitting is forwarded to _prepare_features (True only when called from
        fit(), so e.g. a StandardScaler fits its stats once on the training
        set and only transforms afterward).
        """
        if self._feature_mask:
            X_masked = X_arr[:, self._feature_mask]
        else:
            X_masked = np.zeros((X_arr.shape[0], 1), dtype=X_arr.dtype)
        return self._prepare_features(X_masked, fitting)

    # ------------------------------------------------------------------
    # subclass hooks
    # ------------------------------------------------------------------
    # A concrete predictor implements these; CausalConstrainedPredictor never
    # imports a concrete backbone, loss, or constraints-function class, so
    # this module stays independent of any one estimator (discrete now,
    # continuous later).

    def _prepare_target(self, y):
        """Encode the raw target `y` into a training tensor.

        Returns (y_tensor, n_outputs): y_tensor is fed to the DataLoader
        alongside X_t; n_outputs is the model's output width (n_classes for a
        classifier, 1 for scalar regression, ...). Subclasses may stash
        whatever bookkeeping predict() needs (e.g. the discrete estimator's
        self._classes) as a side effect here.
        """
        raise NotImplementedError

    def _prepare_features(self, X_sel, fitting):
        """Optional hook: transform the already feature-restricted array
        before it becomes a training/eval tensor (e.g. StandardScaler for a
        continuous estimator). Called from _select_features, so it runs
        identically in fit(), constraint_violation(), and predict().

        fitting=True only when called from fit() (fit_transform semantics);
        False everywhere else (transform only, reusing stats fit during
        training). Default: identity -- a classifier over raw category
        indices (the discrete estimator) needs no scaling.
        """
        return X_sel

    def _build_model(self, n_features, n_outputs, X_sel):
        """Instantiate the backbone network (an nn.Module).

        X_sel is the already feature-restricted training array, passed
        through in case a subclass needs to inspect the data itself (e.g. the
        discrete estimator's category cardinality).
        """
        raise NotImplementedError

    def _build_criterion(self):
        """Return the (stateless) loss callable: criterion(logits, y_t) -> scalar tensor."""
        raise NotImplementedError

    def _build_constraints_fn(self, triplets, kept_features, X_sel, y_t):
        """Build the constraints-function object used for ALM training.

        Called only when cfg.use_alm=True and no constraints_fn was supplied
        to the constructor; `triplets` is already filtered to the kept
        features. Must return an object usable as
        constraints_fn(model, X_batch, y_batch) -> 1-D Tensor of violations,
        and MAY additionally implement calibrate_collider_margins(X, y,
        fraction). fit() probes for that method via a hasattr guard (never
        isinstance), so this base module never needs to import a concrete
        constraints-function class.
        """
        raise NotImplementedError

    def _eval_constraints_fn(self, triplets, feature_names, X_sel):
        """Build the constraints-function object used for POST-HOC reporting
        in constraint_violation().

        Should reuse whatever calibration the fitted self._constraints_fn_
        carries (e.g. calibrated collider margins) when available, so training
        and reporting score the exact same constraint rather than silently
        re-deriving an uncalibrated one. Must return an object with
        violations_with_terms(model, X_batch) -> list[(violation, n_terms)].
        """
        raise NotImplementedError

    def _selection_loss(self, model, X_t, y_t):
        """Score used to pick the best epoch (lower is better).

        Default: the plain criterion value, matching plain (unconstrained)
        training's selection rule. Constrained training adds
        alm_selection_weight * mean_violation on top of this in fit() itself.

        Called on the training split when cfg.val_fraction is 0 (default), or
        the held-out validation split when it's > 0 -- either way, whatever
        fit() passes as X_t/y_t here (see val_fraction in the class docstring).
        """
        criterion = self._build_criterion()
        return criterion(model(X_t), y_t).item()

    def predict(self, X):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        # --- resolve hyper-parameters from cfg ---
        # (network architecture knobs like hidden_dim/n_layers are read by the
        #  subclass's own _build_model, e.g. discrete_networks.build_network)
        n_epochs      = self.cfg.get('n_epochs',      50)
        batch_size    = self.cfg.get('batch_size',    32)
        lr            = self.cfg.get('learning_rate', 1e-3)
        val_fraction  = self.cfg.get('val_fraction',   0.0)
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

        # --- target encoding (hook) ---
        y_t_full, n_outputs = self._prepare_target(y)

        # --- tensors (features restricted to self._feature_mask) ---
        X_sel_full = self._select_features(np.asarray(X), fitting=True)

        # --- opt-in held-out split for best_epoch_ selection (val_fraction) ---
        # Split BEFORE anything else is built, so the DataLoader, constraints
        # function (and its collider-margin / slack calibration), and the
        # ALM's full-train dual update all keep operating on the TRAINING
        # split only, exactly as before -- X_t/y_t below still mean "the data
        # this fit() call trains on". X_val_t/y_val_t are used ONLY for the
        # epoch-selection score below; see val_fraction in the class docstring
        # for why this needs to be a genuine held-out split rather than
        # reusing X_t/y_t for that too.
        n_samples = X_sel_full.shape[0]
        if val_fraction > 0 and n_samples > 1:
            n_val = min(n_samples - 1, max(1, int(round(n_samples * val_fraction))))
            perm = torch.randperm(n_samples).numpy()
            val_idx, train_idx = perm[:n_val], perm[n_val:]
        else:
            n_val = 0
            train_idx = np.arange(n_samples)
            val_idx = train_idx  # no split: selection reuses the training data

        # Exposed so a caller scoring THIS SAME X (e.g. run_feature_selection_
        # scikit's per-fold train_error, right after cross_validate hands back
        # the exact X[train_idx] this fit() call received) can restrict to the
        # rows actually used for gradient training, rather than scoring the
        # whole fold including the val_fraction rows above. All-True (a no-op
        # when applied) whenever val_fraction is 0.
        self._train_row_mask_ = np.zeros(n_samples, dtype=bool)
        self._train_row_mask_[train_idx] = True

        X_sel = X_sel_full[train_idx]
        y_t = y_t_full[train_idx]
        X_t = torch.tensor(X_sel, dtype=torch.float32)
        X_val_t = torch.tensor(X_sel_full[val_idx], dtype=torch.float32)
        y_val_t = y_t_full[val_idx]

        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=batch_size, shuffle=True)

        n_features = X_t.shape[1]

        # --- resolve the constraints function (never mutate the ctor param) ---
        # When ALM is on and the user passed none, auto-build one via the
        # subclass's _build_constraints_fn. The result lives in the fitted
        # attribute self._constraints_fn_, leaving the constructor argument
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
            # X_sel_full/y_t_full (not the train-only split), since this hook
            # may infer structural facts (e.g. category cardinality) that the
            # model needs to handle ANY row it's later evaluated on, including
            # the held-out validation split and eventual test data -- sizing
            # it off a shrunk training-only split risks under-counting a
            # category that only appears in the split-out rows.
            constraints_fn = self._build_constraints_fn(triplets, kept, X_sel_full, y_t_full)
        self._constraints_fn_ = constraints_fn

        # Calibrate collider hinge margins from the achievable dependence in the
        # TRUE training labels, so the constraint never asks for more conditional
        # dependence than the data actually contains (else the dual diverges).
        # hasattr, not isinstance: this base module never imports a concrete
        # constraints-function class, so it duck-types the optional method.
        if use_alm and hasattr(self._constraints_fn_, 'calibrate_collider_margins'):
            margin_fraction = self.cfg.get('collider_margin_fraction', 0.5)
            self._constraints_fn_.calibrate_collider_margins(
                X_sel, y_t, margin_fraction)

        # Optionally recalibrate alm_slack itself, per triplet, from each
        # triplet's achievable (true-label) violation floor -- generalizes the
        # collider-margin calibration above to the slack tolerance every
        # constraint is subject to. Run AFTER calibrate_collider_margins so a
        # collider's floor already reflects its calibrated hinge margin. Off
        # by default (calibrate_slacks cfg flag); hasattr, not isinstance,
        # matching the collider-margin probe.
        if (use_alm and self.cfg.get('calibrate_slacks', False)
                and hasattr(self._constraints_fn_, 'calibrate_slacks')):
            slack_fraction = self.cfg.get('slack_calibration_fraction', 1.2)
            self._constraints_fn_.calibrate_slacks(X_sel, y_t, slack_fraction)

        # Resolve the slack actually used below: per-triplet when calibration
        # ran and the constraints function exposes it, else the plain global
        # scalar -- both broadcast the same way against the n_constraints-long
        # violation vector, so the training loop below doesn't need to care
        # which one it got.
        if use_alm and hasattr(self._constraints_fn_, 'slack_vector'):
            alm_slack_use = self._constraints_fn_.slack_vector(default=alm_slack)
        else:
            alm_slack_use = alm_slack

        # Determine the dual dimension AFTER the constraints function exists, so
        # every constraint it returns gets its own dual variable in the ALM.
        n_constraints = getattr(constraints_fn, 'n_constraints',
                                self.cfg.get('n_constraints', 1))

        # --- build network (hook) ---
        # X_sel_full, same reasoning as _build_constraints_fn above: the
        # network's architecture (e.g. embedding table sizes) must fit every
        # row it will ever see, not just the training-only split.
        model = self._build_model(n_features, n_outputs, X_sel_full)
        criterion = self._build_criterion()

        # --- optimiser (plain, Moreau-only, or ALM-wrapped) ---
        if use_alm:
            # humancompatible/train integration:
            # pip install humancompatible-train
            # The SAME MoreauEnvelope smoothing (same mu/beta, read from cfg) as
            # the use_moreau branch below -- otherwise uncon_ME would not be a
            # valid control for this arm: any difference between them would mix
            # "the causal constraint" with "a differently-tuned optimizer".
            from humancompatible.train.dual_optim import ALM, MoreauEnvelope
            # optimizer = MoreauEnvelope(torch.optim.Adam(model.parameters(), lr=lr),
            #                            mu=moreau_mu, beta=moreau_beta)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            dual = ALM(m=n_constraints, lr=alm_lr, momentum=alm_momentum,
                       penalty=alm_penalty, is_ineq=True)
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
        # one: after each epoch the model's selection score is evaluated
        # (no_grad, on X_val_t/y_val_t -- the held-out split when
        # val_fraction > 0, else the training split itself) and the
        # parameters with the lowest score are cached. _selection_loss is
        # used as the selection criterion for both the plain and the
        # constrained variants because it is the underlying prediction
        # objective and is comparable across epochs (unlike the Lagrangian,
        # whose value shifts as the dual variables update). Constrained
        # training selects on _selection_loss + sel_weight * mean violation
        # (violation always measured on the training split -- see
        # val_fraction in the class docstring) so the returned epoch is not
        # the one that fits the data hardest while breaking the causal facts.
        # A fixed sel_weight (rather than the shifting dual variables) keeps
        # the score comparable across epochs; plain training falls back to
        # _selection_loss alone.
        sel_weight = self.cfg.get('alm_selection_weight', 1.0)
        constrained = dual is not None and self._constraints_fn_ is not None
        best_score = float('inf')
        best_epoch = -1
        best_state = copy.deepcopy(model.state_dict())
        # Per-epoch diagnostics (self.train_history_ below): NOT used for
        # training itself, only for post-hoc inspection -- e.g. whether a
        # dual variable is still climbing (or pinned at the ALM's own clamp)
        # when training stops, the signature of a triplet whose achievable
        # violation floor sits above alm_slack and can never satisfy the
        # dual update's implicit target of exactly zero. See CONSTRAINTS.md.
        history = []
        for epoch in range(n_epochs):
            model.train()
            for X_batch, y_batch in loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                if constrained:
                    # Backprop the Lagrangian with the duals held FIXED for the
                    # epoch (forward, not forward_update): the per-minibatch
                    # constraint estimate is noisy and, being nonnegative, would
                    # ratchet the duals upward every step.
                    constraints = self._constraints_fn_(model, X_batch, y_batch) - alm_slack_use
                    lagrangian = dual.forward(loss, constraints)
                    lagrangian.backward()
                else:
                    loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            # --- per-epoch: full-train violation drives the dual update;
            #     the (held-out, when val_fraction > 0) predictive score
            #     drives best-epoch selection ---
            model.eval()
            with torch.no_grad():
                score = self._selection_loss(model, X_val_t, y_val_t)
                entry = {'epoch': epoch, 'selection_loss': score}
                if constrained:
                    c_full = self._constraints_fn_(model, X_t)   # full-train
                    # Step the duals on (violation - slack) so they can decrease
                    # when the violation is within estimation noise.
                    dual.update(c_full - alm_slack_use)
                    score += sel_weight * c_full.mean().item()
                    lam = dual.param_groups[0]['params'][0].data
                    upper = dual.param_groups[0].get('upper_bound')
                    entry['violation_mean'] = c_full.mean().item()
                    entry['dual_mean'] = lam.mean().item()
                    entry['dual_max'] = lam.max().item()
                    entry['n_duals_saturated'] = (
                        int((lam >= upper - 1e-9).sum().item())
                        if upper is not None else 0)
            history.append(entry)
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        model.eval()
        self._rf_model_ = model
        self._w_est = self.w_est
        # Fitted diagnostics attributes -- populated for every fit() call
        # (train_history_ entries only carry the dual/violation keys when
        # constrained is True): best_epoch_ is the epoch fit() actually kept
        # (== n_epochs_run_ - 1 if training was still improving when it
        # stopped -- worth checking against n_epochs_run_ before assuming
        # more epochs wouldn't help); train_history_ is the full per-epoch
        # trace, so a dual's trajectory can be inspected without re-fitting.
        self.train_history_ = history
        self.best_epoch_ = best_epoch
        self.n_epochs_run_ = n_epochs
        self.alm_slack_ = alm_slack_use if constrained else None
        self.n_val_ = n_val
        return self

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def constraint_violation(self, X, aggregate='mean'):
        """How much the fitted model's predictions violate the causal constraints.

        Reuses the very same triplet enumeration (get_target_triplets) as
        constrained training, but substitutes the model's PREDICTED target for
        the true label, so the result measures whether the trained predictor
        respects the conditional-independence constraints implied by w_est.
        Works for any fitted model regardless of cfg.use_alm, enabling a
        constrained-vs-unconstrained comparison.

        The constraints-function object used to score this is built by the
        _eval_constraints_fn hook, which is expected to reuse whatever
        calibration (e.g. collider margins) the fitted self._constraints_fn_
        carries -- so this reports the exact constraint training saw rather
        than silently re-deriving a fresh, uncalibrated one.

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

        cfn = self._eval_constraints_fn(triplets, feature_names, X_sel)

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
