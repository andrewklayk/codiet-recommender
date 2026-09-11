"""Continuous (regression) sibling of DiscreteRecommenderPredictor, sharing the
ALM / Moreau-envelope training loop via CausalConstrainedPredictor.

ContinuousTripletConstraintsFn scores the same chain/fork/collider causal
facts as the discrete TripletConstraintsFn, but over standardized continuous
data: conditional independence A _||_ C | B is proxied by PARTIAL CORRELATION
(not covariance -- see its docstring for why that distinction is load-
bearing), computed via ridge-regularized linear residualization
(Frisch-Waugh-Lovell). The target variable is the model's own prediction
(differentiable), exactly as in the discrete case.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import continuous_networks
from causal_estimator_base import CausalConstrainedPredictor
from causal_triplets import TripletType


class ContinuousTripletConstraintsFn:
    """Constraint function for ALM training, one violation per causal triplet,
    over standardized continuous data.

    Same public API as discrete_estimator.TripletConstraintsFn: n_constraints,
    violations_with_terms(model, X_batch) -> [(violation, n_terms), ...],
    __call__(model, X_batch, y_batch=None) -> 1-D Tensor,
    calibrate_collider_margins(X, y, fraction).

    Math (everything below operates on standardized data; the target variable
    is taken as the model's own prediction y_hat -- shape (batch,), NOT
    (batch, 1) -- wherever it appears in a triplet, exactly like the discrete
    version's soft_val):

        Phi(b)       = [1, phi(b)]          cond_basis: 'linear' -> phi(b)=[b]
                                             'poly:k'   -> phi(b)=[b,b^2,...,b^k]
        residual(v)  = v - Phi @ solve(PhiᵀPhi + ridge·I, Phiᵀv)   (FWL; Phi=[1]
                        alone, i.e. mean-centering, when there is no
                        conditioning variable -- the marginal/eq case below)
        rho(u,v|b)   = <r_u,r_v> / (||r_u||·||r_v|| + delta·||u-mean(u)||·||v-mean(v)|| + 1e-30)
        rho2(u,v|b)  = relu( (rho(u,v|b)^2 · nu - 1) / (nu - 1) ),  nu = n - p
                        (p = number of Phi columns: 1 for the marginal/eq
                        case, 1+dim(phi(b)) when conditioning)

        rho2 is the SQUARED partial correlation, bias-corrected for the
        upward small-sample bias of sample R^2 under the null (E[rho^2] ≈
        1/nu when the true correlation is 0 -- see the "small-batch bias"
        note below); relu clips the rare negative dip from sampling noise.
        The debias step is affine in rho^2, so differentiability and scale
        invariance both survive it unchanged.

        CHAIN and FORK both encode A _||_ C | centre (a chain and a fork over
        the same three roles license the identical conditional-independence
        fact, just via different DAG structures); on standardized data with a
        (possibly nonlinear, via cond_basis) linear conditioning basis they
        collapse to the SAME test, so both use:
            g = rho2(A, C | centre),  n_terms = 1

        COLLIDER:
            eq    = rho2(A, C | ∅) -- ONLY when the target is an ENDPOINT
                    (A or C); when the target is the centre, both endpoints
                    are data columns, so this term is a constant fact about
                    the data (zero gradient) and is skipped, exactly as in
                    the discrete version.
            hinge = max(0, margin_t - sqrt(rho2(A, C | centre) + 1e-12))
                    (the +1e-12 keeps sqrt's gradient finite at rho2 == 0;
                    calibrate_collider_margins uses this exact same sqrt(rho2
                    + 1e-12) expression for margin_t, so hinge and margin are
                    always on the same scale)
            g = eq + hinge,  n_terms = 1 if rho2(A, C | centre) was
                computable, else 0 (eq does not count toward n_terms,
                matching the discrete version's rule that n_terms tracks
                only the hinge / inequality cells, never the marginal
                equality term).

        When the target IS the centre, Phi is built from y_hat, so the FWL
        projector depends on theta; backprop through torch.linalg.solve is
        well-defined as long as ridge > 0 keeps PhiᵀPhi + ridge·I invertible.

    Guards -- the ONLY two cases where a triplet's violation is reported as
    UNCOMPUTABLE (n_terms = 0; g = 0, or eq-only for a collider) rather than
    computed and genuinely near zero:
        - a needed variable is absent from the batch (e.g. after feature
          restriction);
        - a DATA (non-target) variable is constant in this batch, scale-
          relatively: ||v - mean(v)|| <= 1e-6 · ||v|| -- there is nothing to
          correlate a constant against. NOT an absolute threshold: an
          absolute cutoff on a residual NORM changes meaning with batch size
          (||r|| scales like sqrt(n) even for pure noise) and with the
          model's output scale, so it is neither a valid "this is degenerate"
          test nor batch-size invariant.
    The TARGET's own value collapsing to (near-)constant, or being fully
    explained by the conditioning variable, is deliberately NOT guarded: rho
    then goes smoothly (and differentiably) to 0 through the scale-relative
    denominator above, which is the mathematically CORRECT answer -- a
    prediction with no residual variation is trivially uncorrelated with
    anything -- not an unmeasurable one. Getting this backwards previously
    let a collapsed model dodge the collider hinge entirely (constant output
    -> "can't compute" -> 0 penalty, when zero conditional dependence between
    a collider's endpoints is actually the WORST case: hinge should equal the
    full margin).

    Small-batch bias: the raw (pre-debias) sample rho^2 is upward-biased
    under the null (true partial correlation 0) by approximately 1/nu at
    typical minibatch sizes -- e.g. mean rho^2 ≈ 0.036 at batch=32, nu=30, vs.
    the ALM dual only seeing the true (near-zero) violation once per epoch on
    the full training set. The debias term above corrects this so the
    per-minibatch primal gradient is not chasing sampling noise of that
    magnitude.

    Two properties that must NOT be broken by a future edit:

    1. CORRELATION, not covariance. Cov(A, c*y_hat | B) = c*Cov(A, y_hat | B),
       so under a covariance-based penalty ALM could shrink the violation
       arbitrarily just by shrinking the model's output scale -- and as the
       duals grow this becomes actively PROFITABLE, not just harmless. Under
       rho, the invariance to a uniform rescaling y_hat -> c*y_hat (c != 0) is
       EXACT, not approximate: every term of both the numerator and the
       denominator (||r_u||·||r_v|| AND delta·||u-mean(u)||·||v-mean(v)||)
       scales by the same factor |c| when y_hat is one of u/v, so it cancels
       in the ratio exactly. (The old absolute "+ eps" denominator broke this:
       measured deviation at scale=1e-4 was 9.8e-3, not 0.) rho2's debiasing
       is affine in rho^2, so it inherits the same exact invariance.

    2. The n_terms contract from the discrete version: n_terms == 0 means
       "could not be computed", NOT "the constraint holds" -- see "Guards"
       above for exactly which two cases that now covers.

    conditioner selects how the conditioning set is modelled; only 'linear'
    (ridge-regularized linear residualization, as above) is implemented.
    'rank' and 'kernel' are follow-up tiers -- deliberately unimplemented so
    the seam is visible rather than silently falling back to something else.

    Args:
        triplets     : output of causal_triplets.get_target_triplets()
        feature_names: ordered column names of X_batch passed to __call__
        target_col   : name of the target variable (supplied by the model)
        cond_basis   : 'linear' or 'poly:k' (k = polynomial degree)
        ridge        : ridge penalty added to PhiᵀPhi before solving (default
                       1e-4 -- just enough for numerical invertibility)
        collider_margin: fallback hinge margin (overridden per-triplet by
                       calibrate_collider_margins when available)
        conditioner  : 'linear' (default); 'rank'/'kernel' raise NotImplementedError
        delta        : scale-relative regularizer for the rho denominator
                       (default 1e-6) -- about 6 orders of magnitude below a
                       typical ||r_u||·||r_v||, so it is inert in the normal
                       case and only takes over as a residual vanishes,
                       giving rho -> 0 smoothly (with a well-defined
                       gradient) instead of an unstable/undefined 0/0.
    """

    _CONST_REL_TOL = 1e-6   # scale-relative constancy threshold, see Guards above

    def __init__(self, triplets, feature_names, target_col, cond_basis='linear',
                 ridge=1e-4, collider_margin=0.1, conditioner='linear', delta=1e-6):
        if conditioner != 'linear':
            raise NotImplementedError(
                f"conditioner={conditioner!r} is not implemented yet -- only "
                "'linear' (ridge-regularized linear residualization) is "
                "available. 'rank' and 'kernel' are follow-up tiers."
            )
        self.triplets = triplets
        self.feature_names = list(feature_names)
        self.target_col = target_col
        self.cond_basis = cond_basis
        self.ridge = ridge
        self.collider_margin = collider_margin
        self.conditioner = conditioner
        self.delta = delta
        # per-triplet hinge margins, keyed by the (frozen) Triplet; set by
        # calibrate_collider_margins(). Empty -> fall back to collider_margin.
        self._collider_margins = {}
        self._col_idx = {name: i for i, name in enumerate(self.feature_names)}

    @property
    def n_constraints(self):
        return max(1, len(self.triplets))

    def _value(self, name, X, y_hat):
        """Per-sample value vector for `name`.

        Feature -> its (standardized) data column. Target -> the model's own
        prediction y_hat, shape (batch,). None if the variable is absent from
        the batch.
        """
        if name == self.target_col:
            return y_hat
        idx = self._col_idx.get(name)
        return X[:, idx].float() if idx is not None else None

    def _phi(self, b):
        """Column-stack basis expansion of the 1-D conditioning vector `b`."""
        if self.cond_basis == 'linear':
            return b.unsqueeze(1)
        if self.cond_basis.startswith('poly:'):
            k = int(self.cond_basis.split(':', 1)[1])
            return torch.stack([b ** p for p in range(1, k + 1)], dim=1)
        raise ValueError(f"Unknown cond_basis {self.cond_basis!r}")

    def _residual(self, v, b):
        """Ridge-regularized regression residual of v on Phi=[1, phi(b)].

        b=None means no conditioning variable: Phi is just the intercept
        column, so this reduces to (ridge-regularized) mean-centering -- the
        marginal case used by the collider's eq term.
        """
        n = v.shape[0]
        cols = [torch.ones(n, 1, dtype=v.dtype)]
        if b is not None:
            cols.append(self._phi(b))
        Phi = torch.cat(cols, dim=1)
        PtP = Phi.T @ Phi
        ridge_eye = self.ridge * torch.eye(PtP.shape[0], dtype=v.dtype)
        beta = torch.linalg.solve(PtP + ridge_eye, Phi.T @ v)
        return v - Phi @ beta

    @staticmethod
    def _is_effectively_constant(v, rel_tol=_CONST_REL_TOL):
        """Scale-relative constancy test -- see the class docstring's
        "Guards" section for why this must be relative, not absolute."""
        return ((v - v.mean()).norm() <= rel_tol * v.norm()).item()

    def _pcorr2(self, name1, name2, X, y_hat, cond_name=None):
        """Debiased, squared partial correlation rho2(name1, name2 | cond_name)
        (None => marginal, i.e. conditioning is just the intercept / mean-
        centering). See the class docstring's Math and Guards sections.

        Returns None -- "could not be computed", never a phantom zero -- only
        when a named variable is absent from the batch, or when a DATA (non-
        target) variable is constant in this batch. The target's own value
        collapsing is NOT guarded: rho2 goes smoothly (and correctly) to 0
        through the scale-relative denominator instead.
        """
        u = self._value(name1, X, y_hat)
        v = self._value(name2, X, y_hat)
        if u is None or v is None:
            return None
        b = None
        if cond_name is not None:
            b = self._value(cond_name, X, y_hat)
            if b is None:
                return None

        if name1 != self.target_col and self._is_effectively_constant(u):
            return None
        if name2 != self.target_col and self._is_effectively_constant(v):
            return None
        if (cond_name is not None and cond_name != self.target_col
                and self._is_effectively_constant(b)):
            return None

        r_u = self._residual(u, b)
        r_v = self._residual(v, b)
        u_spread = (u - u.mean()).norm()
        v_spread = (v - v.mean()).norm()
        denom = r_u.norm() * r_v.norm() + self.delta * u_spread * v_spread + 1e-30
        rho2 = ((r_u * r_v).sum() / denom) ** 2

        p = 1 if b is None else 1 + self._phi(b).shape[1]
        n = u.shape[0]
        nu = max(n - p, 1)
        return torch.relu((rho2 * nu - 1) / max(nu - 1, 1))

    def _constraint_for_triplet(self, triplet, X, y_hat):
        """(violation, n_terms) for one triplet -- see the class docstring's
        Math section for the formulas and the n_terms contract."""
        A, B, C = triplet.left, triplet.centre, triplet.right

        if triplet.type in (TripletType.CHAIN, TripletType.FORK):
            g = self._pcorr2(A, C, X, y_hat, cond_name=B)
            if g is None:
                return torch.zeros(1), 0
            return g.reshape(1), 1

        if triplet.type == TripletType.COLLIDER:
            if self.target_col in (A, C):
                eq2 = self._pcorr2(A, C, X, y_hat, cond_name=None)
                eq = torch.zeros(1) if eq2 is None else eq2.reshape(1)
            else:
                eq = torch.zeros(1)

            margin = self._collider_margins.get(triplet, self.collider_margin)
            cond2 = self._pcorr2(A, C, X, y_hat, cond_name=B)
            if cond2 is None:
                return eq, 0
            abs_rho = torch.sqrt(cond2 + 1e-12)
            hinge = torch.relu(margin - abs_rho).reshape(1)
            return eq + hinge, 1

        return torch.zeros(1), 0

    def calibrate_collider_margins(self, X_full, y_full, fraction=0.5):
        """Set each collider triplet's hinge margin from the ACHIEVABLE
        dependence, same recipe as the discrete version but from the TRUE
        standardized y instead of one-hot labels:
        margin_t = fraction * sqrt(rho2(A, C | centre) + 1e-12) on the
        training data -- the exact same expression _constraint_for_triplet's
        hinge uses, so margin and hinge are always on the same scale.

        Accepts numpy or torch for y_full (the base class calls this with the
        already-standardized training tensor y_t). Data-only: call once on
        the full training set. Returns the {triplet: margin} map (also stored
        on self).
        """
        X = torch.as_tensor(np.asarray(X_full), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(y_full), dtype=torch.float32).reshape(-1)
        margins = {}
        for t in self.triplets:
            if t.type != TripletType.COLLIDER:
                continue
            A, B, C = t.left, t.centre, t.right
            cond2 = self._pcorr2(A, C, X, y, cond_name=B)
            emp = 0.0 if cond2 is None else float(torch.sqrt(cond2 + 1e-12).item())
            margins[t] = max(0.0, fraction * emp)
        self._collider_margins = margins
        return margins

    def violations_with_terms(self, model, X_batch):
        """[(violation, n_terms), ...] for every triplet, in self.triplets order.

        model(X_batch) must return shape (batch,), not (batch, 1) -- see
        continuous_networks.py's module docstring for why.
        """
        if not self.triplets:
            return []
        y_hat = model(X_batch)
        return [self._constraint_for_triplet(t, X_batch, y_hat)
                for t in self.triplets]

    def __call__(self, model, X_batch, y_batch=None):
        """Per-triplet constraint violations of `model` on the batch.
        `y_batch` is ignored (kept for call-site compatibility)."""
        if not self.triplets:
            return torch.zeros(1)
        return torch.cat([v for v, _n in self.violations_with_terms(model, X_batch)])


class ContinuousRecommenderPredictor(CausalConstrainedPredictor):
    """Scikit-learn compatible neural regressor, the continuous sibling of
    DiscreteRecommenderPredictor over the shared CausalConstrainedPredictor.

    Implements the CausalConstrainedPredictor hooks for the continuous case;
    fit(), constraint_violation(), the feature-restriction baseline, and the
    whole ALM / Moreau-envelope / plain training loop are inherited unchanged
    from the base class (see causal_estimator_base.py for the full hyper-
    parameter reference).

    Training is driven by MSELoss on STANDARDIZED targets (self._y_mean_ /
    self._y_std_ fitted once, in _prepare_target). Features are standardized
    the same way via a StandardScaler fitted once, in _prepare_features
    (fitting=True only inside fit()). predict() un-standardizes back to the
    original scale.

    NN hyper-parameters (read from cfg, all optional):
        network      (str,   default 'mlp') backbone to use; one of
                     continuous_networks.NETWORK_REGISTRY (mlp, deep_mlp,
                     transformer).

    Continuous constraints-function hyper-parameters (read from cfg, used
    only when cfg.use_alm=True and no constraints_fn is supplied):
        cond_basis   (str,   default 'linear') 'linear' or 'poly:k'; see
                     ContinuousTripletConstraintsFn.
        ridge        (float, default 1e-4) ridge penalty for the FWL solve.
        conditioner  (str,   default 'linear') only 'linear' is implemented.

    _selection_loss is intentionally NOT overridden here (uses the base
    default: the plain criterion value). This is deliberate, not an
    oversight: y is standardized to unit variance, so MSELoss is already
    O(1)-scaled -- directly comparable to the constraint violations, which
    are pcorr^2 / hinge terms bounded in [0, 1] by Cauchy-Schwarz -- so the
    same alm_selection_weight=1.0 default that works for the discrete
    estimator's cross-entropy (also an O(1), per-sample-averaged quantity)
    is a sane default here too, without any extra rescaling.
    """

    def __init__(self, w_est, target_col, row_and_col_names, custom_objective,
                 prep_data, cfg, constraints_fn=None):
        super().__init__(w_est, target_col, row_and_col_names, custom_objective,
                         prep_data, cfg, constraints_fn)
        self._scaler_ = None
        self._y_mean_ = None
        self._y_std_ = None

    # ------------------------------------------------------------------
    # CausalConstrainedPredictor hooks
    # ------------------------------------------------------------------

    def _prepare_features(self, X_sel, fitting):
        """StandardScaler on the (already feature-restricted) input columns:
        fit_transform once during fit() (fitting=True), transform only
        afterward (constraint_violation(), predict())."""
        if fitting:
            self._scaler_ = StandardScaler()
            return self._scaler_.fit_transform(X_sel)
        return self._scaler_.transform(X_sel)

    def _prepare_target(self, y):
        """Standardize the raw target and remember (self._y_mean_,
        self._y_std_) for predict() to invert."""
        y_np = np.asarray(y, dtype=np.float64)
        self._y_mean_ = float(y_np.mean())
        self._y_std_ = float(y_np.std())
        if self._y_std_ < 1e-12:
            self._y_std_ = 1.0   # degenerate/constant target: skip scaling
        y_std = (y_np - self._y_mean_) / self._y_std_
        return torch.tensor(y_std, dtype=torch.float32), 1

    def _build_model(self, n_features, n_outputs, X_sel):
        network_name = self.cfg.get('network', 'mlp')
        return continuous_networks.build_network(network_name, n_features, self.cfg)

    def _build_criterion(self):
        return nn.MSELoss()

    def _build_constraints_fn(self, triplets, kept_features, X_sel, y_t):
        return ContinuousTripletConstraintsFn(
            triplets, kept_features, self.target_col,
            cond_basis=self.cfg.get('cond_basis', 'linear'),
            ridge=self.cfg.get('ridge', 1e-4),
            conditioner=self.cfg.get('conditioner', 'linear'))

    def _eval_constraints_fn(self, triplets, feature_names, X_sel):
        """Rebuild the reporting constraints fn, reusing the fitted
        self._constraints_fn_'s calibrated collider margins when it is a
        ContinuousTripletConstraintsFn, so training and reporting score the
        exact same constraint; unconstrained models fall back to the default
        margin."""
        fitted_cfn = self._constraints_fn_
        if isinstance(fitted_cfn, ContinuousTripletConstraintsFn):
            cfn = ContinuousTripletConstraintsFn(
                triplets, feature_names, self.target_col,
                cond_basis=fitted_cfn.cond_basis, ridge=fitted_cfn.ridge,
                collider_margin=fitted_cfn.collider_margin,
                conditioner=fitted_cfn.conditioner)
            cfn._collider_margins = dict(fitted_cfn._collider_margins)
        else:
            cfn = ContinuousTripletConstraintsFn(
                triplets, feature_names, self.target_col,
                cond_basis=self.cfg.get('cond_basis', 'linear'),
                ridge=self.cfg.get('ridge', 1e-4),
                conditioner=self.cfg.get('conditioner', 'linear'))
        return cfn

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def predict(self, X):
        X_t = torch.tensor(self._select_features(np.asarray(X)),
                           dtype=torch.float32)
        with torch.no_grad():
            y_hat = self._rf_model_(X_t).numpy()
        return y_hat * self._y_std_ + self._y_mean_
