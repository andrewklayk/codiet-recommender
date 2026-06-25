import copy
from dataclasses import dataclass
from enum import Enum

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from discrete_networks import build_network


def compute_discrete_predictor_errors_scikit(estimator, X, y):
    """Scoring for discrete estimator: returns classification error (1 - accuracy)."""
    return 1.0 - accuracy_score(y, estimator.predict(X))


class TripletType(Enum):
    """Causal structure of a triplet centred on `centre`.

    CHAIN    — left → centre → right
    FORK     — left ← centre → right   (common cause)
    COLLIDER — left → centre ← right   (common effect / v-structure)
    """
    CHAIN = 'chain'
    FORK = 'fork'
    COLLIDER = 'collider'


@dataclass(frozen=True)
class Triplet:
    """A causal triplet over three named variables, centred on `centre`."""
    type: TripletType
    left: str
    centre: str
    right: str


class TripletConstraintsFn:
    """Constraint function for ALM training, one violation per causal triplet.

    Args:
        triplets     : output of DiscreteRecommenderPredictor.get_target_triplets()
        feature_names: ordered column names of X as passed to fit()
        target_col   : name of the target variable (matches y_batch)
    """

    def __init__(self, triplets, feature_names, target_col, n_values,
                 collider_margin=0.1):
        self.triplets = triplets
        self.feature_names = list(feature_names)
        self.target_col = target_col
        self.n_values = n_values
        self.collider_margin = collider_margin
        self._col_idx = {name: i for i, name in enumerate(self.feature_names)}

    @property
    def n_constraints(self):
        return max(1, len(self.triplets))

    def _get_var(self, name, X_batch, y_batch):
        """Float tensor for variable `name` from the current batch, or None."""
        if name == self.target_col:
            return y_batch.float()
        idx = self._col_idx.get(name)
        return X_batch[:, idx].float() if idx is not None else None

    def expectation(self, var_name, X_batch, y_batch, given=None):
        """E(var_name) or E(var_name | given) computed from the batch.

        given: dict of {var_name: int_value} conditioning events, or None.
        Returns a (1,) Tensor; zeros(1) when the conditioning set is empty.
        """
        vals = self._get_var(var_name, X_batch, y_batch)
        if vals is None:
            return torch.zeros(1)
        if not given:
            return vals.mean().unsqueeze(0)
        mask = torch.ones(len(vals), dtype=torch.bool)
        for cond_name, cond_val in given.items():
            cond = self._get_var(cond_name, X_batch, y_batch)
            if cond is None:
                return torch.zeros(1)
            mask &= cond.long() == int(cond_val)
        return vals[mask].mean().unsqueeze(0) if mask.any() else torch.zeros(1)

    def expectation_product(self, var1, var2, X_batch, y_batch, given=None):
        """E(var1 * var2 | given) computed from the batch.

        Same interface as expectation() but for the joint product of two variables.
        Returns a (1,) Tensor; zeros(1) when conditioning set is empty or either
        variable is absent from the batch.
        """
        v1 = self._get_var(var1, X_batch, y_batch)
        v2 = self._get_var(var2, X_batch, y_batch)
        if v1 is None or v2 is None:
            return torch.zeros(1)
        prod = v1 * v2
        if not given:
            return prod.mean().unsqueeze(0)
        mask = torch.ones(len(prod), dtype=torch.bool)
        for cond_name, cond_val in given.items():
            cond = self._get_var(cond_name, X_batch, y_batch)
            if cond is None:
                return torch.zeros(1)
            mask &= cond.long() == int(cond_val)
        return prod[mask].mean().unsqueeze(0) if mask.any() else torch.zeros(1)

    def _constraint_for_triplet(self, triplet, X_batch, y_batch):
        """Scalar violation for one triplet.

        Chain A→B→C: E(C|A=a, B=b) = E(C|B=b) for all (a,b).
        Returns the mean squared difference across all (a,b) pairs —
        zero when the constraint holds, positive otherwise.
        Fork / collider: placeholder zeros (filled in later steps).
        """
        A, B, C = triplet.left, triplet.centre, triplet.right
        if triplet.type == TripletType.CHAIN:
            # chain A→B→C:  E(C|A=a, B=b) = E(C|B=b)  for all (a,b)
            total = torch.zeros(1)
            for b_val in range(self.n_values):
                e_c_b = self.expectation(C, X_batch, y_batch, given={B: b_val})
                for a_val in range(self.n_values):
                    e_c_ab = self.expectation(C, X_batch, y_batch,
                                              given={A: a_val, B: b_val})
                    total = total + (e_c_ab - e_c_b) ** 2
            return total / self.n_values / self.n_values

        if triplet.type == TripletType.FORK:
            # fork A←B→C:  E(A·C|B=b) = E(A|B=b)·E(C|B=b)  for all b
            total = torch.zeros(1)
            for b_val in range(self.n_values):
                e_ac_b = self.expectation_product(A, C, X_batch, y_batch,
                                                  given={B: b_val})
                e_a_b  = self.expectation(A, X_batch, y_batch, given={B: b_val})
                e_c_b  = self.expectation(C, X_batch, y_batch, given={B: b_val})
                total  = total + (e_ac_b - e_a_b * e_c_b) ** 2
            return total / self.n_values

        if triplet.type == TripletType.COLLIDER:
            # collider A→B←C:
            #   equality  : E(A·C) = E(A)·E(C)          (marginal independence)
            #   inequality: E(A·C|B=b) ≠ E(A|B=b)·E(C|B=b)  for all b
            #               encoded as hinge: max(0, τ − |difference|)
            # --- equality part ---
            eq = (self.expectation_product(A, C, X_batch, y_batch)
                  - self.expectation(A, X_batch, y_batch)
                  * self.expectation(C, X_batch, y_batch)) ** 2

            # --- inequality part: hinge over all b values ---
            hinge = torch.zeros(1)
            for b_val in range(self.n_values):
                e_ac_b = self.expectation_product(A, C, X_batch, y_batch,
                                                  given={B: b_val})
                e_a_b  = self.expectation(A, X_batch, y_batch, given={B: b_val})
                e_c_b  = self.expectation(C, X_batch, y_batch, given={B: b_val})
                diff   = (e_ac_b - e_a_b * e_c_b).abs()
                hinge  = hinge + torch.relu(self.collider_margin - diff)
            hinge = hinge / self.n_values

            return eq + hinge

        return torch.zeros(1)

    def __call__(self, model, X_batch, y_batch):
        if not self.triplets:
            return torch.zeros(1)
        return torch.cat([
            self._constraint_for_triplet(t, X_batch, y_batch)
            for t in self.triplets
        ])


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
        alm_lr       (float, default 0.1)  dual variable learning rate
        alm_momentum (float, default 0.5)  dual variable momentum
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
        self._w_est = w_est
        self._classes = None
        self._feature_mask = None
        self._kept_features = None

    # ------------------------------------------------------------------
    # feature restriction (interaction-graph baseline)
    # ------------------------------------------------------------------

    def _parent_features(self, feature_names):
        """Subset of `feature_names` that are direct causes (parents) of the target.

        Uses the interaction graph w_est (w_est[i, j] != 0 means i -> j): the
        parents of the target are exactly the variables with a directed edge
        into it. Order follows `feature_names`.
        """
        names = list(self.row_and_col_names)
        name_to_idx = {n: i for i, n in enumerate(names)}
        t = name_to_idx[self.target_col]
        parents = {names[i] for i in range(len(names)) if self.w_est[i, t] != 0}
        return [f for f in feature_names if f in parents]

    def _markov_blanket_features(self, feature_names):
        """Subset of `feature_names` in the target's Markov blanket.

        The Markov blanket is parents + children + co-parents (the other parents
        of the target's children). Under the interaction graph w_est these are
        exactly the variables that carry predictive information about the target:
        everything outside the blanket is d-separated from the target given it.
        Order follows `feature_names`.
        """
        names = list(self.row_and_col_names)
        name_to_idx = {n: i for i, n in enumerate(names)}
        t = name_to_idx[self.target_col]
        W = self.w_est
        parents = {i for i in range(len(names)) if W[i, t] != 0}
        children = {j for j in range(len(names)) if W[t, j] != 0}
        coparents = {i for c in children for i in range(len(names)) if W[i, c] != 0}
        mb = {names[i] for i in (parents | children | coparents) - {t}}
        return [f for f in feature_names if f in mb]

    def _select_features(self, X_arr):
        """Apply the fitted column mask; fall back to a constant column.

        When the target has no retained features (e.g. a root node under
        restrict_to_parents) the mask is empty and we feed a single constant
        feature so the network simply learns the class prior.
        """
        if self._feature_mask:
            return X_arr[:, self._feature_mask]
        return np.zeros((X_arr.shape[0], 1), dtype=X_arr.dtype)

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
        alm_lr        = self.cfg.get('alm_lr',        0.1)
        alm_momentum  = self.cfg.get('alm_momentum',  0.5)

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
            kept = self._markov_blanket_features(feature_names)
        elif restrict_mode == 'parents':
            kept = self._parent_features(feature_names)
        else:
            kept = list(feature_names)

        if restrict_mode != 'none':
            kept_set = set(kept)
            self._feature_mask = [i for i, f in enumerate(feature_names)
                                  if f in kept_set]
        else:
            self._feature_mask = list(range(len(feature_names)))
        self._kept_features = kept

        # --- auto-inject TripletConstraintsFn when use_alm=True and no fn provided ---
        if use_alm and self.constraints_fn is None:
            y_arr = np.asarray(y)
            n_values_inferred = int(y_arr.max()) + 1
            triplets = self.get_target_triplets()
            if restrict_mode != 'none':
                # A constraint over an excluded variable is meaningless for a
                # model that never sees it: keep only triplets whose variables
                # all remain (the target plus the retained features).
                allowed = set(kept) | {self.target_col}
                triplets = [tr for tr in triplets
                            if {tr.left, tr.centre, tr.right} <= allowed]
            self.constraints_fn = TripletConstraintsFn(
                triplets, kept, self.target_col, n_values_inferred
            )

        # Determine the dual dimension AFTER the constraints function exists, so
        # every constraint it returns gets its own dual variable in the ALM.
        n_constraints = getattr(self.constraints_fn, 'n_constraints',
                                self.cfg.get('n_constraints', 1))

        # --- label encoding ---
        y_np = np.asarray(y)
        self._classes = np.unique(y_np)
        label_to_idx = {c: i for i, c in enumerate(self._classes)}
        y_enc = np.array([label_to_idx[c] for c in y_np], dtype=np.int64)

        # --- tensors & dataloader (features restricted to self._feature_mask) ---
        X_t = torch.tensor(self._select_features(np.asarray(X)),
                           dtype=torch.float32)
        y_t = torch.tensor(y_enc)
        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=batch_size, shuffle=True)

        # --- build network ---
        n_features = X_t.shape[1]
        n_classes  = len(self._classes)
        # category cardinality for encoders/embeddings: max index over the
        # features (plus 1). Raw-float backbones ignore it.
        n_values = int(X_t.max().item()) + 1 if X_t.numel() else 1
        network_name = self.cfg.get('network', 'mlp')
        model = build_network(network_name, n_features, n_classes, n_values, self.cfg)
        criterion = nn.CrossEntropyLoss()

        # --- optimiser (plain or ALM-wrapped) ---
        if use_alm:
            # humancompatible/train integration:
            # pip install humancompatible-train
            from humancompatible.train.dual_optim import ALM, MoreauEnvelope
            optimizer = MoreauEnvelope(torch.optim.Adam(model.parameters(), lr=lr))
            dual = ALM(m=n_constraints, lr=alm_lr, momentum=alm_momentum)
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
        best_loss = float('inf')
        best_state = copy.deepcopy(model.state_dict())
        for _ in range(n_epochs):
            model.train()
            for X_batch, y_batch in loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                if dual is not None and self.constraints_fn is not None:
                    # constrained update via Augmented Lagrangian
                    constraints = self.constraints_fn(model, X_batch, y_batch)
                    lagrangian = dual.forward_update(loss, constraints)
                    lagrangian.backward()
                else:
                    loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            # --- model selection: track lowest training cross-entropy ---
            model.eval()
            with torch.no_grad():
                epoch_loss = criterion(model(X_t), y_t).item()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
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

        Reuses the very same triplet enumeration (get_target_triplets) and
        TripletConstraintsFn as constrained training, but substitutes the model's
        PREDICTED target for the true label, so the result measures whether the
        trained predictor respects the conditional-independence constraints
        implied by w_est. Works for any fitted model regardless of cfg.use_alm,
        enabling a constrained-vs-unconstrained comparison.

        X : the same feature frame/array passed to fit/predict (the estimator
            applies its own feature mask internally for prediction; constraint
            expectations use the feature columns present in X).
        aggregate : 'mean' (default) or 'sum' over the per-triplet violations.

        Returns dict {'total': float, 'per_triplet': list[float],
        'n_triplets': int}; total is 0.0 when no triplet involves the target.
        """
        if self._rf_model_ is None:
            raise RuntimeError("call fit() before constraint_violation()")
        triplets = self.get_target_triplets()
        if not triplets:
            return {'total': 0.0, 'per_triplet': [], 'n_triplets': 0}
        feature_names = (list(X.columns) if hasattr(X, 'columns')
                         else [n for n in self.row_and_col_names
                               if n != self.target_col])
        X_arr = np.asarray(X)
        preds = np.asarray(self.predict(X))
        n_values = int(max(X_arr.max(initial=0), preds.max(initial=0))) + 1
        cfn = TripletConstraintsFn(triplets, feature_names,
                                   self.target_col, n_values)
        with torch.no_grad():
            viol = cfn(self._rf_model_,
                       torch.tensor(X_arr, dtype=torch.float32),
                       torch.tensor(preds)).detach().cpu().numpy()
        total = float(viol.mean() if aggregate == 'mean' else viol.sum())
        return {'total': total, 'per_triplet': viol.tolist(),
                'n_triplets': len(triplets)}

    # ------------------------------------------------------------------
    # causal structure analysis
    # ------------------------------------------------------------------

    def get_target_triplets(self):
        """Enumerate all chain / fork / collider triplets in w_est that involve target_col.

        w_est convention (confirmed from compute_tools.py / notears_util.py):
            w_est[i, j] != 0  means  i → j  (row = source, column = target)

        For each unordered triplet {target, i, j} the method tests every node as
        the potential centre and reports matching directed structures:
            chain    :  left → centre → right
            fork     :  left ← centre → right   (common cause)
            collider :  left → centre ← right   (common effect / v-structure)

        Returns:
            list[Triplet] — each with .type (a TripletType) and the
            .left / .centre / .right variable name strings.
        """
        names = list(self.row_and_col_names)
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        t = name_to_idx[self.target_col]
        others = [i for i in range(len(names)) if i != t]

        # Directed graph over node indices for d-separation tests.
        # self.w_est[i, j] != 0 means edge i → j.
        G = nx.DiGraph()
        G.add_nodes_from(range(len(names)))
        G.add_edges_from((i, j) for i in range(len(names))
                         for j in range(len(names)) if self.w_est[i, j] != 0)

        triplets = []
        for k in range(len(others)):
            for l in range(k + 1, len(others)):
                i, j = others[k], others[l]
                # rotate through each node as centre; left/right are the remaining two
                for centre, left, right in [(t, i, j), (i, t, j), (j, t, i)]:
                    lc = self.w_est[left, centre] != 0   # left  → centre
                    cl = self.w_est[centre, left] != 0   # centre → left
                    rc = self.w_est[right, centre] != 0  # right → centre
                    cr = self.w_est[centre, right] != 0  # centre → right

                    # The numeric (conditional) independence each constraint
                    # encodes holds only when the conditioning set actually
                    # d-separates the two endpoints in the *full* DAG — not just
                    # when the direct left↔right edge is absent.  Indirect active
                    # paths through other nodes would otherwise invalidate the fact.
                    #   chain / fork : left ⊥ right | {centre}
                    #   collider     : left ⊥ right | {}        (marginal)
                    sep_given_centre = nx.is_d_separator(G, {left}, {right}, {centre})
                    sep_marginal     = nx.is_d_separator(G, {left}, {right}, set())

                    # chain:    left → centre → right
                    if lc and cr and sep_given_centre:
                        triplets.append(Triplet(TripletType.CHAIN, names[left], names[centre], names[right]))
                    # chain:    right → centre → left
                    if rc and cl and sep_given_centre:
                        triplets.append(Triplet(TripletType.CHAIN, names[right], names[centre], names[left]))
                    # fork:     left ← centre → right
                    if cl and cr and sep_given_centre:
                        triplets.append(Triplet(TripletType.FORK, names[left], names[centre], names[right]))
                    # collider: left → centre ← right
                    if lc and rc and sep_marginal:
                        triplets.append(Triplet(TripletType.COLLIDER, names[left], names[centre], names[right]))

        return triplets