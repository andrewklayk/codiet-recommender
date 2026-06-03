import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


def compute_discrete_predictor_errors_scikit(estimator, X, y):
    """Scoring for discrete estimator: returns classification error (1 - accuracy)."""
    return 1.0 - accuracy_score(y, estimator.predict(X))


class _MLP(nn.Module):
    """Simple MLP for multi-class classification."""

    def __init__(self, n_features, n_classes, hidden_dim, n_layers):
        super().__init__()
        layers = [nn.Linear(n_features, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
        if triplet['type'] == 'chain':
            # chain A→B→C:  E(C|A=a, B=b) = E(C|B=b)  for all (a,b)
            A, B, C = triplet['variables']
            total = torch.zeros(1)
            n = 0
            for b_val in range(self.n_values):
                e_c_b = self.expectation(C, X_batch, y_batch, given={B: b_val})
                for a_val in range(self.n_values):
                    e_c_ab = self.expectation(C, X_batch, y_batch,
                                              given={A: a_val, B: b_val})
                    total = total + (e_c_ab - e_c_b) ** 2
                    n += 1
            return total / n

        if triplet['type'] == 'fork':
            # fork A←C→B:  E(A·B|C=c) = E(A|C=c)·E(B|C=c)  for all c
            A, C, B = triplet['variables']  # centre is the fork node
            total = torch.zeros(1)
            for c_val in range(self.n_values):
                e_ab_c = self.expectation_product(A, B, X_batch, y_batch,
                                                  given={C: c_val})
                e_a_c  = self.expectation(A, X_batch, y_batch, given={C: c_val})
                e_b_c  = self.expectation(B, X_batch, y_batch, given={C: c_val})
                total  = total + (e_ab_c - e_a_c * e_b_c) ** 2
            return total / self.n_values

        if triplet['type'] == 'collider':
            # collider A→C←B:
            #   equality  : E(A·B) = E(A)·E(B)          (marginal independence)
            #   inequality: E(A·B|C=c) ≠ E(A|C=c)·E(B|C=c)  for all c
            #               encoded as hinge: max(0, τ − |difference|)
            A, C, B = triplet['variables']  # centre is the collider node

            # --- equality part ---
            eq = (self.expectation_product(A, B, X_batch, y_batch)
                  - self.expectation(A, X_batch, y_batch)
                  * self.expectation(B, X_batch, y_batch)) ** 2

            # --- inequality part: hinge over all c values ---
            hinge = torch.zeros(1)
            for c_val in range(self.n_values):
                e_ab_c = self.expectation_product(A, B, X_batch, y_batch,
                                                  given={C: c_val})
                e_a_c  = self.expectation(A, X_batch, y_batch, given={C: c_val})
                e_b_c  = self.expectation(B, X_batch, y_batch, given={C: c_val})
                diff   = (e_ab_c - e_a_c * e_b_c).abs()
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
    """Scikit-learn compatible MLP classifier for discrete variable prediction.

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
        hidden_dim   (int,   default 64)   hidden layer width
        n_layers     (int,   default 2)    number of hidden layers
        n_epochs     (int,   default 50)   training epochs
        batch_size   (int,   default 32)   mini-batch size
        learning_rate(float, default 1e-3) Adam learning rate

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

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        # --- resolve hyper-parameters from cfg ---
        hidden_dim    = self.cfg.get('hidden_dim',    64)
        n_layers      = self.cfg.get('n_layers',      2)
        n_epochs      = self.cfg.get('n_epochs',      50)
        batch_size    = self.cfg.get('batch_size',    32)
        lr            = self.cfg.get('learning_rate', 1e-3)
        use_alm       = self.cfg.get('use_alm',       False)
        n_constraints = getattr(self.constraints_fn, 'n_constraints',
                               self.cfg.get('n_constraints', 1))
        alm_lr        = self.cfg.get('alm_lr',        0.1)
        alm_momentum  = self.cfg.get('alm_momentum',  0.5)

        # --- auto-inject TripletConstraintsFn when use_alm=True and no fn provided ---
        if use_alm and self.constraints_fn is None:
            feature_names = (list(X.columns) if hasattr(X, 'columns')
                             else [n for n in self.row_and_col_names
                                   if n != self.target_col])
            y_arr = np.asarray(y)
            n_values_inferred = int(y_arr.max()) + 1
            self.constraints_fn = TripletConstraintsFn(
                self.get_target_triplets(), feature_names,
                self.target_col, n_values_inferred
            )

        # --- label encoding ---
        y_np = np.asarray(y)
        self._classes = np.unique(y_np)
        label_to_idx = {c: i for i, c in enumerate(self._classes)}
        y_enc = np.array([label_to_idx[c] for c in y_np], dtype=np.int64)

        # --- tensors & dataloader ---
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.tensor(y_enc)
        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=batch_size, shuffle=True)

        # --- build network ---
        n_features = X_t.shape[1]
        n_classes  = len(self._classes)
        model = _MLP(n_features, n_classes, hidden_dim, n_layers)
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
        for _ in range(n_epochs):
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

        self._rf_model_ = model
        self._w_est = self.w_est
        return self

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def predict(self, X):
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            indices = self._rf_model_(X_t).argmax(dim=1).numpy()
        return self._classes[indices]

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
            list[dict] each with keys:
                'type'      : 'chain' | 'fork' | 'collider'
                'variables' : (left, centre, right) variable name strings
                'center'    : name of the centre node
        """
        W = self.w_est
        names = list(self.row_and_col_names)
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        t = name_to_idx[self.target_col]
        others = [i for i in range(len(names)) if i != t]

        triplets = []
        for k in range(len(others)):
            for l in range(k + 1, len(others)):
                i, j = others[k], others[l]
                # rotate through each node as centre; left/right are the remaining two
                for centre, left, right in [(t, i, j), (i, t, j), (j, t, i)]:
                    lc = W[left, centre] != 0   # left  → centre
                    cl = W[centre, left] != 0   # centre → left
                    rc = W[right, centre] != 0  # right → centre
                    cr = W[centre, right] != 0  # centre → right

                    # chain:    left → centre → right
                    if lc and cr:
                        triplets.append({'type': 'chain',
                                         'variables': (names[left], names[centre], names[right]),
                                         'center': names[centre]})
                    # chain:    right → centre → left
                    if rc and cl:
                        triplets.append({'type': 'chain',
                                         'variables': (names[right], names[centre], names[left]),
                                         'center': names[centre]})
                    # fork:     left ← centre → right
                    if cl and cr:
                        triplets.append({'type': 'fork',
                                         'variables': (names[left], names[centre], names[right]),
                                         'center': names[centre]})
                    # collider: left → centre ← right
                    if lc and rc:
                        triplets.append({'type': 'collider',
                                         'variables': (names[left], names[centre], names[right]),
                                         'center': names[centre]})

        return triplets