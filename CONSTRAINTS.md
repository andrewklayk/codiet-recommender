# Causal constraints in the augmented Lagrangian

This describes how the causally-constrained estimators (`DiscreteRecommenderPredictor`,
`ContinuousRecommenderPredictor`) turn the learned DAG `w_est` into constraint terms
and how those terms drive training via ALM. It documents the mechanism on the
`expectations` branch, not the single-equation SEM constraint used on `main`
(`nn_lagrangian.py` / `xgboost_lagrangian.py`), which this replaces.

## 1. What is being constrained

Instead of one global linear equation derived algebraically from `w_est`, training
here is constrained by every **chain / fork / collider triplet** in the DAG that
involves the target variable — one differentiable violation term per triplet, all
driven toward zero simultaneously by ALM.

## 2. Where the triplets come from

[`causal_triplets.get_target_triplets`](causal_triplets.py#L83-L145) enumerates every
triplet `{target, i, j}` and classifies it by rotating each node through the "centre"
role and checking directed edges + d-separation against the *full* graph (not just
direct edges):

| Structure | Shape | d-separation fact |
|---|---|---|
| **Chain** | `A → B → C` | `A ⊥ C \| B` |
| **Fork** | `A ← B → C` | `A ⊥ C \| B` |
| **Collider** | `A → B ← C` | `A ⊥ C` marginally, but `A ⊥̸ C \| B` (explaining-away) |

When `restrict_to_parents` / `restrict_to_markov_blanket` drops features, triplets
touching a dropped variable are filtered out too
([`causal_estimator_base.py:311-316`](causal_estimator_base.py#L311-L316)) — a
constraint over a variable the model never sees is meaningless.

## 3. The violation for each triplet

Two implementations, same semantics, different math because one is over discrete
categories and one over continuous reals. The target's value is always the model's
own **differentiable prediction** (soft class distribution / raw regression output);
every other variable in a triplet comes straight from the data.

**Discrete** — [`discrete_estimator.TripletConstraintsFn._constraint_for_triplet`](discrete_estimator.py#L119-L212):
- chain: average over all `(a, b)` cells of `(E[C | A=a, B=b] − E[C | B=b])²`
- fork: average over all `b` of `(E[A·C | B=b] − E[A|B=b]·E[C|B=b])²`
- collider: `eq + hinge`, where
  - `eq = (E[A·C] − E[A]·E[C])²` (marginal independence; only counted when the
    target is an endpoint `A`/`C` — if the target is the centre, both endpoints are
    data columns and this term is a constant fact with zero gradient, so it's skipped)
  - `hinge = mean_b max(0, margin − |E[A·C|B=b] − E[A|B=b]·E[C|B=b]|)` — this is the
    part that *requires* dependence once you condition on the collider, not just
    tolerates it

**Continuous** — [`continuous_estimator.ContinuousTripletConstraintsFn._constraint_for_triplet`](continuous_estimator.py#L265-L291):
conditional independence is proxied by **squared partial correlation** `ρ²(A, C | B)`,
computed via ridge-regularized linear residualization (Frisch–Waugh–Lovell) and
debiased for the small-sample upward bias of sample R² under the null. Chain and
fork collapse to the identical `ρ²(A, C | centre)` test (same CI fact, different DAG
shape). Collider is `eq + hinge` exactly as above, with `hinge = max(0, margin −
√ρ²(A,C|B))`. Correlation, not covariance, is used deliberately — the class
docstring proves covariance would let ALM cheat by shrinking the model's output
scale instead of actually satisfying the constraint.

Both versions calibrate each collider's `margin` from data rather than hand-setting
it: `calibrate_collider_margins` measures the dependence the TRUE labels actually
produce given the centre, and sets `margin = collider_margin_fraction × that`
(default fraction `0.5`), so the hinge never demands more conditional dependence
than the data can support — an uncalibrated fixed margin can be unsatisfiable,
which sends the corresponding dual to infinity.

All violations are **non-negative by construction** (squared terms, or a hinge
floored at 0) — "satisfied" means the term is exactly 0, not merely small in either
direction.

## 4. The constraints-function contract

Both classes implement the same interface, consumed generically by the base class:

```
constraints_fn(model, X_batch, y_batch=None) -> 1-D Tensor   # one scalar per triplet
constraints_fn.n_constraints                                  # len(triplets), >= 1
constraints_fn.violations_with_terms(model, X_batch) -> [(violation, n_terms), ...]
```

`n_terms` (how many conditioning cells were actually computable on this batch) lets
callers tell "constraint genuinely satisfied" apart from "every relevant cell was
empty/degenerate here" — see `constraint_violation()` in the base class, which
reuses this for post-hoc reporting.

## 5. How this enters the Lagrangian — [`causal_estimator_base.py:fit()`](causal_estimator_base.py#L239-L420)

```python
dual = ALM(m=n_constraints, lr=alm_lr, momentum=alm_momentum,
           penalty=alm_penalty, is_ineq=True)
...
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:                       # duals held FIXED all epoch
        constraints = constraints_fn(model, X_batch, y_batch)
        lagrangian = dual.forward(loss, constraints)       # L = loss + λᵀc  (unaugmented, alm_penalty=0)
        lagrangian.backward()
        optimizer.step()                                   # <- updates θ, every minibatch

    c_full = constraints_fn(model, X_t)                     # FULL training set
    dual.update(c_full - alm_slack)                          # <- updates λ, once per epoch
```

Current configs (`experiments_conf/solver/*_constrained.yaml`) set `alm_penalty: 0.0` and
`alm_momentum: 0.`, and `augmentation` is never passed to `ALM(...)`. In the installed
`humancompatible-train`, `augmentation` only defaults to `"hpr"` when `penalty > 0`; at
`penalty=0` it stays `None`, which `ALM` treats identically to `"quadratic"` (no augmentation
term at all). So this is deliberately the **plain, unaugmented Lagrangian** — `L = loss + λᵀc`,
`λ ← λ + alm_lr·c` — not an augmented one. (The docs published on the `mpc` branch as of this
writing still show `penalty` defaulting to `1.0` and `augmentation` defaulting unconditionally to
`"hpr"`, under which `penalty=0` would raise `ValueError` — the installed checkout is a few
commits ahead of that and was patched specifically to make this mode well-defined. Re-pinning to
an older/published build of `humancompatible-train` would break it.)

Two deliberate design choices:

- **Primal/dual split.** The primal step (`optimizer.step()`) runs every minibatch
  on the noisy batch-level constraint estimate, with `dual.forward` treating λ as a
  fixed constant. The dual step runs once per epoch, on the honest full-training-set
  violation — decoupling a stable, low-variance dual update from noisy per-batch
  gradients. This is why `alm_lr` (dual learning rate, default `5.0`) is large
  relative to a typical dual step size: it only gets one update per epoch, not one
  per batch, so it needs to move further each time.
- **Slack.** The dual updates on `(c_full − alm_slack)`, not `c_full` directly
  (`alm_slack` default `0.01`). Since every violation is one-sided (≥ 0 always),
  without slack the duals can only ratchet upward — there is no "over-satisfied"
  direction to pull them back down. Subtracting a small slack lets λ shrink again
  once the violation is within estimation noise of zero.

`is_ineq=True` (current setting on this branch) tells the `humancompatible-train`
ALM to treat each triplet as an inequality constraint `c(θ) ≤ 0` rather than an
equality: duals are floored at 0 and relax toward 0 as a constraint becomes
strictly satisfied, instead of being penalized symmetrically in both directions.
Since these violations can never go negative, equality-style symmetric penalization
has no real meaning here — `is_ineq=True` is the mode that actually matches a
quantity with only one feasible side.

**Optimizer.** `use_alm=True` now pairs `ALM` with a **plain** `torch.optim.Adam` — no
`MoreauEnvelope` wrapping ([causal_estimator_base.py:347-352](causal_estimator_base.py#L347-L352)).
This was a deliberate change to detach the Moreau-envelope smoothing's own effect from the
causal constraint's effect, and it also happens to match the only pattern the
`humancompatible-train` docs ever show (`basic_usage.md` / `getting_started.rst` both pair `ALM`
with bare `Adam`/`AdamW`; neither tutorial wraps the primal optimizer in anything).

One consequence: **`uncon_ME` is no longer a matched control for `con_*`.** It was a valid
control back when both arms shared the same Moreau wrapper; now that `con_*` uses plain Adam,
the correct unconstrained comparison is `uncon_all`. `uncon_ME` (Moreau smoothing alone, no
constraints) still exists as its own ablation in `test_all_networks*.py`'s `SETTINGS`, but it
isolates the optimizer's effect in general, not specifically relative to `con_*` anymore.

## 6. Best-epoch selection

The checkpoint kept isn't the last epoch but the one with the lowest full-train
`_selection_loss(...) + alm_selection_weight × mean(c_full)` (constrained) or plain
`_selection_loss(...)` (unconstrained) — the Lagrangian value itself isn't used for
selection because it shifts as λ moves, so it isn't comparable across epochs.

## 7. Where each piece lives

| Piece | File |
|---|---|
| Triplet enumeration (graph-only) | [`causal_triplets.py`](causal_triplets.py) |
| Shared ALM/Moreau training loop | [`causal_estimator_base.py`](causal_estimator_base.py) (`CausalConstrainedPredictor.fit`) |
| Discrete violation math | [`discrete_estimator.py`](discrete_estimator.py) (`TripletConstraintsFn`) |
| Continuous violation math | [`continuous_estimator.py`](continuous_estimator.py) (`ContinuousTripletConstraintsFn`) |
| ALM dual optimizer itself | `humancompatible.train.dual_optim.ALM` (external package, `pip install humancompatible-train`) |
