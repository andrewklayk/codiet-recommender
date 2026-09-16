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
([`causal_estimator_base.py:334-339`](causal_estimator_base.py#L334-L339)) — a
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

## 5. How this enters the Lagrangian — [`causal_estimator_base.py:fit()`](causal_estimator_base.py#L262-L473)

```python
dual = ALM(m=n_constraints, lr=alm_lr, momentum=alm_momentum,
           penalty=alm_penalty, is_ineq=True)
...
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:                       # duals held FIXED all epoch
        constraints = constraints_fn(model, X_batch, y_batch) - alm_slack
        lagrangian = dual.forward(loss, constraints)       # L = loss + λᵀc  (unaugmented, alm_penalty=0)
        lagrangian.backward()
        optimizer.step()                                   # <- updates θ, every minibatch

    c_full = constraints_fn(model, X_t)                     # FULL training set
    dual.update(c_full - alm_slack)                          # <- updates λ, once per epoch
```

`alm_slack` is subtracted in **both** places, not just the dual update (see the bug
this fixed, below the worked example) -- `constraints_fn` always returns the raw,
non-negative violation; `- alm_slack` is what turns that into "the quantity the ALM
formulation's `c(θ) ≤ 0` template is actually about."

Current configs (`experiments_conf/solver/*_constrained.yaml`) set `alm_penalty: 1.0`, which
(with `augmentation` left unset) auto-selects the ALM package's `"hpr"`
(Hestenes–Powell–Rockafellar) augmentation instead of the plain Lagrangian. This was **not**
the original choice: an earlier version of this branch ran `alm_penalty: 0.0` (deliberately the
plain, unaugmented Lagrangian, `L = loss + λᵀc`, `λ ← λ + alm_lr·c`), on the reasoning that it
was the only pattern the `humancompatible-train` tutorials showed. That turned out to be the
bug. Diagnosing why `con_all`/`con_mb` were losing to `uncon_all` by 2–4× on the random-ER-graph
sweep — on train error, not just test, ruling out overfitting — by instrumenting the dual
trajectory directly showed why: under the plain update, a target whose triplets (typically
colliders, which must be pushed *away* from independence rather than just avoid drifting into
it) can't jointly be driven below `alm_slack` (0.01) has **no fixed point** — `c_full − slack`
stays positive every epoch, so `λ ← λ + alm_lr·c` ratchets forever. On a `d=10, s0=15` ER graph,
one target's dual reached the ALM's own clamp (`dual_range` upper bound, 100.0) by epoch ~140 of
a 50-epoch-configured run, at which point `Σλᵀc` (≈17) dwarfed the MSE loss (≈0.1–0.4) and the
model gave up fitting almost entirely (MSE 0.33–0.37 vs. 0.074 unconstrained on the same target/
seed/init). Targets whose triplets *could* be satisfied below slack behaved fine — duals decayed
back to 0 and `con_all` matched `uncon_all`. So the failure was optimizer dynamics, not the
constraint concept: **most** targets in a moderately dense ER graph carry at least one collider
triplet, so this wasn't a corner case.

Under `"hpr"`, the primal step's weight on `∂c/∂θ` is `max(0, λ + ρ·c)` rather than the raw `λ`
(`ρ` = `alm_penalty`) — see [`ALM`'s docstring](https://github.com/andrewklayk/humancompatible-train)
for the exact dual/primal update. This does **not** by itself cap a dual whose constraint stays
persistently above slack (the same unbounded-ratchet risk applies at any `penalty > 0`, just
mediated through this different update); what it changes is that an *over*-satisfied constraint's
multiplier stops pulling on the primal step entirely once `λ + ρ·c ≤ 0`, instead of the raw `λ`
permanently dragging on the objective regardless of how negative `c` has become. Whether this
is enough on its own is an empirical question, not a settled one — check `n_duals_saturated` /
`dual_max` in `train_history_` (see §6) on a fresh sweep before trusting it silently fixed
`con_all`'s numbers. If saturation persists, the more direct fix is loosening `alm_slack` to
match what a target's triplets can actually achieve (empirically closer to 0.03–0.05 for
colliders than the current 0.01), which the per-epoch diagnostics now make possible to check
target-by-target instead of guessing.

**Per-triplet slack calibration (opt-in).** A single global `alm_slack` is a compromise: on the
discrete estimator, `calibrate_slacks.ipynb` measured each triplet's achievable violation floor by
feeding the TRUE training label through the same `_constraint_for_triplet` machinery training
uses (in place of the model's prediction — the same trick `calibrate_collider_margins` already
uses for the collider hinge margin, generalized to every triplet type and to `alm_slack` itself)
and found ~20% of triplets — all `chain`-type, spanning 9 of 10 targets on a `d=10, s0=15` ER
graph — with a floor above the default `0.01`, up to `0.1287` (~13×). Those triplets can never
satisfy `c − alm_slack ≤ 0` no matter how well the model fits, so their dual has no fixed point
regardless of the `"hpr"` fix above. `TripletConstraintsFn.calibrate_slacks(X, y, fraction)` (called
from `fit()` right after `calibrate_collider_margins`, so a collider's floor already reflects its
calibrated hinge margin) sets each triplet's slack to `fraction × floor` (`fraction ≥ 1.0`, default
`1.2` — headroom above the floor, not a minimum ask like `collider_margin_fraction`); `fit()` then
subtracts this **per-triplet** vector instead of the scalar `alm_slack` in both `dual.forward` and
`dual.update`. Gated behind `cfg.calibrate_slacks: true` (default `false` — every existing config
keeps the old scalar behavior unless it opts in) and duck-typed via `hasattr`, matching
`calibrate_collider_margins`; not yet implemented for the continuous estimator, whose floors were
all well under its configs' slack in the same notebook. The slack actually used by a given `fit()`
call is exposed afterward as `self.alm_slack_` (a `Tensor` when calibrated, the plain scalar
otherwise, `None` if unconstrained).

Two deliberate design choices:

- **Primal/dual split.** The primal step (`optimizer.step()`) runs every minibatch
  on the noisy batch-level constraint estimate, with `dual.forward` treating λ as a
  fixed constant. The dual step runs once per epoch, on the honest full-training-set
  violation — decoupling a stable, low-variance dual update from noisy per-batch
  gradients. This is why `alm_lr` (dual learning rate, default `5.0`) is large
  relative to a typical dual step size: it only gets one update per epoch, not one
  per batch, so it needs to move further each time.
- **Slack.** Every constraint value either entry point ever sees is
  `(c − alm_slack)`, not `c` directly (`alm_slack` default `0.01`, but see below —
  it must be subtracted in `dual.forward` too, not only `dual.update`). Since every
  violation is one-sided (≥ 0 always), without slack the duals can only ratchet
  upward — there is no "over-satisfied" direction to pull them back down.
  Subtracting a small slack lets λ shrink again once the violation is within
  estimation noise of zero.

**Bug found and fixed (2026-09-16): `alm_slack` was silently a no-op on the primal
step.** An earlier version of the loop above subtracted `alm_slack` only in
`dual.update(c_full - alm_slack)`, not in the per-minibatch
`dual.forward(loss, constraints)` call — i.e. `constraints` there was the *raw*,
un-slacked violation. Under `"hpr"` (the augmentation every current config uses),
the primal step's weight on `∂c/∂θ` is `max(0, λ + ρ·c)`; with an un-slacked `c`
this is `> 0` on essentially every batch (violations are non-negative by
construction) even while `λ == 0`, since `max(0, 0 + ρ·c) = ρ·c` whenever `c > 0`.
So the primal step was *always* pulling the model toward exactly zero violation,
no matter what `alm_slack` was configured to — confirmed numerically: with `λ=0`,
`ρ=1`, a raw `c=0.05`, `dual.forward` gives a nonzero gradient of `0.05` on that
constraint regardless of whether `alm_slack` is `0.01` or `10`; subtracting
`alm_slack` first (`c - alm_slack`) is what makes that gradient `0.0` once the
violation is truly within slack. This is exactly what surfaced empirically: raising
`alm_slack` to `10.` (meant to make a target's constraints never bind, as a sanity
check that `con_*` should then match `uncon_*`) left the two arms' training
trajectories just as different as at the default `alm_slack=0.01` — because the
`10.` never reached the primal step at all, only the (mostly idle either way) dual
update. It also explains why `con_*` underperforms `uncon_*` on train error, not
just test: the model is fighting an unconditional pressure toward zero violation on
every batch, regardless of the tolerance the config asked for. Fixed by subtracting
`alm_slack` in the `dual.forward` call as well, so both entry points agree on what
"feasible" means for a given constraint (see [`fit()`](causal_estimator_base.py#L419-L436)).

`is_ineq=True` (current setting on this branch) tells the `humancompatible-train`
ALM to treat each triplet as an inequality constraint `c(θ) ≤ 0` rather than an
equality: duals are floored at 0 and relax toward 0 as a constraint becomes
strictly satisfied, instead of being penalized symmetrically in both directions.
Since these violations can never go negative, equality-style symmetric penalization
has no real meaning here — `is_ineq=True` is the mode that actually matches a
quantity with only one feasible side.

**Optimizer.** `use_alm=True` now pairs `ALM` with a **plain** `torch.optim.Adam` — no
`MoreauEnvelope` wrapping ([causal_estimator_base.py:370-375](causal_estimator_base.py#L370-L375)).
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

`fit()` now exposes this (and the raw per-epoch trace) as fitted attributes, so a caller
doesn't have to re-instrument the loop to check any of the above:

- `best_epoch_` — the epoch actually kept. Equal to `n_epochs_run_ - 1` means training was
  still improving when it stopped (more epochs might help); much earlier means later epochs
  were actively getting worse (e.g. a still-climbing dual dragging the model away from a fit
  it already had — exactly the pattern in §5's worked example).
- `n_epochs_run_` — the `n_epochs` this call actually used, to compare against `best_epoch_`.
- `train_history_` — a `list[dict]`, one entry per epoch, always `{'epoch', 'selection_loss'}`
  and — only when `use_alm=True` — also `'violation_mean'`, `'dual_mean'`, `'dual_max'`,
  `'n_duals_saturated'` (how many duals sit at the ALM's `dual_range` upper clamp right after
  that epoch's `dual.update()`).

  `test_all_networks*.py` / `test_constraints*.py` log this per (fold, epoch) into
  `<out>_history.csv`, and `train_history_[-1]` aggregated (mean/max/sum, see
  `_aggregate_fold_diagnostics`) into the main table's `dual_mean`/`dual_max`/
  `n_duals_saturated` columns — **from the actual `solver_cfg.n_runs` cross-validation fold
  models that produced `train_error`/`test_error`** (via `evaluate()`'s `return_estimators=True`
  call into `run_feature_selection_scikit`), not a separate model fit on 100% of the data. That
  separate-model approach (still what `test_shift*.py` uses, since it has no CV to hook into —
  it fits one model directly) turned out to be actively misleading: for one `deep_mlp`/`con_all`
  run, 4 of 5 CV folds had their dual move away from 0 and collapsed to a near-useless fit
  (`selection_loss` ≈ 1.0, `best_epoch` as early as epoch 3 of 50) while the one bystander model
  built solely for diagnostics happened to land in the 1-of-5 regime that converged cleanly
  (`dual_max` = 0.0) — reporting a reassuring `violation`/`dual_max` right next to a `train_error`
  of 0.82. Averaging across folds instead of using a bystander model is what makes a saturated
  dual show up directly in the sweep's own CSV, attached to the row it actually explains.

## 7. Where each piece lives

| Piece | File |
|---|---|
| Triplet enumeration (graph-only) | [`causal_triplets.py`](causal_triplets.py) |
| Shared ALM/Moreau training loop | [`causal_estimator_base.py`](causal_estimator_base.py) (`CausalConstrainedPredictor.fit`) |
| Discrete violation math | [`discrete_estimator.py`](discrete_estimator.py) (`TripletConstraintsFn`) |
| Continuous violation math | [`continuous_estimator.py`](continuous_estimator.py) (`ContinuousTripletConstraintsFn`) |
| ALM dual optimizer itself | `humancompatible.train.dual_optim.ALM` (external package, `pip install humancompatible-train`) |
