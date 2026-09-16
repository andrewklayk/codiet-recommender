# codiet_recommender

Predictors for CoDiet health-biomarker targets (e.g. CRP, glucose, lipids)
from food, microbiome, lipidomics and body-composition features, constrained
by a causal DAG (`w_est`, learned via NOTEARS/MILP) so that the model's
predictions respect the conditional independencies the DAG implies, not just
the training loss.

Two branches implement the causal constraint differently:

- **`main`** — one global linear equation derived algebraically from `w_est`
  (`nn_lagrangian.py`, `xgboost_lagrangian.py`), enforced as a single
  Lagrangian penalty term.
- **`expectations`** (this branch) — every chain/fork/collider **triplet** in
  the DAG that involves the target becomes its own differentiable
  conditional-independence violation term, all driven toward zero together by
  an augmented-Lagrangian dual optimizer (`ALM` from
  [`humancompatible-train`](https://github.com/andrewklayk/humancompatible-train)).
  See [`CONSTRAINTS.md`](CONSTRAINTS.md) for the full mechanism (how triplets
  are enumerated, the violation math, and exactly how ALM is wired into
  training).

## Core pieces

| Piece | File |
|---|---|
| Triplet enumeration from the DAG | [`causal_triplets.py`](causal_triplets.py) |
| Shared ALM/optimizer training loop | [`causal_estimator_base.py`](causal_estimator_base.py) (`CausalConstrainedPredictor.fit`) |
| Discrete-target estimator + violation math | [`discrete_estimator.py`](discrete_estimator.py) (`DiscreteRecommenderPredictor`, `TripletConstraintsFn`) |
| Continuous-target estimator + violation math | [`continuous_estimator.py`](continuous_estimator.py) (`ContinuousRecommenderPredictor`, `ContinuousTripletConstraintsFn`) |
| Pluggable NN backbones | [`discrete_networks.py`](discrete_networks.py), [`continuous_networks.py`](continuous_networks.py) — `mlp`, `deep_mlp`, `onehot_mlp`\*, `embedding_mlp`\*, `transformer` (\*discrete only) |
| DAG structure learning | [`notears_util.py`](notears_util.py), [`solve_milp.py`](solve_milp.py) |
| Diet/food-swap recommendation on top of a fitted predictor | [`recommender.py`](recommender.py), [`recommender_nn.py`](recommender_nn.py), [`recommender_estimator.py`](recommender_estimator.py) |

## Experiments

There are two kinds of experiment in this repo: the **real-data pipeline**,
which fits a recommender to an actual CoDiet/CDS biomarker target, and a
**synthetic validation suite**, which checks — on data generated from a known
ground-truth DAG, where "correct" is well-defined — whether the causal
constraints actually help and isolates *why*.

### Real-data pipeline: `run_experiments.py`

Hydra-driven entry point (`experiments_conf/config.yaml`, defaults to
`solver: mark`, `problem: codiet`) that loads a CoDiet dataset
(`data_helper.load_all_data`), fits the configured estimator to predict a
target biomarker (default `CRP (mg/dL)`, see
[`experiments_conf/problem/codiet.yaml`](experiments_conf/problem/codiet.yaml))
from a fixed feature list spanning food composition, body composition,
serum/urine metabolomics, lipidomics and microbiome features, and logs
metrics/artifacts to MLflow. Other `problem/*.yaml` files swap in different
targets (`codiet_glu`, `codiet_hba`, `codiet_ldl`, `codiet_hdl`,
`codiet_trig`, `codiet_diast`, `codiet_syst`, `codiet_whtr`), different
datasets (`sachs`, `cds_*`, `FRED_16country_quarterly/industry_eu_*`), or the
synthetic `er_graph` problem used by the scripts below.

```bash
python run_experiments.py                              # default: mark solver on codiet/CRP
python run_experiments.py solver=discrete_constrained problem=codiet_glu
```

### Synthetic validation suite

All of these generate data from a **known** DAG (so ground-truth conditional
independencies are exact, not estimated) and compare training regimes across
every discrete backbone (`mlp`, `deep_mlp`, `onehot_mlp`, `embedding_mlp`,
`transformer`) or continuous backbone (`mlp`, `deep_mlp`, `transformer`) via
paired `discrete_*.yaml` / `continuous_*.yaml` solver configs. Each has a
`*_continuous.py` twin (linear-Gaussian SEM + MSE) alongside the discrete
version (CPT-sampled DAG + classification error).

Settings share one label scheme across all three scripts,
`<training regime>_<feature set>`:

| Setting | Regime | Features |
|---|---|---|
| `uncon_all` | vanilla Adam | all other variables |
| `con_all` | ALM-constrained | all other variables |
| `uncon_ME` | vanilla + Moreau-envelope-smoothed optimizer, **no** constraints | all other variables |
| `uncon_parents` | vanilla Adam | target's direct parents only |
| `uncon_mb` | vanilla Adam | target's Markov blanket (parents + children + co-parents) |
| `con_mb` | ALM-constrained | target's Markov blanket |

`uncon_ME` is a *load-bearing control*, not a curiosity: it isolates how much
of `con_all`'s behaviour comes from the constraint itself versus incidental
optimizer differences. `uncon_all` is equally load-bearing as the reference
every all-features method must beat — without it, `uncon_ME`/`con_all` are
the only all-features rows and look artificially strong next to
feature-restricted rows for reasons that have nothing to do with the
constraint.

- **[`test_all_networks.py`](test_all_networks.py)** /
  **[`test_all_networks_continuous.py`](test_all_networks_continuous.py)** —
  the main sweep. Random Erdős–Rényi DAGs (`er_graph.sample_dag` /
  `continuous_er_graph.sample_dag`), every node taken as target in turn,
  averaged over `--n-seeds` independently sampled graphs. Reports train/test
  error and constraint violation per backbone × setting, as long-form CSV and
  a formatted XLSX (best setting per row bolded).

  ```bash
  python test_all_networks.py --n-seeds 10
  python test_all_networks_continuous.py --n-seeds 10
  ```

- **[`test_constraints.py`](test_constraints.py)** /
  **[`test_constraints_continuous.py`](test_constraints_continuous.py)** —
  same settings, but on tiny hand-built 3-node graphs where *exactly one*
  causal structure (chain `A→B→C`, fork `A←B→C`, or collider `A→B←C`) is
  active at a time, with every node taking a turn as target. Isolates
  structure-specific failure modes the random-DAG sweep averages away — e.g.
  a parents-only baseline fails outright when the target is a chain root
  (no parents) even though its descendants are predictive.

- **[`test_shift.py`](test_shift.py)** /
  **[`test_shift_continuous.py`](test_shift_continuous.py)** — the
  distribution-shift test, on a fixed chain `A→B→C` with target `C`. The
  target's own mechanism `P(C|B)` is held fixed between train and test, but
  the upstream mechanisms `P(A)` and `P(B|A)` are re-rolled for the test set,
  flipping the in-sample `A`–`C` correlation. In-distribution the constraint
  is expected to add little (an unconstrained fit on DAG-sampled data already
  respects the independencies approximately); the constraint is meant to pay
  off exactly here, where the numbers shift but the graph doesn't — a model
  that leaned on the now-broken `A`–`C` shortcut should degrade more than one
  restricted to (or penalized toward) the Markov blanket `{B}`.

  ```bash
  python test_shift.py --n-seeds 30 --n-train 150
  ```

All six scripts accept `--n-epochs`/`--n-runs`/`--targets` overrides for a
fast smoke test, e.g. `python test_all_networks.py --n-seeds 2 --n-epochs 2
--n-runs 2 --targets X0,X1`.

## Requirements

Python environment with `torch`, `hydra-core`, `omegaconf`, `mlflow`,
`scikit-learn`, `xgboost`, `networkx`, `pandas`, `openpyxl`, and
[`humancompatible-train`](https://github.com/andrewklayk/humancompatible-train)
(`pip install humancompatible-train`) for the `ALM` dual optimizer used by
every `*_constrained.yaml` / `use_alm: true` config.
