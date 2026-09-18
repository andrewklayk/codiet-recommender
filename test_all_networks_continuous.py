"""Compare CONTINUOUS network backbones across five training regimes and
several er_graph seeds. Continuous twin of test_all_networks.py -- same
design, over a linear-Gaussian SEM (continuous_er_graph.sample_dag) instead of
discrete CPTs, scored by MSE instead of classification error.

For each backbone family we cross two independent axes:

  * constraints : vanilla MSE (use_alm=false) vs causally constrained ALM
                  training (use_alm=true)
  * features    : all features / the target's Markov blanket (parents +
                  children + co-parents), via restrict_to_markov_blanket

giving four settings per backbone:

    uncon_all   vanilla,     all features
    con_all     constrained, all features
    uncon_mb    vanilla,     Markov blanket
    con_mb      constrained, Markov blanket

plus a fifth setting, uncon_ME, which isolates the effect of swapping in the
Moreau-envelope-smoothed optimizer alone (same "all features", no ALM
duals/constraints) -- a control for how much of con_all's effect (if any) is
just "different optimizer" rather than "causal constraint":

    uncon_ME    vanilla + Moreau-envelope optimizer, all features

Backbone -> (unconstrained config, constrained config, Moreau-only config):

    mlp         -> continuous             / continuous_constrained             / continuous_me
    deep_mlp    -> continuous_deep        / continuous_deep_constrained        / continuous_deep_me
    transformer -> continuous_transformer / continuous_transformer_constrained / continuous_transformer_me

No onehot_mlp / embedding_mlp siblings: continuous_networks.py has no
categorical-only backbones (inputs are continuous, see its module docstring).

Every (backbone, setting, target) is evaluated on N_SEEDS independently
generated er_graph datasets. Results are written to:

    <out>.csv   — raw long-form rows (seed, target, network, setting, errors)
    <out>.xlsx  — sheet 'summary_overall' : avg over seeds AND features, one row
                                            per backbone, the five settings side
                                            by side for train/test error AND
                                            test-set constraint violation; best
                                            (lowest) per metric bold.
                  sheet 'by_feature_test' : avg test error over seeds, one row per
                                            target, five settings per backbone;
                                            best per backbone bold.
                  sheet 'by_feature_train': same for train error.
                  sheet 'violation_overall' : avg test-set constraint violation
                                            over seeds & features, network x
                                            setting; lowest per network bold.
                  sheet 'violation_by_feature': avg test-set violation per target.
                  sheet 'raw'             : every individual run.

Constraint violation measures how much each fitted model's predictions break the
causal-independence constraints implied by w_est (see
ContinuousRecommenderPredictor.constraint_violation, via ContinuousTripletConstraintsFn);
it is logged for every setting, constrained or not.

violation_test / violation_train restrict that computation to just a fold's
own CV test rows, or just the rows it actually trained on (excluding any
val_fraction held-out rows), so a gap between them -- constraint satisfaction
in-sample vs. whether it generalizes to unseen data -- is visible directly.
The summary sheets and the console printout use violation_test as the
headline number, for the same reason test_error (not train_error) is the
headline accuracy metric.

The 'raw' sheet (and the CSV) also carries, per row: n_triplets and, for each
of violation_test/violation_train, its own chain/fork/collider breakdown (one
structure's violations can't hide in the averaged total -- a target's
colliders can be satisfiable while its forks aren't, or vice versa);
best_epoch and n_epochs_run (best_epoch == n_epochs_run - 1 means training
was still improving when it stopped); and, for use_alm=True settings,
dual_mean/dual_max/n_duals_saturated. n_duals_saturated > 0 means a dual is
sitting at the optimizer's own clamp: the direct symptom of a target whose
triplets can't jointly be driven below alm_slack, which makes its dual
ratchet without bound and its Σλᵀc penalty term swamp the loss (see
CONSTRAINTS.md sec. 5).

ALL of these diagnostics are aggregated over the SAME solver_cfg.n_runs
cross-validation folds that produced train_error/test_error, NOT a separate
model fit on 100% of the data -- a target's fold models can range from
perfectly converged to fully collapsed (most inits fine, one fold's dual
runs away and wrecks that fold's fit), and a lone bystander model tells you
nothing about which happened to the folds actually being scored.
violation_*/dual_mean/best_epoch are averaged across folds; dual_max/
n_duals_saturated take the worst fold's max/sum instead, so one collapsed
fold isn't averaged away by four fine ones. See evaluate()'s docstring and
CONSTRAINTS.md sec. 5/6.

    <out>_history.csv — the full per-epoch trace behind that aggregate
                        snapshot: one row per (seed, target, network, setting,
                        fold, epoch) with selection_loss and, for use_alm=True
                        settings, violation_mean/dual_mean/dual_max/
                        n_duals_saturated at that epoch, for EVERY one of the
                        n_runs fold models (tagged by 'fold'). This is what
                        shows a dual's actual climb in the specific fold(s)
                        that diverged, rather than only the epoch-level
                        aggregate or a bystander model's trajectory.

Usage:
    python test_all_networks_continuous.py
    python test_all_networks_continuous.py --n-seeds 3 --networks mlp,transformer
    python test_all_networks_continuous.py --seed 7      # only er_graph seed 7
    # fast smoke test:
    python test_all_networks_continuous.py --n-seeds 2 --n-epochs 2 --n-runs 2 --targets X0,X1
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from openpyxl.styles import Font

from recommender_utils import run_feature_selection_scikit

logging.basicConfig(level=logging.WARNING)

CONF_DIR = Path(__file__).parent / "experiments_conf"

# (label, unconstrained solver config, constrained solver config,
#  Moreau-envelope-only solver config)
NETWORKS = [
    ("mlp",         "continuous",             "continuous_constrained",             "continuous_me"),
    ("deep_mlp",    "continuous_deep",        "continuous_deep_constrained",        "continuous_deep_me"),
    ("transformer", "continuous_transformer", "continuous_transformer_constrained", "continuous_transformer_me"),
]

# (setting label, which config variant, feature-restriction mode)
#   variant: 'uncon' = vanilla Adam, 'con' = ALM-constrained, 'me' = vanilla
#            but with the Moreau-envelope optimizer (no constraints)
#   mode: 'none' = all features, 'markov_blanket' = parents + children + co-parents
SETTINGS = [
    ("uncon_all", "uncon", "none"),
    ("con_all",   "con",   "none"),
    ("uncon_ME",  "me",    "none"),
    ("uncon_mb",  "uncon", "markov_blanket"),
    ("con_mb",    "con",   "markov_blanket"),
]
SETTING_ORDER = [s[0] for s in SETTINGS]


def build_dataset(problem_cfg, seed):
    """Generate the continuous er_graph dataset + true DAG for a given seed.

    Returns (prep_data, w_est, column_names).
    """
    if problem_cfg.name != "er_graph":
        raise ValueError(
            f"test_all_networks_continuous only supports problem=er_graph, got "
            f"'{problem_cfg.name}'."
        )

    from continuous_er_graph import sample_dag

    X, dag = sample_dag(
        n_samples=problem_cfg.n_samples,
        d=problem_cfg.d,
        s0=problem_cfg.s0,
        graph_type=problem_cfg.get("graph_type", "ER"),
        seed=seed,
        return_dag=True,
    )
    col_names = [f"X{i}" for i in range(problem_cfg.d)]
    prep_data = pd.DataFrame(X, columns=col_names)
    return prep_data, dag["B"], col_names


def load_solver(name, n_epochs=None, n_runs=None):
    """Load a solver config, optionally overriding epochs / CV folds."""
    cfg = OmegaConf.load(CONF_DIR / "solver" / f"{name}.yaml")
    OmegaConf.set_struct(cfg, False)  # allow adding restrict_to_parents
    if n_epochs is not None:
        cfg.n_epochs = n_epochs
    if n_runs is not None:
        cfg.n_runs = n_runs
    return cfg


# Diagnostic columns every experiment script attaches to each row, on top of
# train/test error: constraint violation broken down by causal structure (a
# target's colliders can be satisfiable while its forks aren't, or vice versa
# -- averaging them into one number hides that), how many triplets that
# breakdown is over, which epoch fit() actually kept, and -- when
# use_alm=True -- the ALM dual variables' final state. n_duals_saturated > 0
# (a dual sitting at the optimizer's own clamp) is the direct symptom of a
# constraint whose achievable violation floor sits above alm_slack: see
# CONSTRAINTS.md sec. 5 for the worked example this was built to catch.
EMPTY_DIAGNOSTICS = {
    "n_triplets": 0,
    # violation_test/violation_train (set in evaluate(), not here -- they
    # need per-fold row indices this dict has no access to) restrict
    # constraint_violation() to just a fold's CV test rows, or just its true
    # training rows, each with its own chain/fork/collider breakdown so one
    # structure's violations can't hide inside the total.
    "violation_test": float("nan"), "violation_train": float("nan"),
    "violation_test_chain": float("nan"), "violation_test_fork": float("nan"),
    "violation_test_collider": float("nan"),
    "violation_train_chain": float("nan"), "violation_train_fork": float("nan"),
    "violation_train_collider": float("nan"),
    "best_epoch": -1, "n_epochs_run": 0,
    "dual_mean": float("nan"), "dual_max": float("nan"),
    "n_duals_saturated": 0,
}


def _by_type(cv, key):
    """Pull one chain/fork/collider entry out of a constraint_violation()
    result's by_type dict, NaN if that type has no triplets for this target."""
    v = cv["by_type"].get(key, pd.NA)
    return float("nan") if v is pd.NA else float(v)


def diagnostics_from_model(model):
    """Pull every post-hoc, model-native diagnostic off an already-fitted
    model into one dict (see EMPTY_DIAGNOSTICS for the field list /
    rationale). violation_train/violation_test and their chain/fork/collider
    breakdowns are NOT set here -- they need per-fold row indices this
    function has no access to; evaluate() fills them in from its own
    constraint_violation() calls on each fold's train/test rows."""
    out = dict(EMPTY_DIAGNOSTICS)
    out.update({
        "best_epoch": int(getattr(model, "best_epoch_", -1)),
        "n_epochs_run": int(getattr(model, "n_epochs_run_", 0)),
    })
    history = getattr(model, "train_history_", None)
    if history:
        last = history[-1]
        out["dual_mean"] = float(last.get("dual_mean", float("nan")))
        out["dual_max"] = float(last.get("dual_max", float("nan")))
        out["n_duals_saturated"] = int(last.get("n_duals_saturated", 0))
    return out


def _aggregate_fold_diagnostics(per_fold):
    """Combine one diagnostics_from_model() dict per CV fold into one row.

    violation_*/dual_mean/best_epoch: mean across folds (typical behaviour).
    dual_max/n_duals_saturated: max/sum across folds instead of mean -- a
    single collapsed fold (see CONSTRAINTS.md sec. 5/6: most inits can
    converge cleanly while one fold's dual runs away) would otherwise be
    averaged down to near-invisible by four fine folds. n_triplets/
    n_epochs_run are structural (same graph/config for every fold), so just
    read off the first fold rather than averaged.
    """
    if not per_fold:
        return dict(EMPTY_DIAGNOSTICS)

    def _finite(key):
        return [d[key] for d in per_fold if not (isinstance(d[key], float) and np.isnan(d[key]))]

    def _mean(key):
        vals = _finite(key)
        return float(np.mean(vals)) if vals else float("nan")

    def _max(key):
        vals = _finite(key)
        return float(np.max(vals)) if vals else float("nan")

    return {
        "n_triplets": int(per_fold[0]["n_triplets"]),
        "violation_test": _mean("violation_test"),
        "violation_train": _mean("violation_train"),
        "violation_test_chain": _mean("violation_test_chain"),
        "violation_test_fork": _mean("violation_test_fork"),
        "violation_test_collider": _mean("violation_test_collider"),
        "violation_train_chain": _mean("violation_train_chain"),
        "violation_train_fork": _mean("violation_train_fork"),
        "violation_train_collider": _mean("violation_train_collider"),
        "best_epoch": _mean("best_epoch"),
        "n_epochs_run": int(per_fold[0]["n_epochs_run"]),
        "dual_mean": _mean("dual_mean"),
        "dual_max": _max("dual_max"),
        "n_duals_saturated": int(sum(d["n_duals_saturated"] for d in per_fold)),
    }


def evaluate(prep_data, w_est, row_and_col_names, target, features, solver_cfg, seed):
    """Run one (target, solver) evaluation via cross-validation.

    Returns (train_err, test_err, diag, history):
        train_err, test_err : mean over solver_cfg.n_runs CV folds, as before.
        diag    : diagnostics built from the SAME fold-fitted estimators that
                  produced train_err/test_err -- not a separate model fit on
                  100% of the data (that approach was a bystander: a
                  target's fold models can range from perfectly converged to
                  fully collapsed, and a lone extra model tells you nothing
                  about which happened to the ones actually being scored;
                  see CONSTRAINTS.md sec. 5/6). Aggregated via
                  _aggregate_fold_diagnostics.
        history : per-epoch trace, one entry per (fold, epoch), each tagged
                  with 'fold' -- the caller adds its own seed/target/network/
                  setting id columns on top.
    """
    torch.manual_seed(seed)  # reproducible network init/training per seed
    (_best, train_err, test_err,
     *_rest, estimators, indices) = run_feature_selection_scikit(
        prep_data.copy(),
        solver_cfg.model_name,
        solver_cfg.custom_objective,
        target,
        w_est, row_and_col_names,
        solver_cfg.n_runs,
        len(features),
        features,
        solver_cfg=solver_cfg,
        return_estimators=True,
    )

    X_full = prep_data[features]
    per_fold = []
    for est, tr_idx, te_idx in zip(estimators, indices["train"], indices["test"]):
        d = diagnostics_from_model(est)
        # violation_test/violation_train: constraint_violation() restricted
        # to this fold's own CV test rows / true training rows, via the
        # positional indices run_feature_selection_scikit's
        # cross_validate(return_indices=True) hands back -- these index into
        # X_selected there, which is the SAME row order as X_full here as
        # long as neither of that function's two dropna calls drops a
        # row/column for this dataset (true for the synthetic er_graph data
        # this script uses; would silently misalign otherwise).
        # _train_row_mask_ further excludes any val_fraction held-out rows
        # from "true training rows", same as the train_error fix.
        X_test_fold = X_full.iloc[te_idx]
        cv_test = est.constraint_violation(X_test_fold)
        d["violation_test"] = float(cv_test["total"])
        d["violation_test_chain"] = _by_type(cv_test, "chain")
        d["violation_test_fork"] = _by_type(cv_test, "fork")
        d["violation_test_collider"] = _by_type(cv_test, "collider")
        d["n_triplets"] = int(cv_test["n_triplets"])

        X_train_fold = X_full.iloc[tr_idx]
        mask = getattr(est, "_train_row_mask_", None)
        if mask is not None:
            X_train_fold = X_train_fold.iloc[mask]
        cv_train = est.constraint_violation(X_train_fold)
        d["violation_train"] = float(cv_train["total"])
        d["violation_train_chain"] = _by_type(cv_train, "chain")
        d["violation_train_fork"] = _by_type(cv_train, "fork")
        d["violation_train_collider"] = _by_type(cv_train, "collider")
        per_fold.append(d)
    history = [{"fold": i, **h} for i, est in enumerate(estimators)
              for h in getattr(est, "train_history_", [])]
    diag = _aggregate_fold_diagnostics(per_fold)
    return float(train_err), float(test_err), diag, history


def bold_min_cells(ws, df, groups):
    """Bold, per data row, the cell with the smallest value within each group.

    `groups` is a list of column-name lists; within each list the row-wise
    minimum cell is set to bold. Assumes df was written with its index (index in
    excel column A, data from column B, header on row 1).
    """
    pos = {c: i for i, c in enumerate(df.columns)}
    for r, (_, row) in enumerate(df.iterrows()):
        for group in groups:
            vals = [(c, row[c]) for c in group if pd.notna(row[c])]
            if not vals:
                continue
            best = min(vals, key=lambda t: t[1])[0]
            ws.cell(row=r + 2, column=pos[best] + 2).font = Font(bold=True)


def wide_by_feature(g, net_labels, targets_sorted, metric):
    """Wide per-target table for one metric: <net>_<setting>_<metric> columns.

    Returns (DataFrame, groups) where each group is the five settings of one
    backbone (for bold-min highlighting of the best setting per backbone).
    """
    val_col = f"{metric}_error"
    df = pd.DataFrame(index=targets_sorted)
    df.index.name = "feature"
    groups = []
    for net in net_labels:
        cols = []
        for setting in SETTING_ORDER:
            col = f"{net}_{setting}_{metric}"
            df[col] = g.loc[(net, setting)].reindex(targets_sorted)[val_col].values
            cols.append(col)
        groups.append(cols)
    return df.round(4), groups


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--problem", default="er_graph")
    parser.add_argument("--out", default="network_comparison_continuous",
                        help="output basename for .csv and .xlsx")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="number of er_graph seeds, runs 0..n-1 (default 10)")
    parser.add_argument("--seed", type=int, default=None,
                        help="run only this single er_graph seed (overrides --n-seeds)")
    parser.add_argument("--networks", default=None,
                        help="comma-separated subset of backbone labels "
                             f"(default all: {','.join(n[0] for n in NETWORKS)})")
    parser.add_argument("--targets", default=None,
                        help="comma-separated subset of target columns "
                             "(default: all nodes)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="override n_epochs in every solver (for quick runs)")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="override n_runs (CV folds) in every solver")
    args = parser.parse_args()

    problem_cfg = OmegaConf.load(CONF_DIR / "problem" / f"{args.problem}.yaml")

    nets = NETWORKS
    if args.networks:
        wanted = {s.strip() for s in args.networks.split(",")}
        nets = [n for n in NETWORKS if n[0] in wanted]
        if not nets:
            raise SystemExit(f"No backbones match {wanted}")
    net_labels = [n[0] for n in nets]

    # pre-load all solver configs (one per variant per backbone)
    solvers = {}  # (label, variant) -> cfg
    for label, uncon, con, me in nets:
        solvers[(label, "uncon")] = load_solver(uncon, args.n_epochs, args.n_runs)
        solvers[(label, "con")] = load_solver(con, args.n_epochs, args.n_runs)
        solvers[(label, "me")] = load_solver(me, args.n_epochs, args.n_runs)

    seeds = [args.seed] if args.seed is not None else list(range(args.n_seeds))
    n_runs_eff = solvers[(net_labels[0], "uncon")].n_runs
    print(f"Backbones: {net_labels}")
    print(f"Seeds: {seeds} | {n_runs_eff}-fold CV | settings: {SETTING_ORDER}\n")

    records = []
    history_records = []
    for seed in seeds:
        prep_data, w_est, col_names = build_dataset(problem_cfg, seed)
        row_and_col_names = list(prep_data.columns)
        targets = col_names
        if args.targets:
            wanted = {s.strip() for s in args.targets.split(",")}
            targets = [t for t in col_names if t in wanted]

        for target in targets:
            features = [c for c in col_names if c != target]
            for label in net_labels:
                for setting, variant, mode in SETTINGS:
                    cfg = solvers[(label, variant)]
                    cfg.restrict_to_parents = False
                    cfg.restrict_to_markov_blanket = (mode == "markov_blanket")
                    try:
                        tr, te, diag, history = evaluate(
                            prep_data, w_est, row_and_col_names,
                            target, features, cfg, seed)
                    except Exception as exc:  # keep the long sweep alive
                        print(f"  [WARN] seed={seed} {target} {label}/{setting} "
                              f"failed: {exc}")
                        tr, te = float("nan"), float("nan")
                        diag, history = dict(EMPTY_DIAGNOSTICS), []
                    id_cols = {"seed": seed, "target": target,
                              "network": label, "setting": setting}
                    history_records.extend({**id_cols, **h} for h in history)
                    records.append({
                        "seed": seed, "target": target, "network": label,
                        "setting": setting, "variant": variant,
                        "features": mode,
                        "train_error": tr, "test_error": te,
                        **diag,
                    })
                    print(f"  seed={seed} {target:>3s} {label:14s} {setting:9s} "
                          f"train={tr:.4f} test={te:.4f} "
                          f"viol_tr={diag['violation_train']:.4f} "
                          f"viol_te={diag['violation_test']:.4f} "
                          f"trip={diag['n_triplets']} best_ep={diag['best_epoch']}"
                          f"/{diag['n_epochs_run']} dual_max={diag['dual_max']:.2f} "
                          f"sat={diag['n_duals_saturated']}")

    raw = pd.DataFrame(records)
    csv_path = Path(f"{args.out}.csv")
    raw.to_csv(csv_path, index=False)

    # Full per-epoch trace behind the last-epoch snapshot in `raw` -- see the
    # module docstring's <out>_history.csv entry. Unconstrained settings'
    # entries only have {'epoch', 'selection_loss'}; pandas fills the
    # constrained-only columns (dual_mean, ...) with NaN for those rows.
    history_csv_path = Path(f"{args.out}_history.csv")
    pd.DataFrame(history_records).to_csv(history_csv_path, index=False)

    targets_sorted = sorted(raw["target"].unique(), key=lambda s: int(s[1:]))

    # ---- per-feature summaries (avg over seeds), test and train ----
    g = raw.groupby(["network", "setting", "target"])[
        ["train_error", "test_error"]].mean()
    by_test, by_test_groups = wide_by_feature(g, net_labels, targets_sorted, "test")
    by_train, by_train_groups = wide_by_feature(g, net_labels, targets_sorted, "train")

    # ---- overall summary: avg over seeds AND features ----
    # ('violation' is a third metric block alongside train/test so constraint
    # satisfaction is visible in the same headline table, not only in the
    # dedicated violation_overall/violation_by_feature sheets below; sourced
    # from violation_test -- whether constraint satisfaction generalizes --
    # for the same reason the headline error metric is test, not train)
    metric_cols = {"train": "train_error", "test": "test_error",
                   "violation": "violation_test"}
    o = raw.groupby(["network", "setting"])[list(metric_cols.values())].mean()
    overall = pd.DataFrame(index=net_labels)
    overall.index.name = "network"
    for metric, col in metric_cols.items():
        for setting in SETTING_ORDER:
            overall[f"{setting}_{metric}"] = [
                o.loc[(n, setting), col] for n in net_labels]
    overall = overall.round(4)
    overall_groups = [[f"{s}_train" for s in SETTING_ORDER],
                      [f"{s}_test" for s in SETTING_ORDER],
                      [f"{s}_violation" for s in SETTING_ORDER]]

    # ---- constraint-violation summaries (avg over seeds); test-set, since
    # that's what says whether constraint satisfaction generalizes ----
    gv = raw.groupby(["network", "setting", "target"])["violation_test"].mean()
    viol_by_feat = pd.DataFrame(index=targets_sorted)
    viol_by_feat.index.name = "feature"
    viol_feat_groups = []
    for net in net_labels:
        cols = []
        for setting in SETTING_ORDER:
            col = f"{net}_{setting}"
            viol_by_feat[col] = gv.loc[(net, setting)].reindex(targets_sorted).values
            cols.append(col)
        viol_feat_groups.append(cols)
    viol_by_feat = viol_by_feat.round(4)

    ov = raw.groupby(["network", "setting"])["violation_test"].mean()
    viol_overall = pd.DataFrame(index=net_labels)
    viol_overall.index.name = "network"
    for setting in SETTING_ORDER:
        viol_overall[setting] = [ov.get((n, setting), np.nan) for n in net_labels]
    viol_overall = viol_overall.round(4)

    # ---- write xlsx with bold-min (best setting) formatting ----
    xlsx_path = Path(f"{args.out}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="summary_overall")
        by_test.to_excel(writer, sheet_name="by_feature_test")
        by_train.to_excel(writer, sheet_name="by_feature_train")
        viol_overall.to_excel(writer, sheet_name="violation_overall")
        viol_by_feat.to_excel(writer, sheet_name="violation_by_feature")
        raw.to_excel(writer, sheet_name="raw", index=False)
        bold_min_cells(writer.sheets["summary_overall"], overall, overall_groups)
        bold_min_cells(writer.sheets["by_feature_test"], by_test, by_test_groups)
        bold_min_cells(writer.sheets["by_feature_train"], by_train, by_train_groups)
        bold_min_cells(writer.sheets["violation_overall"], viol_overall,
                       [SETTING_ORDER])
        bold_min_cells(writer.sheets["violation_by_feature"], viol_by_feat,
                       viol_feat_groups)

    print("\n==== Overall test error (avg over seeds & features; lower is better) ====")
    print(overall[[f"{s}_test" for s in SETTING_ORDER]].to_string())
    print("\n==== Overall test-set constraint violation (avg over seeds & features; lower is better) ====")
    print(overall[[f"{s}_violation" for s in SETTING_ORDER]].to_string())
    print(f"\nWrote {csv_path.resolve()}")
    print(f"Wrote {xlsx_path.resolve()}  (bold = best of the five settings)")
    print(f"Wrote {history_csv_path.resolve()}  (full per-epoch trace)")


if __name__ == "__main__":
    main()
