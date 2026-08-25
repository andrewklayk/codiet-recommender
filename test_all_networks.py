"""Compare discrete network backbones across four training regimes and several
er_graph seeds.

For each backbone family we cross two independent axes:

  * constraints : vanilla cross-entropy (use_alm=false) vs causally constrained
                  ALM training (use_alm=true)
  * features    : all features / only the target's direct causes (parents) /
                  the target's Markov blanket (parents + children + co-parents),
                  via restrict_to_parents / restrict_to_markov_blanket

giving six settings per backbone:

    uncon_all   vanilla,     all features
    con_all     constrained, all features
    uncon_par   vanilla,     parents only        (interaction-graph baseline)
    con_par     constrained, parents only        (baseline + constraints)
    uncon_mb    vanilla,     Markov blanket
    con_mb      constrained, Markov blanket

giving a seventh setting, uncon_ME, which isolates the effect of swapping in
the Moreau-envelope-smoothed optimizer alone (same "all features", no ALM
duals/constraints) -- a control for how much of con_all's effect (if any) is
just "different optimizer" rather than "causal constraint":

    uncon_ME    vanilla + Moreau-envelope optimizer, all features

Backbone -> (unconstrained config, constrained config, Moreau-only config):

    mlp           -> discrete             / discrete_constrained             / discrete_me
    deep_mlp      -> discrete_deep        / discrete_deep_constrained        / discrete_deep_me
    onehot_mlp    -> discrete_onehot      / discrete_onehot_constrained      / discrete_onehot_me
    embedding_mlp -> discrete_embedding   / discrete_embedding_constrained   / discrete_embedding_me
    transformer   -> discrete_transformer / discrete_transformer_constrained / discrete_transformer_me

Every (backbone, setting, target) is evaluated on N_SEEDS independently
generated er_graph datasets. Results are written to:

    <out>.csv   — raw long-form rows (seed, target, network, setting, errors)
    <out>.xlsx  — sheet 'summary_overall' : avg over seeds AND features, one row
                                            per backbone, the four settings side
                                            by side; best (lowest) per metric bold.
                  sheet 'by_feature_test' : avg test error over seeds, one row per
                                            target, four settings per backbone;
                                            best per backbone bold.
                  sheet 'by_feature_train': same for train error.
                  sheet 'violation_overall' : avg constraint violation over
                                            seeds & features, network x setting;
                                            lowest per network bold.
                  sheet 'violation_by_feature': avg violation per target.
                  sheet 'raw'             : every individual run.

Constraint violation measures how much each fitted model's predictions break the
causal-independence constraints implied by w_est (see
DiscreteRecommenderPredictor.constraint_violation); it is logged for every
setting, constrained or not.

This is the multi-backbone sibling of test_all_features.py.

Usage:
    python test_all_networks.py
    python test_all_networks.py --n-seeds 3 --networks mlp,transformer
    python test_all_networks.py --seed 7            # only er_graph seed 7
    # fast smoke test:
    python test_all_networks.py --n-seeds 2 --n-epochs 2 --n-runs 2 --targets X0,X1
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
from discrete_estimator import DiscreteRecommenderPredictor

logging.basicConfig(level=logging.WARNING)

CONF_DIR = Path(__file__).parent / "experiments_conf"

# (label, unconstrained solver config, constrained solver config,
#  Moreau-envelope-only solver config)
NETWORKS = [
    ("mlp",           "discrete",             "discrete_constrained",             "discrete_me"),
    ("deep_mlp",      "discrete_deep",        "discrete_deep_constrained",        "discrete_deep_me"),
    ("onehot_mlp",    "discrete_onehot",      "discrete_onehot_constrained",      "discrete_onehot_me"),
    ("embedding_mlp", "discrete_embedding",   "discrete_embedding_constrained",   "discrete_embedding_me"),
    ("transformer",   "discrete_transformer", "discrete_transformer_constrained", "discrete_transformer_me"),
]

# (setting label, which config variant, feature-restriction mode)
#   variant: 'uncon' = vanilla Adam, 'con' = ALM-constrained, 'me' = vanilla
#            but with the Moreau-envelope optimizer (no constraints)
#   mode: 'none' = all features, 'parents' = direct causes only,
#         'markov_blanket' = parents + children + co-parents
SETTINGS = [
    ("uncon_all", "uncon", "none"),
    ("con_all",   "con",   "none"),
    ("uncon_ME",  "me",    "none"),
#    ("uncon_par", "uncon", "parents"),
#    ("con_par",   "con",   "parents"),
    ("uncon_mb",  "uncon", "markov_blanket"),
    ("con_mb",    "con",   "markov_blanket"),
]
SETTING_ORDER = [s[0] for s in SETTINGS]


def build_dataset(problem_cfg, seed):
    """Generate the er_graph dataset + true DAG for a given seed.

    Mirrors run_experiments.py's er_graph branch, but with an explicit seed so
    the caller can sweep multiple independent datasets.
    Returns (prep_data, w_est, column_names).
    """
    if problem_cfg.name != "er_graph":
        raise ValueError(
            f"test_all_networks only supports problem=er_graph, got "
            f"'{problem_cfg.name}'."
        )

    from er_graph import sample_dag

    X, dag = sample_dag(
        n_samples=problem_cfg.n_samples,
        d=problem_cfg.d,
        s0=problem_cfg.s0,
        graph_type=problem_cfg.get("graph_type", "ER"),
        n_values=problem_cfg.get("n_values", 3),
        dominant_prob=problem_cfg.get("dominant_prob", 0.8),
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


def evaluate(prep_data, w_est, row_and_col_names, target, features, solver_cfg, seed):
    """Run one (target, solver) evaluation; return (train_err, test_err)."""
    torch.manual_seed(seed)  # reproducible network init/training per seed
    (_best, train_err, test_err,
     *_rest) = run_feature_selection_scikit(
        prep_data.copy(),
        solver_cfg.model_name,
        solver_cfg.custom_objective,
        target,
        w_est, row_and_col_names,
        solver_cfg.n_runs,
        len(features),
        features,
        solver_cfg=solver_cfg,
    )
    return float(train_err), float(test_err)


def fit_violation(prep_data, w_est, row_and_col_names, target, features,
                  solver_cfg, seed):
    """Constraint violation of one model fitted on the full data.

    The CV pipeline does not expose its fitted estimator, so we fit one extra
    model on the full (X, y) with the same config/features as evaluate() and
    reuse DiscreteRecommenderPredictor.constraint_violation. Returns the total
    violation (NaN on failure). Honours the cfg restriction flags already set by
    the caller.
    """
    try:
        torch.manual_seed(seed)
        X = prep_data[features]
        y = prep_data[target]
        model = DiscreteRecommenderPredictor(
            w_est, target, row_and_col_names, solver_cfg.custom_objective,
            None, solver_cfg)
        model.fit(X, y)
        return float(model.constraint_violation(X)["total"])
    except Exception:
        return float("nan")


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

    Returns (DataFrame, groups) where each group is the four settings of one
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
    parser.add_argument("--out", default="network_comparison",
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
                    cfg.restrict_to_parents = (mode == "parents")
                    cfg.restrict_to_markov_blanket = (mode == "markov_blanket")
                    try:
                        tr, te = evaluate(prep_data, w_est, row_and_col_names,
                                          target, features, cfg, seed)
                    except Exception as exc:  # keep the long sweep alive
                        print(f"  [WARN] seed={seed} {target} {label}/{setting} "
                              f"failed: {exc}")
                        tr, te = float("nan"), float("nan")
                    viol = fit_violation(prep_data, w_est, row_and_col_names,
                                         target, features, cfg, seed)
                    records.append({
                        "seed": seed, "target": target, "network": label,
                        "setting": setting, "variant": variant,
                        "features": mode,
                        "train_error": tr, "test_error": te, "violation": viol,
                    })
                    print(f"  seed={seed} {target:>3s} {label:14s} {setting:9s} "
                          f"train={tr:.4f} test={te:.4f} viol={viol:.4f}")

    raw = pd.DataFrame(records)
    csv_path = Path(f"{args.out}.csv")
    raw.to_csv(csv_path, index=False)

    targets_sorted = sorted(raw["target"].unique(), key=lambda s: int(s[1:]))

    # ---- per-feature summaries (avg over seeds), test and train ----
    g = raw.groupby(["network", "setting", "target"])[
        ["train_error", "test_error"]].mean()
    by_test, by_test_groups = wide_by_feature(g, net_labels, targets_sorted, "test")
    by_train, by_train_groups = wide_by_feature(g, net_labels, targets_sorted, "train")

    # ---- overall summary: avg over seeds AND features ----
    o = raw.groupby(["network", "setting"])[["train_error", "test_error"]].mean()
    overall = pd.DataFrame(index=net_labels)
    overall.index.name = "network"
    for metric in ["train", "test"]:
        for setting in SETTING_ORDER:
            overall[f"{setting}_{metric}"] = [
                o.loc[(n, setting), f"{metric}_error"] for n in net_labels]
    overall = overall.round(4)
    overall_groups = [[f"{s}_train" for s in SETTING_ORDER],
                      [f"{s}_test" for s in SETTING_ORDER]]

    # ---- constraint-violation summaries (avg over seeds) ----
    gv = raw.groupby(["network", "setting", "target"])["violation"].mean()
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

    ov = raw.groupby(["network", "setting"])["violation"].mean()
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
    print(f"\nWrote {csv_path.resolve()}")
    print(f"Wrote {xlsx_path.resolve()}  (bold = best of the four settings)")


if __name__ == "__main__":
    main()