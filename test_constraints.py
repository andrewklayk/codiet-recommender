"""Evaluate the causal constraints in isolation, one structure at a time.

Unlike test_all_networks.py (random ER graphs), this experiment uses tiny,
hand-built graphs in which *exactly one* causal structure is active:

    chain     A -> B -> C
    fork      A <- B -> C
    collider  A -> B <- C

For each structure we take every node in turn as the prediction target (so for
the chain the target is first A, then B, then C) and compare four methods, each
run for all five network backbones:

    baseline   unconstrained, features = direct PARENTS of the target only
               (the naive "from A->B->T keep only B" baseline)
    constr_all constrained (ALM),  features = ALL other variables
    uncon_ME   unconstrained + Moreau-envelope optimizer, features = ALL other
               variables (isolates the optimizer's own effect from the causal
               constraint's effect -- a control for constr_all)
    dsep_uncon unconstrained,      features = the target's MARKOV BLANKET
               (the vertices with predictive power according to d-separation:
                parents + children + co-parents)
    dsep_con   constrained (ALM),  features = the target's MARKOV BLANKET

The point: the naive baseline fixes the feature set from the parents alone and
fails whenever the target has no parents (e.g. the root of a chain) even though
descendants are predictive; the constrained estimator on the full feature set,
and the d-separation feature sets, should recover the predictive signal.
uncon_ME checks that any improvement from constr_all is not simply an artifact
of training with the Moreau-envelope-wrapped optimizer.

The estimator is NOT modified: the parents-only restriction reuses the existing
`restrict_to_parents` flag, and the Markov-blanket restriction is applied at the
data level via the `full_feats` argument of run_feature_selection_scikit.

Output (errors are normalized by the majority-class baseline; lower is better):
    <out>.csv   raw long-form rows (seed, structure, target, network, method)
    <out>.xlsx  by_method_test  : avg test err over seeds/structures/targets,
                                  rows=network, cols=method; best method/row bold
                by_method_train : same for train error
                by_setting_test : rows=(structure,target), cols=network x method;
                                  best method per network bold
                by_setting_train: same for train error
                violation_by_method : avg constraint violation, rows=network,
                                  cols=method; lowest per network bold
                violation_by_setting: avg violation per structure/target
                raw             : every individual run

Constraint violation (DiscreteRecommenderPredictor.constraint_violation)
measures how much a fitted model's predictions break the causal-independence
constraints implied by w_est; logged for every method, constrained or not.

Usage:
    python test_constraints.py
    python test_constraints.py --n-seeds 3 --networks mlp,transformer
    # fast smoke test:
    python test_constraints.py --n-seeds 1 --n-epochs 3 --n-runs 2 --networks mlp
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from openpyxl.styles import Font

from er_graph import _generate_cpt, _topological_order, _sample_categorical
from test_all_networks import (NETWORKS, load_solver, evaluate, bold_min_cells,
                               fit_violation)

logging.basicConfig(level=logging.WARNING)

NODES = ["A", "B", "C"]

# directed edges (src -> dst) defining one active structure each
STRUCTURES = {
    "chain":    [("A", "B"), ("B", "C")],   # A -> B -> C
    "fork":     [("B", "A"), ("B", "C")],   # A <- B -> C
    "collider": [("A", "B"), ("C", "B")],   # A -> B <- C
}

# (label, config variant, restrict_to_parents, feature set)
#   variant: 'uncon' = vanilla Adam, 'con' = ALM-constrained, 'me' = vanilla
#            but with the Moreau-envelope optimizer (no constraints)
#   feature set: 'all' = every other variable, 'mb' = Markov blanket
METHODS = [
    ("baseline",   "uncon", True,  "all"),   # parents-only, naive baseline
    ("constr_all", "con",   False, "all"),   # constraints on all features
    ("uncon_ME",   "me",    False, "all"),   # optimizer-only control, all features
    ("dsep_uncon", "uncon", False, "mb"),    # Markov-blanket, no constraints
    ("dsep_con",   "con",   False, "mb"),    # Markov-blanket, constraints
]
METHOD_ORDER = [m[0] for m in METHODS]


def build_structure_dataset(edges, n_samples, n_values, dominant_prob, seed):
    """Sample a dataset from a fixed DAG (given by `edges`) over NODES.

    Reuses er_graph's CPT generation and ancestral sampling so the data-
    generating process matches the random-graph experiments exactly; only the
    adjacency is fixed instead of random.
    Returns (DataFrame over NODES, adjacency matrix B with B[i,j]=1 => i->j).
    """
    np.random.seed(seed)
    d = len(NODES)
    idx = {n: i for i, n in enumerate(NODES)}
    B = np.zeros((d, d), dtype=int)
    for src, dst in edges:
        B[idx[src], idx[dst]] = 1

    parents = [sorted(np.where(B[:, j] == 1)[0].tolist()) for j in range(d)]
    cpts = [_generate_cpt(len(pa), n_values, dominant_prob) for pa in parents]

    X = np.zeros((n_samples, d), dtype=int)
    for j in _topological_order(B):
        pa = parents[j]
        if not pa:
            probs = np.tile(cpts[j][0], (n_samples, 1))
        else:
            config_idx = np.zeros(n_samples, dtype=int)
            for k, p in enumerate(pa):
                config_idx += X[:, p] * (n_values ** (len(pa) - 1 - k))
            probs = cpts[j][config_idx]
        X[:, j] = _sample_categorical(probs)

    return pd.DataFrame(X, columns=NODES), B


def markov_blanket(B, target):
    """Markov-blanket variable names of `target`: parents, children, co-parents.

    These are exactly the vertices that carry predictive information about the
    target under d-separation (everything else is d-separated from the target
    given this set).
    """
    t = NODES.index(target)
    parents = {i for i in range(len(NODES)) if B[i, t]}
    children = {j for j in range(len(NODES)) if B[t, j]}
    coparents = {i for c in children for i in range(len(NODES)) if B[i, c]}
    mb = (parents | children | coparents) - {t}
    return [NODES[i] for i in sorted(mb)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="constraint_comparison",
                        help="output basename for .csv and .xlsx")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="number of dataset seeds, runs 0..n-1 (default 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="run only this single seed (overrides --n-seeds)")
    parser.add_argument("--networks", default=None,
                        help="comma-separated subset of backbone labels "
                             f"(default all: {','.join(n[0] for n in NETWORKS)})")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="samples per dataset (default 1000)")
    parser.add_argument("--n-values", type=int, default=3,
                        help="categories per variable (default 3)")
    parser.add_argument("--dominant-prob", type=float, default=0.8,
                        help="CPT dominant-outcome probability (default 0.8)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="override n_epochs in every solver (for quick runs)")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="override n_runs (CV folds) in every solver")
    args = parser.parse_args()

    nets = NETWORKS
    if args.networks:
        wanted = {s.strip() for s in args.networks.split(",")}
        nets = [n for n in NETWORKS if n[0] in wanted]
        if not nets:
            raise SystemExit(f"No backbones match {wanted}")
    net_labels = [n[0] for n in nets]

    # pre-load each backbone's unconstrained / constrained / Moreau-only config
    solvers = {}
    for label, uncon, con, me in nets:
        solvers[(label, "uncon")] = load_solver(uncon, args.n_epochs, args.n_runs)
        solvers[(label, "con")] = load_solver(con, args.n_epochs, args.n_runs)
        solvers[(label, "me")] = load_solver(me, args.n_epochs, args.n_runs)

    seeds = [args.seed] if args.seed is not None else list(range(args.n_seeds))
    n_runs_eff = solvers[(net_labels[0], "uncon")].n_runs
    print(f"Backbones: {net_labels}")
    print(f"Structures: {list(STRUCTURES)} | targets: {NODES}")
    print(f"Seeds: {seeds} | {n_runs_eff}-fold CV | methods: {METHOD_ORDER}\n")

    records = []
    for seed in seeds:
        for struct, edges in STRUCTURES.items():
            data, B = build_structure_dataset(
                edges, args.n_samples, args.n_values, args.dominant_prob, seed)
            row_and_col_names = list(data.columns)
            for target in NODES:
                all_feats = [c for c in NODES if c != target]
                mb_feats = markov_blanket(B, target)
                for label in net_labels:
                    for method, variant, restrict, fs in METHODS:
                        cfg = solvers[(label, variant)]
                        cfg.restrict_to_parents = restrict
                        if fs == "mb":
                            # Reduced subproblem over the Markov blanket + target.
                            # The estimator assumes X holds *all* non-target
                            # columns of row_and_col_names (column identity is
                            # lost once the CV pipeline converts X to numpy), so
                            # we hand it a graph restricted to exactly the kept
                            # variables instead of reducing via full_feats.
                            keep = [n for n in NODES
                                    if n in set(mb_feats) | {target}]
                            kidx = [NODES.index(n) for n in keep]
                            p_data, p_B, p_names = data[keep], B[np.ix_(kidx, kidx)], keep
                            features = mb_feats
                        else:
                            p_data, p_B, p_names = data, B, row_and_col_names
                            features = all_feats
                        if not features:
                            # no usable features (e.g. empty Markov blanket):
                            # nothing to learn from -> majority-class baseline
                            tr = te = 1.0
                            viol = 0.0
                        else:
                            try:
                                tr, te = evaluate(
                                    p_data, p_B, p_names, target,
                                    features, cfg, seed)
                            except Exception as exc:
                                print(f"  [WARN] seed={seed} {struct}/{target} "
                                      f"{label}/{method} failed: {exc}")
                                tr = te = float("nan")
                            viol = fit_violation(p_data, p_B, p_names, target,
                                                 features, cfg, seed)
                        records.append({
                            "seed": seed, "structure": struct, "target": target,
                            "network": label, "method": method,
                            "features": fs, "train_error": tr, "test_error": te,
                            "violation": viol,
                        })
                        print(f"  seed={seed} {struct:8s} T={target} "
                              f"{label:14s} {method:10s} "
                              f"train={tr:.4f} test={te:.4f} viol={viol:.4f}")

    raw = pd.DataFrame(records)
    csv_path = Path(f"{args.out}.csv")
    raw.to_csv(csv_path, index=False)

    # ---- headline: avg over seeds/structures/targets, rows=network, cols=method
    def by_method(val):
        g = raw.groupby(["network", "method"])[val].mean()
        df = pd.DataFrame(index=net_labels)
        df.index.name = "network"
        for m in METHOD_ORDER:
            df[m] = [g.get((n, m), np.nan) for n in net_labels]
        # bold the best (lowest) method per network row
        return df.round(4), [METHOD_ORDER]

    # ---- detail: rows=(structure,target), cols=network x method
    # single-level index ("structure/target") so bold_min_cells column maths
    # (which assume exactly one index column) stay aligned.
    settings = [(s, t) for s in STRUCTURES for t in NODES]

    def by_setting(val):
        g = raw.groupby(["structure", "target", "network", "method"])[val].mean()
        df = pd.DataFrame(index=[f"{s}/{t}" for s, t in settings])
        df.index.name = "structure/target"
        groups = []
        for net in net_labels:
            cols = []
            for m in METHOD_ORDER:
                col = f"{net}_{m}"
                df[col] = [g.get((s, t, net, m), np.nan) for s, t in settings]
                cols.append(col)
            groups.append(cols)   # bold best method within each network block
        return df.round(4), groups

    bm_test, bm_test_g = by_method("test_error")
    bm_train, bm_train_g = by_method("train_error")
    bm_viol, bm_viol_g = by_method("violation")
    bs_test, bs_test_g = by_setting("test_error")
    bs_train, bs_train_g = by_setting("train_error")
    bs_viol, bs_viol_g = by_setting("violation")

    xlsx_path = Path(f"{args.out}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        bm_test.to_excel(writer, sheet_name="by_method_test")
        bm_train.to_excel(writer, sheet_name="by_method_train")
        bs_test.to_excel(writer, sheet_name="by_setting_test")
        bs_train.to_excel(writer, sheet_name="by_setting_train")
        bm_viol.to_excel(writer, sheet_name="violation_by_method")
        bs_viol.to_excel(writer, sheet_name="violation_by_setting")
        raw.to_excel(writer, sheet_name="raw", index=False)
        bold_min_cells(writer.sheets["by_method_test"], bm_test, bm_test_g)
        bold_min_cells(writer.sheets["by_method_train"], bm_train, bm_train_g)
        bold_min_cells(writer.sheets["by_setting_test"], bs_test, bs_test_g)
        bold_min_cells(writer.sheets["by_setting_train"], bs_train, bs_train_g)
        bold_min_cells(writer.sheets["violation_by_method"], bm_viol, bm_viol_g)
        bold_min_cells(writer.sheets["violation_by_setting"], bs_viol, bs_viol_g)

    print("\n==== Test error by method (avg over seeds/structures/targets) ====")
    print(bm_test.to_string())
    print(f"\nWrote {csv_path.resolve()}")
    print(f"Wrote {xlsx_path.resolve()}  (bold = best method per network)")


if __name__ == "__main__":
    main()