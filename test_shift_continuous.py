"""Distribution-shift experiment for the CONTINUOUS estimator: do the causal
constraints help under shift? Continuous twin of test_shift.py -- same design,
same five methods, same backbones-as-siblings idea, just over a linear-
Gaussian SEM instead of discrete CPTs.

In-distribution the constraints add little, because an MSE model fit on data
sampled from a DAG already (approximately) respects the conditional
independencies the constraints encode. The constraints are meant to pay off
when the *numbers* change but the *structure* does not -- i.e. under a
covariate / mechanism shift that keeps the graph (and its independencies)
intact.

Design (chain  A -> B -> C, target = C, features = {A, B}):

    * The target's own mechanism  P(C | B)  (linear coefficient + noise) is
      held FIXED across train and test. Therefore the Bayes-optimal predictor
      E[C|B] is INVARIANT, and a model that respects  A _||_ C | B  (uses only
      B, the Markov blanket of C) transfers perfectly.
    * The UPSTREAM mechanisms  P(A)  and  P(B | A)  (mean/std of A, linear
      coefficient + noise of B|A) are RE-ROLLED for the test set. This flips
      the in-sample A-C correlation, so a model that leaned on A to predict C
      breaks at test.

We compare, on a held-out IN-DISTRIBUTION set and on the SHIFTED set:

    uncon_all   vanilla,                      features {A, B}  (free to (ab)use A)
    con_all     constrained,                  features {A, B}  (penalised for A-C dep.)
    uncon_ME    vanilla + Moreau-env. optim,  features {A, B}  (optimizer-only control)
    uncon_mb    vanilla,                      Markov blanket {B} only (A dropped -> invariant)
    con_mb      constrained,                  Markov blanket {B} only

uncon_ME isolates how much of con_all's behaviour (if any) comes from the
Moreau-envelope-wrapped optimizer alone, as opposed to the ALM causal
constraint itself: same features as uncon_all/con_all, no constraints.

Expectation: in-distribution all are similar; under shift *_mb stay low
while uncon_all degrades most, with con_all in between if the constraint is
doing its job. Errors are MSE normalised by each eval set's own variance (1.0
= no better than predicting that set's mean) -- the continuous analogue of
test_shift.py's majority-class-normalised classification error.

Usage:
    python test_shift_continuous.py
    python test_shift_continuous.py --n-seeds 30 --n-train 150 --networks mlp,deep_mlp
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openpyxl.styles import Font

from continuous_estimator import ContinuousRecommenderPredictor
from test_all_networks_continuous import NETWORKS, load_solver

logging.basicConfig(level=logging.WARNING)

NODES = ["A", "B", "C"]
W = np.zeros((3, 3))
W[0, 1] = 1   # A -> B
W[1, 2] = 1   # B -> C  (target C; Markov blanket of C is {B})

# (label, solver-config variant, restrict_to_markov_blanket)
#   variant: 'uncon' = vanilla Adam, 'con' = ALM-constrained, 'me' = vanilla but
#            with the Moreau-envelope optimizer (no constraints)
# Labels follow the scheme shared by every experiment script here,
# <training regime>_<feature set>.  uncon_parents is omitted on purpose: the
# target C's parents ({B}) ARE its Markov blanket in this chain, so that method
# would be bit-identical to uncon_mb.
METHODS = [
    ("uncon_all",  "uncon", False),
    ("con_all",    "con",   False),
    ("uncon_ME",   "me",    False),
    ("uncon_mb",   "uncon", True),
    ("con_mb",     "con",   True),
]
METHOD_ORDER = [m[0] for m in METHODS]


def _sample(rng, n, mean_A, std_A, a_B, noise_B, a_C, noise_C):
    """Ancestral sample of the linear-Gaussian chain A->B->C."""
    A = rng.normal(mean_A, std_A, n)
    B = a_B * A + rng.normal(0, noise_B, n)
    C = a_C * B + rng.normal(0, noise_C, n)
    return pd.DataFrame({"A": A, "B": B, "C": C})


def make_datasets(seed, n_train, n_eval):
    """Train / in-distribution-test / shifted-test sharing P(C|B).

    Upstream mechanisms P(A), P(B|A) are re-rolled for the shifted set;
    P(C|B) (a_C, noise_C) is kept.
    """
    rng = np.random.default_rng(seed)
    mean_A, std_A = rng.uniform(-1, 1), rng.uniform(0.5, 1.5)
    a_B, noise_B = rng.uniform(0.5, 1.2), rng.uniform(0.3, 0.8)
    a_C, noise_C = rng.uniform(0.5, 1.2), rng.uniform(0.3, 0.8)   # P(C|B) FIXED
    train = _sample(rng, n_train, mean_A, std_A, a_B, noise_B, a_C, noise_C)
    indist = _sample(rng, n_eval, mean_A, std_A, a_B, noise_B, a_C, noise_C)
    # --- shift: new upstream, same target mechanism ---
    mean_A2, std_A2 = rng.uniform(-2, 2), rng.uniform(0.5, 2.0)
    a_B2, noise_B2 = rng.uniform(0.5, 1.5), rng.uniform(0.3, 1.0)
    shift = _sample(rng, n_eval, mean_A2, std_A2, a_B2, noise_B2, a_C, noise_C)
    return train, indist, shift


def norm_error(model, X_eval, y_eval):
    """MSE normalised by the eval set's own variance (1.0 = predicting the mean)."""
    pred = np.asarray(model.predict(X_eval))
    mse = np.mean((pred - y_eval.values) ** 2)
    return mse / max(1e-9, np.var(y_eval.values))


def bold_min_cells(ws, df, groups):
    """Bold the smallest cell within each column-group, per data row."""
    pos = {c: i for i, c in enumerate(df.columns)}
    for r, (_, row) in enumerate(df.iterrows()):
        for group in groups:
            vals = [(c, row[c]) for c in group if pd.notna(row[c])]
            if not vals:
                continue
            best = min(vals, key=lambda t: t[1])[0]
            ws.cell(row=r + 2, column=pos[best] + 2).font = Font(bold=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="shift_comparison_continuous")
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--n-epochs", type=int, default=None,
                   help="override n_epochs in every solver "
                        "(default: whatever the solver yaml says)")
    p.add_argument("--networks", default=None,
                   help="comma-separated subset of backbone labels "
                        f"(default all: {','.join(n[0] for n in NETWORKS)})")
    args = p.parse_args()

    nets = NETWORKS
    if args.networks:
        wanted = {s.strip() for s in args.networks.split(",")}
        nets = [n for n in NETWORKS if n[0] in wanted]
        if not nets:
            raise SystemExit(f"No backbones match {wanted}")
    net_labels = [n[0] for n in nets]
    # one solver cfg per (backbone, variant), loaded from experiments_conf --
    # identical conditions to test_all_networks*.py / test_constraints*.py
    solvers = {}
    for label, uncon, con, me in nets:
        solvers[(label, "uncon")] = load_solver(uncon, args.n_epochs)
        solvers[(label, "con")] = load_solver(con, args.n_epochs)
        solvers[(label, "me")] = load_solver(me, args.n_epochs)
    print(f"Backbones: {net_labels} | seeds: {args.n_seeds} | "
          f"n_train={args.n_train} n_eval={args.n_eval} | "
          f"chain A->B->C, target C, P(C|B) fixed, upstream re-rolled\n")

    records = []
    for seed in range(args.n_seeds):
        train, indist, shift = make_datasets(seed, args.n_train, args.n_eval)
        for net in net_labels:
            for method, variant, mb in METHODS:
                # the SAME solver yaml the other experiments load, so a
                # hand-built cfg can never drift from experiments_conf again
                cfg = solvers[(net, variant)]
                cfg.restrict_to_parents = False
                cfg.restrict_to_markov_blanket = mb
                torch.manual_seed(seed)
                m = ContinuousRecommenderPredictor(W, "C", NODES, "none", None, cfg)
                m.fit(train[["A", "B"]], train["C"])
                e_in = norm_error(m, indist[["A", "B"]], indist["C"])
                e_sh = norm_error(m, shift[["A", "B"]], shift["C"])
                records.append({"seed": seed, "network": net, "method": method,
                                "in_dist": e_in, "shift": e_sh,
                                "degradation": e_sh - e_in})
                print(f"  seed={seed:2d} {net:9s} {method:10s} "
                      f"in_dist={e_in:.4f} shift={e_sh:.4f} deg={e_sh-e_in:+.4f}")

    raw = pd.DataFrame(records)
    raw.to_csv(Path(f"{args.out}.csv"), index=False)

    # summary: mean over seeds, rows = network x method, cols = metrics
    g = raw.groupby(["network", "method"])[["in_dist", "shift", "degradation"]].mean()
    rows = []
    idx = []
    for net in net_labels:
        for method in METHOD_ORDER:
            idx.append((net, method))
            rows.append(g.loc[(net, method)])
    summary = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        idx, names=["network", "method"])).round(4)

    # also a wide per-backbone view for bold-min on the 'shift' column
    wide = pd.DataFrame(index=METHOD_ORDER)
    wide.index.name = "method"
    shift_groups = []
    for net in net_labels:
        for metric in ["in_dist", "shift"]:
            col = f"{net}_{metric}"
            wide[col] = [g.loc[(net, m), metric] for m in METHOD_ORDER]
        shift_groups.append([f"{net}_in_dist"])
        shift_groups.append([f"{net}_shift"])
    wide = wide.round(4)

    xlsx = Path(f"{args.out}.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary")
        wide.to_excel(writer, sheet_name="by_backbone")
        raw.to_excel(writer, sheet_name="raw", index=False)
        bold_min_cells(writer.sheets["by_backbone"], wide, shift_groups)

    print("\n==== Mean over seeds (lower is better) ====")
    print(summary.to_string())
    print(f"\nWrote {Path(f'{args.out}.csv').resolve()}")
    print(f"Wrote {xlsx.resolve()}  (bold = best method per backbone/metric)")


if __name__ == "__main__":
    main()
