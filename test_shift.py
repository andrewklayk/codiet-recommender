"""Distribution-shift experiment: do the causal constraints help under shift?

In-distribution the constraints add little, because a cross-entropy model fit on
data sampled from a DAG already (approximately) respects the conditional
independencies the constraints encode. The constraints are meant to pay off when
the *numbers* change but the *structure* does not -- i.e. under a covariate /
mechanism shift that keeps the graph (and its independencies) intact.

Design (chain  A -> B -> C, target = C, features = {A, B}):

    * The target's own mechanism  P(C | B)  is held FIXED across train and test.
      Therefore the Bayes-optimal predictor P(C|B) is INVARIANT, and a model that
      respects  A _||_ C | B  (uses only B, the Markov blanket of C) transfers
      perfectly.
    * The UPSTREAM CPTs  P(A)  and  P(B | A)  are RE-ROLLED for the test set
      (same shape, new random dominant outcomes). This flips the in-sample
      A - C correlation, so a model that leaned on A to predict C breaks at test.

We compare, on a held-out IN-DISTRIBUTION set and on the SHIFTED set:

    uncon_all   vanilla,                      features {A, B}  (free to (ab)use A)
    con_all     constrained,                  features {A, B}  (penalised for A-C dep.)
    uncon_ME    vanilla + Moreau-env. optim,  features {A, B}  (optimizer-only control)
    dsep_uncon  vanilla,                      Markov blanket {B} only (A dropped -> invariant)
    dsep_con    constrained,                  Markov blanket {B} only

uncon_ME isolates how much of con_all's behaviour (if any) comes from the
Moreau-envelope-wrapped optimizer alone, as opposed to the ALM causal
constraint itself: same features as uncon_all/con_all, no constraints.

Expectation: in-distribution all are similar; under shift dsep_* stay low while
uncon_all degrades most, with con_all in between if the constraint is doing its
job. Errors are classification error normalised by each eval set's own
majority-class baseline (1.0 = no better than predicting that set's majority).

Usage:
    python test_shift.py
    python test_shift.py --n-seeds 30 --n-train 150 --networks mlp,deep_mlp
    # all five backbones used by test_all_networks.py / test_constraints.py:
    #   mlp, deep_mlp, onehot_mlp, embedding_mlp, transformer (now the default)
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openpyxl.styles import Font

from er_graph import _generate_cpt, _sample_categorical
from discrete_estimator import DiscreteRecommenderPredictor

logging.basicConfig(level=logging.WARNING)

NODES = ["A", "B", "C"]
W = np.zeros((3, 3))
W[0, 1] = 1   # A -> B
W[1, 2] = 1   # B -> C  (target C; Markov blanket of C is {B})

# (label, use_alm, restrict_to_markov_blanket, use_moreau)
METHODS = [
    ("uncon_all",  False, False, False),
    ("con_all",    True,  False, False),
    ("uncon_ME",   False, False, True),
    ("dsep_uncon", False, True,  False),
    ("dsep_con",   True,  True,  False),
]
METHOD_ORDER = [m[0] for m in METHODS]


def _sample(n, cpt_A, cpt_B, cpt_C, n_values):
    """Ancestral sample of the chain A->B->C from the given CPTs."""
    A = _sample_categorical(np.tile(cpt_A[0], (n, 1)))
    B = _sample_categorical(cpt_B[A])
    C = _sample_categorical(cpt_C[B])
    return pd.DataFrame({"A": A, "B": B, "C": C})


def make_datasets(seed, n_train, n_eval, n_values, dominant_prob):
    """Train / in-distribution-test / shifted-test sharing P(C|B).

    Upstream CPTs P(A), P(B|A) are re-rolled for the shifted set; P(C|B) is kept.
    """
    np.random.seed(seed)
    cpt_A = _generate_cpt(0, n_values, dominant_prob)   # P(A)        (1, k)
    cpt_B = _generate_cpt(1, n_values, dominant_prob)   # P(B|A)      (k, k)
    cpt_C = _generate_cpt(1, n_values, dominant_prob)   # P(C|B) FIXED (k, k)
    train = _sample(n_train, cpt_A, cpt_B, cpt_C, n_values)
    indist = _sample(n_eval, cpt_A, cpt_B, cpt_C, n_values)
    # --- shift: new upstream, same target mechanism ---
    cpt_A2 = _generate_cpt(0, n_values, dominant_prob)
    cpt_B2 = _generate_cpt(1, n_values, dominant_prob)
    shift = _sample(n_eval, cpt_A2, cpt_B2, cpt_C, n_values)
    return train, indist, shift


def norm_error(model, X_eval, y_eval):
    """Classification error normalised by the eval set's majority-class error."""
    acc = (np.asarray(model.predict(X_eval)) == y_eval.values).mean()
    maj = y_eval.value_counts(normalize=True).max()
    return (1.0 - acc) / max(1e-9, 1.0 - maj)


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
    p.add_argument("--out", default="shift_comparison")
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--n-values", type=int, default=3)
    p.add_argument("--dominant-prob", type=float, default=0.8)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--networks",
                   default="mlp,deep_mlp,onehot_mlp,embedding_mlp,transformer")
    args = p.parse_args()

    net_labels = [s.strip() for s in args.networks.split(",")]
    print(f"Backbones: {net_labels} | seeds: {args.n_seeds} | "
          f"n_train={args.n_train} n_eval={args.n_eval} | "
          f"chain A->B->C, target C, P(C|B) fixed, upstream re-rolled\n")

    records = []
    for seed in range(args.n_seeds):
        train, indist, shift = make_datasets(
            seed, args.n_train, args.n_eval, args.n_values, args.dominant_prob)
        for net in net_labels:
            for method, use_alm, mb, use_moreau in METHODS:
                cfg = {"network": net, "n_epochs": args.n_epochs,
                       "use_alm": use_alm, "restrict_to_markov_blanket": mb,
                       "use_moreau": use_moreau}
                torch.manual_seed(seed)
                m = DiscreteRecommenderPredictor(W, "C", NODES, "none", None, cfg)
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
    rows, groups = [], []
    idx = []
    for net in net_labels:
        block = []
        for method in METHOD_ORDER:
            idx.append((net, method))
            rows.append(g.loc[(net, method)])
        # one bold-group per backbone per metric handled below
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