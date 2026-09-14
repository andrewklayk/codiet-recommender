"""Guards that every training regime is compared under IDENTICAL conditions.

The comparisons these experiments make (constrained vs. unconstrained vs.
Moreau-envelope-only, all-features vs. parents vs. Markov blanket) are only
meaningful if the arms differ in exactly the intended way. Three ways that has
silently broken before, each covered by a test here:

  1. A metaparameter drifting between the three solver yamls of one backbone
     (continuous_*_constrained.yaml carried batch_size 256 while its siblings
     carried 32, so the constrained arm got 8x fewer optimizer steps).
  2. The Moreau-envelope smoothing of the ALM arm diverging from the *_me
     arm's. The ALM optimizer wraps Adam in the SAME MoreauEnvelope, so
     uncon_ME is the control for con_*; if mu/beta differ, that control is
     void.
  3. A script building a cfg by hand and falling back on code defaults that no
     longer match the yaml (test_shift.py trained onehot/embedding backbones
     with dropout 0.0 while the yamls said 0.1), or dropping a load-bearing
     control method from its comparison table (test_constraints.py had no
     plain-Adam-on-all-features arm, which made uncon_ME look like a winner
     when it merely was the only unrestricted-feature method present).

Run from the repo root:  python -m unittest discover -s tests -t .
"""
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOLVER_DIR = ROOT / "experiments_conf" / "solver"

# backbone label -> config basename, per estimator family. The three variants
# of one entry are <base>.yaml / <base>_constrained.yaml / <base>_me.yaml.
FAMILIES = {
    "discrete": ["discrete", "discrete_deep", "discrete_onehot",
                 "discrete_embedding", "discrete_transformer"],
    "continuous": ["continuous", "continuous_deep", "continuous_transformer"],
}

# Keys the three variants of one backbone are ALLOWED to differ on -- i.e. the
# knobs that define what "constrained" / "Moreau-only" / "plain" even mean.
# Everything else (architecture, epochs, batch size, learning rate, CV folds,
# constraint-estimator knobs) must be identical, or the arms are not comparable.
VARIANT_KEYS = {
    "name", "use_alm", "use_moreau", "moreau_mu", "moreau_beta",
    "alm_lr", "alm_penalty", "alm_slack", "alm_momentum",
    "n_constraints", "alm_selection_weight",
}


def _load(base):
    return yaml.safe_load((SOLVER_DIR / f"{base}.yaml").read_text(encoding="utf-8"))


class SolverConfigConditions(unittest.TestCase):

    def test_variants_of_a_backbone_differ_only_in_the_regime_knobs(self):
        for family, bases in FAMILIES.items():
            for base in bases:
                cfgs = {"plain": _load(base),
                        "constrained": _load(base + "_constrained"),
                        "me": _load(base + "_me")}
                keys = set().union(*map(set, cfgs.values())) - VARIANT_KEYS
                for key in sorted(keys):
                    values = {n: c.get(key, "<absent>") for n, c in cfgs.items()}
                    self.assertEqual(
                        len(set(map(repr, values.values()))), 1,
                        f"{family}/{base}: '{key}' differs across variants "
                        f"({values}) -- the arms are not comparable")

    def test_alm_arm_uses_the_same_moreau_smoothing_as_the_me_arm(self):
        # uncon_ME is the control for con_*: causal_estimator_base.fit wraps the
        # ALM optimizer in MoreauEnvelope with cfg's mu/beta, so those values
        # must match the *_me sibling's, and both must be stated explicitly
        # rather than left to a code default.
        for bases in FAMILIES.values():
            for base in bases:
                con, me = _load(base + "_constrained"), _load(base + "_me")
                for key in ("moreau_mu", "moreau_beta"):
                    self.assertIn(key, con, f"{base}_constrained is missing {key}")
                    self.assertIn(key, me, f"{base}_me is missing {key}")
                    self.assertEqual(
                        con[key], me[key],
                        f"{base}: ALM arm's {key} ({con[key]}) != Moreau-only "
                        f"arm's ({me[key]}); uncon_ME is then not a control "
                        f"for con_*")

    def test_backbone_code_defaults_match_the_yaml(self):
        # A script that builds a cfg by hand (test_shift*.py used to) must train
        # the same network as one that loads the yaml.
        import discrete_networks, continuous_networks  # noqa: F401

        class _Probe(dict):
            """Records every cfg.get() default a backbone falls back on."""

            def __init__(self):
                super().__init__()
                self.defaults = {}

            def get(self, key, default=None):
                self.defaults[key] = default
                return default

        # discrete: build_network(name, n_features, n_classes, n_values, cfg)
        # continuous: build_network(name, n_features, cfg)
        cases = [
            (discrete_networks.build_network, (4, 3, 3), "mlp", "discrete"),
            (discrete_networks.build_network, (4, 3, 3), "deep_mlp", "discrete_deep"),
            (discrete_networks.build_network, (4, 3, 3), "onehot_mlp", "discrete_onehot"),
            (discrete_networks.build_network, (4, 3, 3), "embedding_mlp", "discrete_embedding"),
            (discrete_networks.build_network, (4, 3, 3), "transformer", "discrete_transformer"),
            (continuous_networks.build_network, (4,), "mlp", "continuous"),
            (continuous_networks.build_network, (4,), "deep_mlp", "continuous_deep"),
            (continuous_networks.build_network, (4,), "transformer", "continuous_transformer"),
        ]
        for build, shape_args, name, base in cases:
            for suffix in ("", "_constrained", "_me"):
                cfg = _load(base + suffix)
                probe = _Probe()
                build(name, *shape_args, probe)
                for key, default in probe.defaults.items():
                    if key in cfg:
                        self.assertEqual(
                            default, cfg[key],
                            f"{name}: code default {key}={default} != "
                            f"{base}{suffix}.yaml's {cfg[key]}")


class ExperimentMethodTables(unittest.TestCase):
    """Every comparison must carry both of its load-bearing controls."""

    def test_every_experiment_has_uncon_all_and_uncon_ME(self):
        import test_all_networks, test_all_networks_continuous
        import test_constraints, test_constraints_continuous
        import test_shift, test_shift_continuous

        tables = {
            "test_all_networks": test_all_networks.SETTING_ORDER,
            "test_all_networks_continuous": test_all_networks_continuous.SETTING_ORDER,
            "test_constraints": test_constraints.METHOD_ORDER,
            "test_constraints_continuous": test_constraints_continuous.METHOD_ORDER,
            "test_shift": test_shift.METHOD_ORDER,
            "test_shift_continuous": test_shift_continuous.METHOD_ORDER,
        }
        for script, labels in tables.items():
            self.assertIn(
                "uncon_all", labels,
                f"{script}: without a plain-Adam-on-all-features arm, every "
                f"all-features method's apparent gain is just 'all features vs "
                f"a restricted feature set'")
            self.assertIn(
                "uncon_ME", labels,
                f"{script}: without the Moreau-only arm, con_all's effect "
                f"cannot be separated from its optimizer's")

    def test_method_labels_follow_the_shared_naming_scheme(self):
        import test_all_networks, test_all_networks_continuous
        import test_constraints, test_constraints_continuous
        import test_shift, test_shift_continuous

        known = {"uncon_all", "con_all", "uncon_ME",
                 "uncon_parents", "con_parents", "uncon_mb", "con_mb"}
        for mod in (test_all_networks, test_all_networks_continuous,
                    test_constraints, test_constraints_continuous,
                    test_shift, test_shift_continuous):
            labels = getattr(mod, "SETTING_ORDER", None) or mod.METHOD_ORDER
            for label in labels:
                self.assertIn(
                    label, known,
                    f"{mod.__name__}: '{label}' is not one of the shared "
                    f"<regime>_<feature set> labels {sorted(known)}")


if __name__ == "__main__":
    unittest.main()
