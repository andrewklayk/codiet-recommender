"""Pluggable neural-network backbones for ContinuousRecommenderPredictor.

Same registry pattern as discrete_networks.py: swap backbones by name via
cfg.network, with no change to the estimator.

    net = build_network(name, n_features, cfg)
    y_hat = net(x)     # x: (batch, n_features) float tensor of standardized
                       #    continuous features
                       # y_hat: (batch,) -- NOT (batch, 1). MSELoss silently
                       #    broadcasts (batch, 1) against (batch,) into a
                       #    nonsense loss instead of raising, so every
                       #    backbone below ends forward() with squeeze(-1)
                       #    (see nn_lagrangian.MLPRegressor / recommender_nn's
                       #    MLPRegressor for the existing precedent of this
                       #    exact fix in this repo).

Add a new backbone by subclassing ContinuousNetwork and decorating it with
@register_network("<name>"); it becomes selectable via cfg.network without any
change to the estimator.

Available backbones (cfg.network):
    mlp           — shallow MLP on the standardized features (default)
    deep_mlp      — deeper MLP with input BatchNorm + dropout
    transformer   — FT-Transformer style: a linear per-column tokenizer (each
                    scalar feature gets its own learned affine map into
                    d_model space) + CLS readout. No embedding lookup /
                    one-hot -- inputs are continuous, not categorical.
"""
import torch
import torch.nn as nn

NETWORK_REGISTRY = {}


def register_network(name):
    """Class decorator that registers a backbone under `name` for build_network."""
    def _decorator(cls):
        NETWORK_REGISTRY[name] = cls
        return cls
    return _decorator


def build_network(name, n_features, cfg):
    """Instantiate a registered backbone by name.

    Args:
        name       : key into NETWORK_REGISTRY (case-insensitive); None -> 'mlp'.
        n_features : number of input columns.
        cfg        : object exposing .get(key, default) for hyper-parameters.

    Returns:
        nn.Module implementing forward(x) -> y_hat, shape (batch,).
    """
    key = (name or "mlp").lower()
    try:
        cls = NETWORK_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Unknown network '{name}'. Available: {sorted(NETWORK_REGISTRY)}"
        )
    return cls(n_features, cfg)


def _mlp_head(in_dim, hidden_dim, n_layers, dropout=0.0):
    """A plain ReLU MLP: in_dim -> [hidden_dim]*n_layers -> 1 scalar output."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)


class ContinuousNetwork(nn.Module):
    """Common base for all backbones.

    Subclasses implement __init__(self, n_features, cfg) and forward(self, x)
    where x is a (batch, n_features) float tensor of standardized continuous
    features and the return is (batch,) predictions -- NOT (batch, 1); every
    subclass below ends forward() with squeeze(-1) (see module docstring).
    """

    def __init__(self, n_features, cfg):
        super().__init__()
        self.n_features = n_features


@register_network("mlp")
class MLPNetwork(ContinuousNetwork):
    """Shallow MLP on the standardized features.

    cfg knobs: hidden_dim (64), n_layers (2).
    """

    def __init__(self, n_features, cfg):
        super().__init__(n_features, cfg)
        hidden_dim = cfg.get("hidden_dim", 64)
        n_layers = cfg.get("n_layers", 2)
        self.net = _mlp_head(n_features, hidden_dim, n_layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@register_network("deep_mlp")
class DeepMLPNetwork(ContinuousNetwork):
    """Deeper MLP with input normalisation and dropout for regularisation.

    cfg knobs: hidden_dim (128), n_layers (4), dropout (0.1).
    """

    def __init__(self, n_features, cfg):
        super().__init__(n_features, cfg)
        hidden_dim = cfg.get("hidden_dim", 128)
        n_layers = cfg.get("n_layers", 4)
        dropout = cfg.get("dropout", 0.1)
        self.input_norm = nn.BatchNorm1d(n_features)
        self.net = _mlp_head(n_features, hidden_dim, n_layers, dropout)

    def forward(self, x):
        return self.net(self.input_norm(x)).squeeze(-1)


@register_network("transformer")
class TransformerNetwork(ContinuousNetwork):
    """FT-Transformer style: a linear tokenizer per feature + CLS readout.

    Each scalar feature x_i becomes its own token via a learned affine map
    x_i * w_i + b_i into d_model space -- the continuous analogue of the
    discrete transformer's per-value embedding lookup, since there is no
    fixed vocabulary to embed. A learned CLS token is prepended and its final
    state is read out to a scalar.

    cfg knobs: d_model (32), n_heads (4), n_layers (2), dim_feedforward (64),
               dropout (0.1).
    """

    def __init__(self, n_features, cfg):
        super().__init__(n_features, cfg)
        d_model = cfg.get("d_model", 32)
        n_heads = cfg.get("n_heads", 4)
        n_layers = cfg.get("n_layers", 2)
        dim_ff = cfg.get("dim_feedforward", 64)
        dropout = cfg.get("dropout", 0.1)

        # per-feature linear tokenizer: x_i * tok_weight[i] + tok_bias[i]
        self.tok_weight = nn.Parameter(torch.zeros(n_features, d_model))
        self.tok_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.tok_weight, std=0.02)
        nn.init.normal_(self.tok_bias, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, F) -> tokens: (B, F, d_model), one affine map per feature
        tokens = x.unsqueeze(-1) * self.tok_weight + self.tok_bias
        cls = self.cls.expand(x.size(0), -1, -1)             # (B, 1, d_model)
        seq = torch.cat([cls, tokens], dim=1)                 # (B, F+1, d_model)
        encoded = self.encoder(seq)
        return self.head(encoded[:, 0]).squeeze(-1)            # CLS readout
