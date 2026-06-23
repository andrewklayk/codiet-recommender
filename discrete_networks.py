"""Pluggable neural-network backbones for DiscreteRecommenderPredictor.

Every backbone shares one API so the estimator can swap them by name with no
type checks:

    net = build_network(name, n_features, n_classes, n_values, cfg)
    logits = net(x)            # x: (batch, n_features) float tensor of category
                               #    indices; logits: (batch, n_classes)

Add a new backbone by subclassing DiscreteNetwork and decorating it with
@register_network("<name>"); it becomes selectable via cfg.network without any
change to the estimator.

Networks read their own hyper-parameters from `cfg` (any object with a
.get(key, default) method — a dict or an OmegaConf node), so each one documents
and defaults its knobs independently.

Available backbones (cfg.network):
    mlp           — shallow MLP on the raw integer features (default)
    deep_mlp      — deeper MLP with input BatchNorm + dropout
    onehot_mlp    — one-hot encode every feature, then an MLP
    embedding_mlp — a learned embedding per categorical feature, then an MLP
    transformer   — tabular transformer: one token per feature + CLS readout
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NETWORK_REGISTRY = {}


def register_network(name):
    """Class decorator that registers a backbone under `name` for build_network."""
    def _decorator(cls):
        NETWORK_REGISTRY[name] = cls
        return cls
    return _decorator


def build_network(name, n_features, n_classes, n_values, cfg):
    """Instantiate a registered backbone by name.

    Args:
        name       : key into NETWORK_REGISTRY (case-insensitive); None -> 'mlp'.
        n_features : number of input columns.
        n_classes  : number of target classes.
        n_values   : category cardinality used by encoders/embeddings (max
                     category index across features + 1). Backbones that treat
                     inputs as raw floats ignore it.
        cfg        : object exposing .get(key, default) for hyper-parameters.

    Returns:
        nn.Module implementing forward(x) -> logits.
    """
    key = (name or "mlp").lower()
    try:
        cls = NETWORK_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Unknown network '{name}'. Available: {sorted(NETWORK_REGISTRY)}"
        )
    return cls(n_features, n_classes, n_values, cfg)


def _mlp_head(in_dim, hidden_dim, n_layers, n_classes, dropout=0.0):
    """A plain ReLU MLP: in_dim -> [hidden_dim]*n_layers -> n_classes."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, n_classes))
    return nn.Sequential(*layers)


class DiscreteNetwork(nn.Module):
    """Common base for all backbones.

    Subclasses implement __init__(self, n_features, n_classes, n_values, cfg)
    and forward(self, x) where x is a (batch, n_features) float tensor of
    integer-valued category indices and the return is (batch, n_classes) logits.
    """

    def __init__(self, n_features, n_classes, n_values, cfg):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_values = n_values


@register_network("mlp")
class MLPNetwork(DiscreteNetwork):
    """Shallow MLP on the raw integer features (the original backbone).

    cfg knobs: hidden_dim (64), n_layers (2).
    """

    def __init__(self, n_features, n_classes, n_values, cfg):
        super().__init__(n_features, n_classes, n_values, cfg)
        hidden_dim = cfg.get("hidden_dim", 64)
        n_layers = cfg.get("n_layers", 2)
        self.net = _mlp_head(n_features, hidden_dim, n_layers, n_classes)

    def forward(self, x):
        return self.net(x)


@register_network("deep_mlp")
class DeepMLPNetwork(DiscreteNetwork):
    """Deeper MLP with input normalisation and dropout for regularisation.

    cfg knobs: hidden_dim (128), n_layers (4), dropout (0.1).
    """

    def __init__(self, n_features, n_classes, n_values, cfg):
        super().__init__(n_features, n_classes, n_values, cfg)
        hidden_dim = cfg.get("hidden_dim", 128)
        n_layers = cfg.get("n_layers", 4)
        dropout = cfg.get("dropout", 0.1)
        self.input_norm = nn.BatchNorm1d(n_features)
        self.net = _mlp_head(n_features, hidden_dim, n_layers, n_classes, dropout)

    def forward(self, x):
        return self.net(self.input_norm(x))


@register_network("onehot_mlp")
class OneHotMLPNetwork(DiscreteNetwork):
    """One-hot encode every categorical feature, then an MLP.

    cfg knobs: hidden_dim (64), n_layers (2), dropout (0.0).
    """

    def __init__(self, n_features, n_classes, n_values, cfg):
        super().__init__(n_features, n_classes, n_values, cfg)
        hidden_dim = cfg.get("hidden_dim", 64)
        n_layers = cfg.get("n_layers", 2)
        dropout = cfg.get("dropout", 0.0)
        self.net = _mlp_head(n_features * n_values, hidden_dim, n_layers,
                             n_classes, dropout)

    def forward(self, x):
        idx = x.long().clamp(0, self.n_values - 1)
        onehot = F.one_hot(idx, num_classes=self.n_values).float()  # (B, F, V)
        return self.net(onehot.view(x.size(0), -1))


@register_network("embedding_mlp")
class EmbeddingMLPNetwork(DiscreteNetwork):
    """A learned embedding per categorical feature, concatenated, then an MLP.

    cfg knobs: emb_dim (8), hidden_dim (64), n_layers (2), dropout (0.0).
    """

    def __init__(self, n_features, n_classes, n_values, cfg):
        super().__init__(n_features, n_classes, n_values, cfg)
        emb_dim = cfg.get("emb_dim", 8)
        hidden_dim = cfg.get("hidden_dim", 64)
        n_layers = cfg.get("n_layers", 2)
        dropout = cfg.get("dropout", 0.0)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n_values, emb_dim) for _ in range(n_features)]
        )
        self.net = _mlp_head(n_features * emb_dim, hidden_dim, n_layers,
                             n_classes, dropout)

    def forward(self, x):
        idx = x.long().clamp(0, self.n_values - 1)
        embs = [emb(idx[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.net(torch.cat(embs, dim=1))


@register_network("transformer")
class TransformerNetwork(DiscreteNetwork):
    """Tabular transformer: each feature becomes a token, read out via a CLS token.

    A shared value-embedding table maps category indices to d_model vectors; a
    per-column embedding tells the encoder which feature each token came from. A
    learned CLS token is prepended and its final state is classified.

    cfg knobs: d_model (32), n_heads (4), n_layers (2), dim_feedforward (64),
               dropout (0.1).
    """

    def __init__(self, n_features, n_classes, n_values, cfg):
        super().__init__(n_features, n_classes, n_values, cfg)
        d_model = cfg.get("d_model", 32)
        n_heads = cfg.get("n_heads", 4)
        n_layers = cfg.get("n_layers", 2)
        dim_ff = cfg.get("dim_feedforward", 64)
        dropout = cfg.get("dropout", 0.1)

        self.value_emb = nn.Embedding(n_values, d_model)
        self.col_emb = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.col_emb, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        idx = x.long().clamp(0, self.n_values - 1)
        tokens = self.value_emb(idx) + self.col_emb          # (B, F, d_model)
        cls = self.cls.expand(x.size(0), -1, -1)             # (B, 1, d_model)
        seq = torch.cat([cls, tokens], dim=1)                # (B, F+1, d_model)
        encoded = self.encoder(seq)
        return self.head(encoded[:, 0])                      # CLS readout