"""View encoders: one trainable network per view.

Weights are NOT shared across views by default (share_weights=False). Shared
weights would force a common representational geometry across views, which
manufactures exactly the cross-view dependence this project exists to
measure -- setting share_weights=True invalidates all downstream ENIV
results and must never be used outside of an explicit, clearly-labeled
ablation.

`MultiViewEncoder.forward(views, view_mask) -> [B, V, feature_dim]` is the
stable call signature for the rest of the pipeline. New encoder backbones
(CNN, Transformer, ...) are added via `register_encoder` and selected per
view through `EncoderConfig.encoder_type`; MultiViewEncoder itself never
needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn


@dataclass
class EncoderConfig:
    input_dim: int
    hidden_dims: list = field(default_factory=lambda: [64, 64])
    feature_dim: int = 32
    dropout: float = 0.0
    encoder_type: str = "mlp"


class MLPEncoder(nn.Module):
    """Feed-forward encoder: input_dim -> hidden_dims -> feature_dim."""

    def __init__(self, input_dim: int, hidden_dims: list, feature_dim: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        dims = [input_dim] + list(hidden_dims)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], feature_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_ENCODER_REGISTRY: dict[str, Callable[[EncoderConfig], nn.Module]] = {
    "mlp": lambda cfg: MLPEncoder(cfg.input_dim, cfg.hidden_dims, cfg.feature_dim, cfg.dropout),
}


def register_encoder(name: str, builder: Callable[[EncoderConfig], nn.Module]) -> None:
    """Register a new encoder backbone (e.g. "cnn", "transformer").

    Once registered, any view can select it via `EncoderConfig(encoder_type=name, ...)`
    with no change to MultiViewEncoder's forward call signature.
    """
    _ENCODER_REGISTRY[name] = builder


def build_encoder(cfg: EncoderConfig) -> nn.Module:
    try:
        builder = _ENCODER_REGISTRY[cfg.encoder_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown encoder_type {cfg.encoder_type!r}; registered types: {list(_ENCODER_REGISTRY)}"
        ) from exc
    return builder(cfg)


class MultiViewEncoder(nn.Module):
    """One encoder per view, applied under view_mask.

    Masked (view_mask == False) entries never enter the per-view encoder's
    forward computation: their output is a constant zero (from torch.zeros,
    not a zeroed encoder output), so they carry no gradient and are excluded
    from all downstream statistics, per the contract's view_mask convention.

    Each view's config may declare a different `input_dim` (<= the tensor's
    last dimension D); the extra trailing features of `views[..., v, :]` are
    simply unused for that view. This lets a single [B, V, D] tensor host
    views of heterogeneous native dimensionality without changing the
    contract's tensor shape.
    """

    def __init__(self, view_configs: list, share_weights: bool = False):
        super().__init__()
        if len(view_configs) == 0:
            raise ValueError("view_configs must be non-empty")

        feature_dims = {cfg.feature_dim for cfg in view_configs}
        if len(feature_dims) != 1:
            raise ValueError(f"All view encoders must share feature_dim, got {feature_dims}")
        self.feature_dim = feature_dims.pop()
        self.n_views = len(view_configs)
        self.share_weights = share_weights
        self.input_dims = [cfg.input_dim for cfg in view_configs]

        if share_weights:
            # DANGER: forces identical representational geometry across
            # views, manufacturing cross-view dependence. Never use this
            # outside an explicit ablation -- see module docstring.
            shared = build_encoder(view_configs[0])
            self.encoders = nn.ModuleList([shared for _ in view_configs])
        else:
            self.encoders = nn.ModuleList([build_encoder(cfg) for cfg in view_configs])

    def forward(self, views: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        """
        views:      [B, V, D]
        view_mask:  [B, V] boolean, True = view present
        returns:    [B, V, feature_dim]
        """
        batch_size = views.shape[0]
        if view_mask.shape != (batch_size, self.n_views):
            raise ValueError(
                f"view_mask shape {tuple(view_mask.shape)} != ({batch_size}, {self.n_views})"
            )

        outputs = []
        for v in range(self.n_views):
            mask_v = view_mask[:, v]
            out_v = views.new_zeros(batch_size, self.feature_dim)
            if mask_v.any():
                view_input = views[mask_v, v, : self.input_dims[v]]
                out_v[mask_v] = self.encoders[v](view_input)
            outputs.append(out_v)

        return torch.stack(outputs, dim=1)
