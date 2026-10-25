"""One trainer for every dataset, with the encoder widths passed in.

Same optimiser, loss, annealing schedule, per-view loss weight, batch
iteration and same-seed-same-init discipline as `run_ait_clean.train_ait` and
`run_lanl.train_lanl`. The one change is `native_dims`: those two build their
`EncoderConfig` list through a dataset module's `view_configs(sources, ...)`,
which reads `FEATURES[source]`. Here the widths arrive as ints.
"""

from __future__ import annotations

from typing import Sequence

import torch

from prismflow.data.loaders import iter_batches
from prismflow.models.encoders import EncoderConfig
from prismflow.models.prismflow import PrismFlow
from prismflow.train import compute_loss
from prismflow.utils.seed import set_seed


def encoder_configs(native_dims: Sequence[int], hidden_dims=(64, 64),
                    feature_dim: int = 32,
                    dropout: float = 0.0) -> list[EncoderConfig]:
    """One EncoderConfig per view at that view's NATIVE input width.

    Declaring the native width is what keeps the padded columns unreachable,
    which is why a wrong width here is not a performance bug but a silent
    change to what the model can see.
    """
    return [
        EncoderConfig(
            input_dim=int(d),
            hidden_dims=list(hidden_dims),
            feature_dim=feature_dim,
            dropout=dropout,
        )
        for d in native_dims
    ]


def train(dataset, indices, native_dims: Sequence[int], n_classes: int,
          seed: int, use_discount: bool, model_cfg: dict,
          training_cfg: dict) -> PrismFlow:
    """Train one PrismFlow on `indices` of `dataset`.

    `dataset` need only satisfy what `iter_batches` asks of it; this function
    never reads `.sources` or any other dataset-specific attribute, which is
    what makes it reusable. `model_cfg` carries hidden_dims, feature_dim and
    dropout; `training_cfg` carries lr, epochs, batch_size, anneal_epochs and
    per_view_loss_weight -- the same two config blocks both runners read.

    No validation split and no early stopping: Part 28 has neither, and a
    split inside the training scenario would be a within-scenario split.
    """
    set_seed(seed)
    model = PrismFlow(
        encoder_configs(
            native_dims,
            hidden_dims=model_cfg["hidden_dims"],
            feature_dim=model_cfg["feature_dim"],
            dropout=model_cfg["dropout"],
        ),
        n_classes=n_classes,
        use_discount=use_discount,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=training_cfg["lr"])
    for epoch in range(training_cfg["epochs"]):
        model.train()
        for batch in iter_batches(
            dataset, indices, training_cfg["batch_size"], shuffle=True,
            seed=seed + epoch,
        ):
            targets = torch.nn.functional.one_hot(
                batch.labels, num_classes=n_classes
            ).to(batch.views.dtype)
            loss = compute_loss(
                model(batch.views, batch.view_mask),
                targets,
                epoch=epoch,
                anneal_epochs=training_cfg["anneal_epochs"],
                per_view_loss_weight=training_cfg["per_view_loss_weight"],
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    model.eval()
    return model
