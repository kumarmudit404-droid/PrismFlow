"""Two-timescale training for a learned dependence estimator, with an auditor.

THE PROBLEM THIS SOLVES

If the amortized estimator is trained jointly with the encoder, and its output
feeds the discount, then gradient descent has two ways to reduce the loss:

  1. learn genuinely independent representations, or
  2. add orthogonal noise to the evidence so that MEASURED dependence falls
     while actual redundancy is unchanged.

The second is far cheaper. A few extra directions of noise decorrelate the
evidence vectors without the encoder learning anything, measured dependence
drops, the discount stops discounting, and the model becomes confident again
while displaying an excellent independence score. **A measure stops being a
measure the moment it becomes a target.**

V1 avoids this by keeping alpha out of the graph entirely (see
`prismflow/eniv/discount.py`): the estimator has no parameters, so there is
nothing to game and no gradient path into it. A LEARNED estimator cannot take
that route, because it must be fitted to something. So two defences replace it.

DEFENCE 1: ALTERNATING OPTIMISATION

Encoder and estimator are never updated on the same step, as in actor-critic or
GAN training:

    N_enc steps   update the encoder, estimator FROZEN
    N_est steps   update the estimator, encoder FROZEN

Default ratio 5:1. Freezing is by `requires_grad`, and the frozen side's
parameters are asserted unchanged across its frozen phase in
`tests/unit/test_per_sample.py`. Alternating does not by itself make gaming
impossible -- it makes it slower and visible, because the estimator is always
catching up to an encoder that moved, and the gap that opens is measurable.

DEFENCE 2: A HELD-OUT AUDITOR

A second dependence estimate, computed on HELD-OUT data by the V1 statistical
estimator, which is never in any loss and has no parameters to fit. Every epoch
both are recorded:

    trained_dependence   what the amortized estimator reports
    audited_dependence   what the untrainable statistical estimator measures

If the encoder is genuinely learning independent representations, both fall
together. If the encoder is gaming the estimator, the trained value falls and
the audited value does not, and **the gap between them is the evidence**. The
gap is logged every epoch by `AuditLog`, and
`experiments/per_sample/run_per_sample.py` plots it. A run without that plot is
not evidence that gaming did not occur; it is an absence of evidence either way.

The auditor is deliberately the V1 estimator, not a second learned one: a
learned auditor could be gamed too, and an auditor with no parameters cannot be.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from prismflow.data.loaders import iter_batches
from prismflow.eniv.amortized import dependence_fit_loss
from prismflow.statistics.dependence import available_views, feature_dependence_matrix
from prismflow.train import compute_loss


@dataclass
class TwoTimescaleConfig:
    encoder_steps: int = 5
    estimator_steps: int = 1
    encoder_lr: float = 1e-3
    estimator_lr: float = 1e-3
    epochs: int = 40
    batch_size: int = 64
    anneal_epochs: int = 10
    per_view_loss_weight: float = 1.0
    audit_every: int = 1
    null_permutations: int = 8

    def validate(self) -> None:
        if self.encoder_steps < 1 or self.estimator_steps < 1:
            raise ValueError("encoder_steps and estimator_steps must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")


@dataclass
class AuditLog:
    """Per-epoch record. `gap` is the gaming evidence; positive means the
    trained estimator reports LESS dependence than the auditor measures."""

    epoch: list[int] = field(default_factory=list)
    encoder_loss: list[float] = field(default_factory=list)
    estimator_loss: list[float] = field(default_factory=list)
    trained_dependence: list[float] = field(default_factory=list)
    audited_dependence: list[float] = field(default_factory=list)
    gap: list[float] = field(default_factory=list)
    wall_clock: list[float] = field(default_factory=list)

    def record(self, epoch, encoder_loss, estimator_loss, trained, audited, elapsed):
        self.epoch.append(int(epoch))
        self.encoder_loss.append(float(encoder_loss))
        self.estimator_loss.append(float(estimator_loss))
        self.trained_dependence.append(float(trained))
        self.audited_dependence.append(float(audited))
        self.gap.append(float(audited - trained))
        self.wall_clock.append(float(elapsed))

    def as_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "encoder_loss": self.encoder_loss,
            "estimator_loss": self.estimator_loss,
            "trained_dependence": self.trained_dependence,
            "audited_dependence": self.audited_dependence,
            "gap": self.gap,
            "wall_clock": self.wall_clock,
        }

    @property
    def final_gap(self) -> float:
        return self.gap[-1] if self.gap else float("nan")

    @property
    def max_gap(self) -> float:
        return max(self.gap) if self.gap else float("nan")


@contextmanager
def frozen(module: torch.nn.Module):
    """Freeze a module's parameters for the duration of the block."""
    previous = [p.requires_grad for p in module.parameters()]
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(module.parameters(), previous):
            parameter.requires_grad_(flag)


def _mean_off_diagonal(matrix) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[-1]
    off = ~np.eye(n, dtype=bool)
    values = matrix[..., off]
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


@torch.no_grad()
def audit_dependence(model, dataset, indices, batch_size, config: TwoTimescaleConfig) -> float:
    """Mean off-diagonal dependence measured by the V1 estimator on HELD-OUT data.

    No parameters, no gradient, never in a loss. This is the number the encoder
    cannot move except by actually changing what its features share.
    """
    values = []
    for batch in iter_batches(dataset, indices, batch_size):
        features = model.encoder(batch.views, batch.view_mask)
        evidence = model.evidence_head(features, batch.view_mask)
        matrix = feature_dependence_matrix(
            features, evidence, batch.view_mask,
            method=model.dependence_method,
            conditioning=model.dependence_conditioning,
            seed=model.dependence_seed,
            null_permutations=config.null_permutations,
        )
        value = _mean_off_diagonal(matrix)
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def train_two_timescale(
    model,
    estimator,
    dataset,
    splits,
    n_classes: int,
    config: TwoTimescaleConfig | None = None,
    seed: int = 0,
    logger=None,
) -> AuditLog:
    """Alternate encoder and estimator updates; audit every epoch.

    `model` supplies `.encoder` and `.evidence_head`; `estimator` is an
    `AmortizedDependence`. The estimator is fitted to what the V1 statistical
    estimator measures on the same batch, which is why it is a dependence
    estimator rather than an arbitrary learned matrix.
    """
    config = config or TwoTimescaleConfig()
    config.validate()

    encoder_optimiser = torch.optim.Adam(model.parameters(), lr=config.encoder_lr)
    estimator_optimiser = torch.optim.Adam(estimator.parameters(), lr=config.estimator_lr)
    log = AuditLog()
    started = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        estimator.train()
        encoder_total = estimator_total = 0.0
        encoder_batches = estimator_batches = 0
        step = 0

        for batch in iter_batches(
            dataset, splits["train"], config.batch_size, shuffle=True, seed=seed + epoch
        ):
            cycle = config.encoder_steps + config.estimator_steps
            updating_encoder = (step % cycle) < config.encoder_steps
            step += 1

            if updating_encoder:
                # Estimator frozen: no gradient reaches psi on this step.
                with frozen(estimator):
                    targets = torch.nn.functional.one_hot(
                        batch.labels, num_classes=n_classes
                    ).to(batch.views.dtype)
                    output = model(batch.views, batch.view_mask)
                    loss = compute_loss(
                        output, targets, epoch=epoch,
                        anneal_epochs=config.anneal_epochs,
                        per_view_loss_weight=config.per_view_loss_weight,
                    )
                    encoder_optimiser.zero_grad()
                    loss.backward()
                    encoder_optimiser.step()
                encoder_total += float(loss.detach())
                encoder_batches += 1
            else:
                # Encoder frozen: psi is fitted to the statistical estimator's
                # reading of features the encoder is not currently changing.
                with frozen(model):
                    with torch.no_grad():
                        features = model.encoder(batch.views, batch.view_mask)
                        evidence = model.evidence_head(features, batch.view_mask)
                        target_matrix = feature_dependence_matrix(
                            features, evidence, batch.view_mask,
                            method=model.dependence_method,
                            conditioning=model.dependence_conditioning,
                            seed=model.dependence_seed,
                            null_permutations=config.null_permutations,
                        )
                    predicted = estimator(evidence.detach(), batch.view_mask)
                    loss = dependence_fit_loss(predicted, target_matrix, batch.view_mask)
                    estimator_optimiser.zero_grad()
                    loss.backward()
                    estimator_optimiser.step()
                estimator_total += float(loss.detach())
                estimator_batches += 1

        if epoch % config.audit_every == 0 or epoch == config.epochs - 1:
            model.eval()
            estimator.eval()
            with torch.no_grad():
                trained = []
                for batch in iter_batches(dataset, splits["val"], config.batch_size):
                    features = model.encoder(batch.views, batch.view_mask)
                    evidence = model.evidence_head(features, batch.view_mask)
                    trained.append(_mean_off_diagonal(estimator(evidence, batch.view_mask).cpu().numpy()))
                trained_value = float(np.nanmean(trained)) if trained else float("nan")
                audited_value = audit_dependence(
                    model, dataset, splits["val"], config.batch_size, config
                )
            log.record(
                epoch,
                encoder_total / max(encoder_batches, 1),
                estimator_total / max(estimator_batches, 1),
                trained_value,
                audited_value,
                time.perf_counter() - started,
            )
            if logger is not None:
                logger.info(
                    "epoch %3d enc=%.4f est=%.4f trained_dep=%.4f audited_dep=%.4f gap=%+.4f",
                    epoch, log.encoder_loss[-1], log.estimator_loss[-1],
                    trained_value, audited_value, log.gap[-1],
                )

    model.eval()
    estimator.eval()
    return log
