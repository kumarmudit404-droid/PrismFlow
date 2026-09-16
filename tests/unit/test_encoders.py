import pytest
import torch

from prismflow.models.encoders import (
    EncoderConfig,
    MLPEncoder,
    MultiViewEncoder,
    build_encoder,
    register_encoder,
)


def _make_configs(n_views=3, input_dim=10, feature_dim=16, hidden_dims=None):
    hidden_dims = hidden_dims if hidden_dims is not None else [32, 32]
    return [
        EncoderConfig(input_dim=input_dim, hidden_dims=hidden_dims, feature_dim=feature_dim, dropout=0.0)
        for _ in range(n_views)
    ]


# --- shapes -----------------------------------------------------------


def test_forward_shape():
    configs = _make_configs(n_views=4, input_dim=8, feature_dim=12)
    encoder = MultiViewEncoder(configs)
    views = torch.randn(5, 4, 8)
    mask = torch.ones(5, 4, dtype=torch.bool)
    out = encoder(views, mask)
    assert out.shape == (5, 4, 12)


def test_forward_shape_various_batch_and_view_counts():
    configs = _make_configs(n_views=2, input_dim=6, feature_dim=4)
    encoder = MultiViewEncoder(configs)
    views = torch.randn(1, 2, 6)
    mask = torch.ones(1, 2, dtype=torch.bool)
    out = encoder(views, mask)
    assert out.shape == (1, 2, 4)


def test_feature_dim_mismatch_raises():
    configs = [
        EncoderConfig(input_dim=4, feature_dim=8),
        EncoderConfig(input_dim=4, feature_dim=16),
    ]
    with pytest.raises(ValueError):
        MultiViewEncoder(configs)


def test_empty_view_configs_raises():
    with pytest.raises(ValueError):
        MultiViewEncoder([])


def test_view_mask_wrong_shape_raises():
    configs = _make_configs(n_views=3, input_dim=5, feature_dim=6)
    encoder = MultiViewEncoder(configs)
    views = torch.randn(4, 3, 5)
    bad_mask = torch.ones(4, 2, dtype=torch.bool)
    with pytest.raises(ValueError):
        encoder(views, bad_mask)


# --- weights are NOT shared by default ---------------------------------


def test_encoders_are_distinct_objects():
    configs = _make_configs(n_views=3, input_dim=6, feature_dim=8)
    encoder = MultiViewEncoder(configs, share_weights=False)

    for i in range(3):
        for j in range(i + 1, 3):
            assert encoder.encoders[i] is not encoder.encoders[j]
            for p_i, p_j in zip(encoder.encoders[i].parameters(), encoder.encoders[j].parameters()):
                assert p_i is not p_j


def test_encoder_params_diverge_after_optimizer_step():
    configs = _make_configs(n_views=2, input_dim=6, feature_dim=8)
    encoder = MultiViewEncoder(configs, share_weights=False)
    head = torch.nn.Linear(8, 3)

    w0_before = [p.clone() for p in encoder.encoders[0].parameters()]
    w1_before = [p.clone() for p in encoder.encoders[1].parameters()]

    views = torch.randn(16, 2, 6)
    mask = torch.ones(16, 2, dtype=torch.bool)
    labels = torch.randint(0, 3, (16,))

    optimizer = torch.optim.SGD(list(encoder.parameters()) + list(head.parameters()), lr=0.1)
    out = encoder(views, mask)
    logits = head(out.mean(dim=1))
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()

    w0_after = list(encoder.encoders[0].parameters())
    w1_after = list(encoder.encoders[1].parameters())

    # encoder 0 changed from its own starting point
    assert any(not torch.equal(a, b) for a, b in zip(w0_before, w0_after))
    # encoder 1 changed from its own starting point
    assert any(not torch.equal(a, b) for a, b in zip(w1_before, w1_after))
    # and the two encoders are not numerically identical to each other
    assert any(not torch.equal(a, b) for a, b in zip(w0_after, w1_after))


def test_share_weights_true_uses_identical_parameter_objects():
    configs = _make_configs(n_views=3, input_dim=6, feature_dim=8)
    encoder = MultiViewEncoder(configs, share_weights=True)
    for p0, p1 in zip(encoder.encoders[0].parameters(), encoder.encoders[1].parameters()):
        assert p0 is p1


# --- gradients respect view_mask ----------------------------------------


def test_gradients_flow_to_every_unmasked_encoder():
    configs = _make_configs(n_views=3, input_dim=5, feature_dim=7)
    encoder = MultiViewEncoder(configs)
    views = torch.randn(10, 3, 5)
    mask = torch.ones(10, 3, dtype=torch.bool)

    out = encoder(views, mask)
    loss = out.pow(2).sum()
    loss.backward()

    for v in range(3):
        grads = [p.grad for p in encoder.encoders[v].parameters()]
        assert all(g is not None for g in grads)
        assert any(torch.any(g != 0) for g in grads)


def test_fully_masked_view_receives_no_gradient():
    configs = _make_configs(n_views=3, input_dim=5, feature_dim=7)
    encoder = MultiViewEncoder(configs)
    views = torch.randn(10, 3, 5)
    mask = torch.ones(10, 3, dtype=torch.bool)
    mask[:, 1] = False  # view 1 masked out for the entire batch

    out = encoder(views, mask)
    loss = out.pow(2).sum()
    loss.backward()

    for p in encoder.encoders[1].parameters():
        assert p.grad is None or torch.all(p.grad == 0)

    for v in (0, 2):
        grads = [p.grad for p in encoder.encoders[v].parameters()]
        assert all(g is not None for g in grads)
        assert any(torch.any(g != 0) for g in grads)


def test_partially_masked_samples_output_zero_and_dont_poison_loss():
    configs = _make_configs(n_views=2, input_dim=4, feature_dim=6)
    encoder = MultiViewEncoder(configs)
    views = torch.randn(4, 2, 4)
    mask = torch.ones(4, 2, dtype=torch.bool)
    mask[:, 0] = False  # view 0 masked for every sample

    out = encoder(views, mask)
    assert torch.all(out[:, 0, :] == 0)
    assert torch.any(out[:, 1, :] != 0)


# --- per-view input_dim / registry (factory substitutability) -----------


def test_per_view_input_dim_can_differ():
    configs = [
        EncoderConfig(input_dim=8, feature_dim=6),
        EncoderConfig(input_dim=4, feature_dim=6),  # only uses first 4 of D=8 features
    ]
    encoder = MultiViewEncoder(configs)
    views = torch.randn(3, 2, 8)
    mask = torch.ones(3, 2, dtype=torch.bool)
    out = encoder(views, mask)
    assert out.shape == (3, 2, 6)


def test_build_encoder_unknown_type_raises():
    with pytest.raises(ValueError):
        build_encoder(EncoderConfig(input_dim=4, feature_dim=4, encoder_type="does_not_exist"))


def test_registering_new_encoder_type_keeps_multiview_call_signature():
    class TinyEncoder(torch.nn.Module):
        def __init__(self, input_dim, feature_dim):
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, feature_dim)

        def forward(self, x):
            return self.linear(x)

    register_encoder("tiny_test_encoder", lambda cfg: TinyEncoder(cfg.input_dim, cfg.feature_dim))

    configs = [
        EncoderConfig(input_dim=5, feature_dim=9, encoder_type="tiny_test_encoder") for _ in range(2)
    ]
    encoder = MultiViewEncoder(configs)

    views = torch.randn(3, 2, 5)
    mask = torch.ones(3, 2, dtype=torch.bool)
    out = encoder(views, mask)  # same call signature as the MLP-backed encoder
    assert out.shape == (3, 2, 9)


def test_mlp_encoder_direct_forward_shape():
    enc = MLPEncoder(input_dim=10, hidden_dims=[20], feature_dim=5, dropout=0.0)
    out = enc(torch.randn(7, 10))
    assert out.shape == (7, 5)
