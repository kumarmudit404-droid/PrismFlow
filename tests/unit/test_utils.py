import logging
import textwrap

import pytest

from prismflow.utils.checkpoint import load_checkpoint, save_checkpoint
from prismflow.utils.config import Config, load_config
from prismflow.utils.device import get_device
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

torch = pytest.importorskip("torch")


def test_config_dot_access(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            seed:
              list: [0, 1, 2, 3, 4]
            paths:
              results_dir: results
            """
        )
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)
    assert cfg.seed.list == [0, 1, 2, 3, 4]
    assert cfg.paths.results_dir == "results"
    assert cfg["paths"]["results_dir"] == "results"


def test_config_to_dict_roundtrip(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("a:\n  b: 1\n  c: [1, 2]\n")
    cfg = load_config(cfg_path)
    assert cfg.to_dict() == {"a": {"b": 1, "c": [1, 2]}}


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert len(cfg.seed.list) >= 5
    assert cfg.device.prefer in ("auto", "cpu", "cuda")


def test_set_seed_is_bit_identical():
    set_seed(123)
    a_torch = torch.rand(4, 4)
    a_numpy = __import__("numpy").random.rand(4, 4)

    set_seed(123)
    b_torch = torch.rand(4, 4)
    b_numpy = __import__("numpy").random.rand(4, 4)

    assert torch.equal(a_torch, b_torch)
    assert (a_numpy == b_numpy).all()


def test_set_seed_different_seeds_differ():
    set_seed(1)
    a = torch.rand(4, 4)
    set_seed(2)
    b = torch.rand(4, 4)
    assert not torch.equal(a, b)


def test_get_device_auto_returns_valid_device():
    device = get_device("auto")
    assert str(device) in ("cpu", "cuda")


def test_get_device_cpu_forced():
    device = get_device("cpu")
    assert str(device) == "cpu"


def test_get_device_invalid_preference_raises():
    with pytest.raises(ValueError):
        get_device("tpu")


def test_get_logger_returns_logger():
    logger = get_logger("prismflow_test_utils")
    assert isinstance(logger, logging.Logger)
    logger2 = get_logger("prismflow_test_utils")
    assert logger is logger2


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    state = {"epoch": 3, "value": torch.tensor([1.0, 2.0, 3.0])}
    path = save_checkpoint(state, tmp_path, "test_ckpt")
    assert path.exists()

    loaded = load_checkpoint(tmp_path, "test_ckpt")
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["value"], state["value"])


def test_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path, "does_not_exist")
