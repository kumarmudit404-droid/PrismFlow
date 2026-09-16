"""YAML-backed configuration with dot access.

No paths are hard-coded here: every path used elsewhere in the codebase must
come from a loaded config file (e.g. configs/default.yaml), not from a
literal in source.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator

import yaml


class Config:
    """A read-only, dict-backed config object supporting dot access.

    Example:
        cfg = Config.from_yaml("configs/default.yaml")
        cfg.seed.list          # -> [0, 1, 2, 3, 4]
        cfg.paths.results_dir  # -> "results"
        cfg["seed"]["list"]    # dict-style access also works
    """

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", {})
        for key, value in data.items():
            self._data[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(item) for item in value]
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping, got {type(data)!r} in {path}")
        return cls(data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(f"Config has no key '{name}'") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Config is read-only; construct a new Config to change values")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            return self.to_dict() == other.to_dict()
        return NotImplemented

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        def unwrap(value: Any) -> Any:
            if isinstance(value, Config):
                return value.to_dict()
            if isinstance(value, list):
                return [unwrap(item) for item in value]
            return value

        return {key: unwrap(value) for key, value in self._data.items()}

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self.to_dict()))


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a dot-accessible Config object."""
    return Config.from_yaml(path)
