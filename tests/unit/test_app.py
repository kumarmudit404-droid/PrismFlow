"""Part 16: the demo layer must stay a demo layer.

The Part 16 brief requires a test asserting `app.py` imports no `torch.nn`
module directly. The rule it protects is broader than the letter: the app and
its panels must not contain model logic, because a demo that recomputes a
quantity its own way is how a figure that no experiment produced ends up on a
screen -- and then in a paper.

These tests parse the source with `ast` rather than importing it. Importing
`app.py` would execute Streamlit page setup, and importing is also the wrong
instrument: it cannot tell a direct `torch.nn` import from one pulled in
transitively by the engine, which is expected and fine.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app.py"
PANELS = ROOT / "prismflow" / "app" / "panels.py"
DEMO_FILES = (APP, PANELS)

# Names that would mean model logic had migrated into the demo layer.
FORBIDDEN_MODULES = ("torch.nn", "torch.nn.functional", "torch.optim")
FORBIDDEN_ATTRIBUTES = ("torch.nn", "nn.Module", "nn.Linear", "torch.optim")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(tree: ast.Module):
    """Every module name this file imports, as written."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
                for alias in node.names:
                    names.add(f"{node.module}.{alias.name}")
    return names


@pytest.mark.parametrize("path", DEMO_FILES, ids=lambda p: p.name)
def test_demo_layer_exists(path: Path):
    assert path.exists(), f"{path} is required by the Part 16 brief"


@pytest.mark.parametrize("path", DEMO_FILES, ids=lambda p: p.name)
def test_demo_layer_does_not_import_torch_nn(path: Path):
    """THE BRIEF'S TEST. No direct torch.nn (or torch.optim) import."""
    imported = _imported_modules(_tree(path))
    offending = sorted(name for name in imported if name in FORBIDDEN_MODULES)
    assert not offending, (
        f"{path.name} imports {offending} directly. The demo layer must call the "
        "engine, not build or train models itself."
    )
    # `from torch import nn` registers as module "torch" + name "torch.nn"
    assert "torch.nn" not in imported, f"{path.name} does `from torch import nn`"


def _attribute_chains(tree: ast.Module):
    """Every dotted attribute access, as a string, e.g. "torch.nn.Linear".

    Inspects AST nodes rather than the source text. String matching would trip
    over the docstrings in these files, which discuss `torch.nn` precisely
    because the rule against using it is the point.
    """
    chains = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parts, current = [], node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            chains.add(".".join(reversed(parts)))
    return chains


@pytest.mark.parametrize("path", DEMO_FILES, ids=lambda p: p.name)
def test_demo_layer_has_no_nn_attribute_access(path: Path):
    """Catches `torch.nn.Linear(...)` reached through a bare `import torch`."""
    chains = _attribute_chains(_tree(path))
    for forbidden in FORBIDDEN_ATTRIBUTES:
        offending = sorted(c for c in chains if c == forbidden or c.startswith(forbidden + "."))
        assert not offending, f"{path.name} touches {offending}"


def test_app_does_not_define_a_training_loop():
    """No optimiser step, no backward pass, no gradient handling in the demo."""
    for path in DEMO_FILES:
        chains = _attribute_chains(_tree(path))
        called = {c.rsplit(".", 1)[-1] for c in chains}
        for needle in ("backward", "zero_grad", "state_dict", "requires_grad", "requires_grad_"):
            assert needle not in called, f"{path.name} calls {needle!r}"


def test_app_delegates_training_to_the_existing_engine():
    """The app must call the experiments' own train functions, not its own."""
    imported = _imported_modules(_tree(APP))
    assert "experiments.calibration.run_calibration.train" in imported
    assert "experiments.real.run_real.train_real" in imported


def test_app_defines_every_scenario_the_brief_requires():
    """clean / clone / missing / noisy / Chorus / adaptive."""
    tree = _tree(APP)
    scenarios = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SCENARIOS":
                    scenarios = {k.value for k in node.value.keys}
    assert scenarios is not None, "app.py must define SCENARIOS"
    assert scenarios == {"clean", "clone", "missing", "noisy", "chorus", "adaptive"}


def test_panels_expose_every_required_result_element():
    """Screen 4 of the brief, plus the independence budget screen."""
    functions = {
        node.name
        for node in ast.walk(_tree(PANELS))
        if isinstance(node, ast.FunctionDef)
    }
    for required in (
        "render_independence_budget",
        "render_prediction",
        "render_eniv",
        "render_dependence_heatmap",
        "render_tail_dependence",
        "render_evidence_bars",
        "render_suspicion",
        "render_view_status",
        "render_saved_plots",
    ):
        assert required in functions, f"panels.py is missing {required}"


def test_panels_do_not_import_the_engine():
    """Panels render values handed to them; they must not fetch their own."""
    imported = _imported_modules(_tree(PANELS))
    engine = sorted(n for n in imported if n.startswith("prismflow"))
    assert not engine, f"panels.py imports engine modules {engine}; it should take values"
