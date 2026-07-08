"""
Examples import check — WP0.4 of MODERNIZATION_PLAN.md.

For every examples/**/*.py, parse the AST (never execute) and assert that every
`gds_fdtd` symbol it imports actually exists in the installed package. This makes
it impossible for examples to silently rot when APIs are removed (as happened to
examples 01/05/07/08 when t3d_tools/make_t3d_sim were deleted in Aug 2025).

Scope rules (per plan):
- Runs meaningfully only when optional extras are importable; a missing THIRD-PARTY
  optional dep (tidy3d, gdsfactory, ...) is a skip, not a failure.
- Known-broken legacy examples are xfailed until their WP6.1 rewrite.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).parent.parent / "examples"

# Legacy examples importing APIs deleted in Aug 2025 — rewritten in WP6.1.
KNOWN_BROKEN: dict[str, str] = {}  # WP6.1: all examples rewritten to current APIs

EXAMPLE_FILES = sorted(EXAMPLES_DIR.rglob("*.py"))


def _gds_fdtd_imports(tree: ast.Module):
    """Yield (module, symbol_or_None) for every gds_fdtd import in the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "gds_fdtd" or alias.name.startswith("gds_fdtd."):
                    yield alias.name, None
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "gds_fdtd" or node.module.startswith("gds_fdtd."):
                for alias in node.names:
                    yield node.module, alias.name


@pytest.mark.parametrize("example", EXAMPLE_FILES, ids=lambda p: str(p.relative_to(EXAMPLES_DIR)))
def test_example_imports_resolve(example: pathlib.Path):
    if example.name in KNOWN_BROKEN:
        pytest.xfail(f"{example.name}: {KNOWN_BROKEN[example.name]} — rewritten in WP6.1")

    tree = ast.parse(example.read_text(encoding="utf-8"), filename=str(example))

    for module_name, symbol in _gds_fdtd_imports(tree):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            if e.name and not e.name.startswith("gds_fdtd"):
                # a gds_fdtd module that itself requires an optional extra
                pytest.skip(f"optional dependency not installed: {e.name}")
            pytest.fail(f"{example.name}: gds_fdtd module {module_name!r} does not exist")
        except ImportError as e:
            pytest.skip(f"optional dependency import problem: {e}")

        if symbol is not None and not hasattr(module, symbol):
            pytest.fail(f"{example.name}: {module_name}.{symbol} does not exist")
