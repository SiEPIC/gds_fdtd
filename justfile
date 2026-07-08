# Canonical dev tasks — humans and CI run the same commands.
# Install just: https://github.com/casey/just  (or read the recipes and run them directly)

# Show available recipes
default:
    @just --list

# Install package + dev tools (editable)
install:
    pip install -e ".[dev]"

# Run linters exactly as CI does
lint:
    ruff check .
    ruff format --check .

# Auto-fix lint + formatting
fix:
    ruff check . --fix
    ruff format .

# Type check (advisory during Phase 0/1)
typecheck:
    mypy src/gds_fdtd

# Run the test suite with coverage
test:
    pytest --cov=gds_fdtd --cov-branch --cov-report=term-missing tests

# Fast test run (the pre/post-WP gate)
gate:
    ruff check .
    pytest -q tests

# Build documentation
docs:
    pip install -e ".[docs]"
    cd docs && make html

# Build sdist + wheel
build:
    python -m build

# Remove build/test artifacts (not the 2.6 GB of local sim outputs — see clean-artifacts)
clean:
    rm -rf build/ dist/ *.egg-info/ docs/_build/ .coverage coverage.xml
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Remove large local simulation outputs (hdf5 etc.) — untracked, safe to delete
clean-artifacts:
    find examples -name "*.hdf5" -size +10M -delete
    rm -f mode_solver.hdf5
