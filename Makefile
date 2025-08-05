.PHONY: help clean test docs docs-serve bump-patch bump-minor bump-major release install check-version

help:	## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:	## Install package in development mode
	pip install -e .[dev]

test:	## Run tests
	pytest --cov=gds_fdtd tests/

docs:	## Build documentation
	cd docs && make html

docs-serve:	## Build and serve documentation locally
	cd docs && make html && python -m http.server -d _build/html 8000

clean:	## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

bump-patch:	## Bump patch version (e.g., 0.4.0 -> 0.4.1)
	bump2version patch

bump-minor:	## Bump minor version (e.g., 0.4.0 -> 0.5.0)
	bump2version minor

bump-major:	## Bump major version (e.g., 0.4.0 -> 1.0.0)
	bump2version major

release: clean test docs	## Build and release package
	python -m build
	@echo "Package built successfully!"
	@echo "To upload to PyPI, run: twine upload dist/*"
	@echo "To upload to test PyPI, run: twine upload --repository testpypi dist/*"

check-version:	## Show current version information
	@echo "Current version information:"
	@echo "  pyproject.toml: $$(grep '^version' pyproject.toml | cut -d'"' -f2)"
	@echo "  __init__.py:    $$(grep '__version__' gds_fdtd/__init__.py | cut -d'"' -f2)"
	@echo "  docs/conf.py:   $$(grep "release = " docs/conf.py | cut -d"'" -f2)"
	@echo "  Git tags:       $$(git tag --sort=-version:refname | head -5 | tr '\n' ' ')" 