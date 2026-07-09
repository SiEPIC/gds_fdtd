## What

<!-- One paragraph: what changes and why. Link the issue if one exists. -->

## Validation

<!-- How was this verified? Paste the relevant command + result. -->

- [ ] `uv run pytest tests` passes locally (paste the exit line, not `| tail`)
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run codespell src tests` clean
- [ ] New behavior has a test that fails without the change
- [ ] No `cloud`/`licensed` test runs in CI paths (marker taxonomy respected)

## Cost

<!-- If anything here spent cloud credits or license time, say how much. -->
