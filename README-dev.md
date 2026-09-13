# Development Guide

This document provides instructions for developers on how to set up and run checks before making changes to the repository.

## Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to run automated checks on your code before you commit it. This helps catch issues early and ensures consistency.

### Installation

1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```

2. Install the pre-commit hooks in your local repository:
   ```bash
   pre-commit install
   ```

### Running Checks

- **Before committing**: The hooks will run automatically when you commit. If any checks fail, the commit will be blocked until you fix the issues.

- **Run all checks manually** (recommended before pushing):
  ```bash
  pre-commit run --all-files
  ```

- **Run specific checks**:
  ```bash
  pre-commit run markdown-link-check --all-files
  ```

### What the Checks Do

- **Markdown Link Check**: Verifies that all links in `.md` files are working and not broken. This includes:
  - External URLs (excluding YouTube links to avoid rate limiting)
  - Relative links within the repository
  - GitHub links

### Configuration

- `.pre-commit-config.yaml`: Defines which hooks to run
- `.markdown-link-check.json`: Configuration for the link checker (ignores YouTube links, sets timeouts, etc.)

### Troubleshooting

- If a link check fails for a valid link, you may need to update the configuration in `.markdown-link-check.json`
- For local development, you can temporarily disable hooks with `git commit --no-verify`, but make sure to run checks manually before pushing

## GitHub Actions

In addition to local pre-commit checks, we have GitHub Actions that run the same checks in CI:

- **Link Check Workflow**: Runs on pushes and pull requests to check for broken links, in two
  passes — repository markdown (with on-disk file links) and `docs/` (web links only, since
  those pages use MkDocs-relative links). Shared exclusions live in `.lycheeignore`.
- **Deploy docs Workflow**: Builds the site with `mkdocs build --strict` and publishes it to
  <https://ubern-mia.github.io/bender/>. The strict build is what validates internal links
  and image paths under `docs/`.

## Documentation site

The website lives in `docs/`, configured by `mkdocs.yml`:

```bash
make docs-install   # add mkdocs-material to .venv
make docs-serve     # live preview at http://127.0.0.1:8000
make docs-build     # render into site/ (same strict build CI runs)
```

Episode figures are not copied into `docs/`. They stay next to the code that produced them,
and `hooks/repo_assets.py` publishes them under `/figures/` at build time. The checklist and
glossary pages include the repository markdown via `pymdownx.snippets` section markers
(`<!-- --8<-- [start:body] -->`), so those files remain the single source of truth — do not
remove those marker comments.

Make sure your local checks pass before opening a pull request, as the CI will also run these checks.