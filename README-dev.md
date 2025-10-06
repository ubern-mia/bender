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

- **Link Check Workflow**: Runs on pushes and pull requests to check for broken links

Make sure your local checks pass before opening a pull request, as the CI will also run these checks.