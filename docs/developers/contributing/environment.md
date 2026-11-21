---
title: Development environment
---

# Development environment

Work from a clone of your fork with a contribution branch based on the project's `main`. The [contribution workflow](../../api/contributing.md#fork-the-repository-and-create-a-branch) gives the Git commands. Run the commands on this page from the repository root so that installation, tests, and documentation use the intended checkout.

The [project configuration](https://github.com/waleed-sh/neuraLQX/blob/main/pyproject.toml) records Python requirements, runtime dependencies, development extras, and tool settings. Use an isolated environment rather than modifying the Python installation used for an unrelated simulation.

## Source and test development

Create and activate a virtual environment, then install the checkout in editable mode with the `dev` and `docs` extras.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

The activation command above uses a POSIX shell. With PowerShell on Windows, activate the same environment using `.venv\Scripts\Activate.ps1`. The subsequent `python -m ...` commands refer to the active environment in either case.

Editable installation makes source changes in the checkout available without reinstalling the package after every edit. The `dev` extra supplies tools including pytest, Ruff, coverage, and mypy. The `docs` extra supplies the documentation builder and theme. If package metadata, dependencies, or entry points change, reinstall the editable package so the environment reflects those changes.

Confirm the import comes from the checkout you intend to modify.

```python
import neuralqx

print(neuralqx.__file__)
```

Most focused numerical tests can run on CPU. To select it explicitly for a test command in a POSIX shell, use `JAX_PLATFORMS=cpu` as shown in the [testing guide](testing.md). Accelerator-specific changes should also be checked in the intended environment. The [GPU](../../distributed/single-gpu.md) and [multiple-device](../../distributed/multi-gpu.md) guides are the place for simulation-specific setup.

## Documentation-only development

The documentation can be built without installing the simulation stack. Use a separate environment containing the pinned documentation requirements.

```sh
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install --upgrade pip
python -m pip install -r docs/requirements.txt
python -m mkdocs build --strict
```

API generation analyses the source without importing neuraLQX. A successful build therefore checks the site structure and rendering, but does not execute scientific code examples. Run those examples in the full development environment when their content changes.

If a documentation dependency changes, keep `docs/requirements.txt`, the `docs` optional dependency extra, and the `docs` dependency group in `pyproject.toml` consistent. The [documentation infrastructure tests](testing.md#documentation-checks) verify that alignment.

## Formatting and static checks

Use the project's Ruff settings instead of choosing a separate formatter configuration. The following commands check the complete source and test trees. During focused development, replace those directories with the specific files you changed.

```sh
python -m ruff check neuralqx tests
python -m ruff format --check neuralqx tests
```

Apply formatting with `python -m ruff format` followed by the intended file paths. Use `ruff check --fix` only after inspecting which changes it proposes, then review the diff. Formatting is part of preparing a contribution, but a formatter does not verify mathematical behaviour or API compatibility.

For changes involving annotated interfaces, run the configured type checker on the affected code. The project-wide command is:

```sh
python -m mypy
```

Report relevant pre-existing failures separately from failures introduced by the patch. Do not weaken checks or include a repository-wide formatting rewrite merely to make a focused contribution appear clean. The [source guide](index.md) explains docstring and annotation expectations.

## Previewing the documentation

Start a local preview when you need to inspect links, navigation, code blocks, or equations.

```sh
python -m mkdocs serve --dev-addr 127.0.0.1:8000
```

Open the address printed by MkDocs. If that port is already in use, choose another available port. Stop the preview with `Ctrl+C` when finished. Keep the strict build as a separate check, since a running preview is not the same as a successful release build.

The documentation workflow on GitHub builds published stable releases starting with v2.0.0. It does not build ordinary pushes or pull requests. Run the strict build locally and include its result in documentation pull requests.

## Keeping the working tree focused

Use `git status --short` and `git diff` before staging changes. Virtual environments, caches, built sites, benchmark outputs, and local simulation data do not belong in an ordinary source or documentation patch. Saved notebook outputs are an intentional exception when they form part of the tutorial being updated.

When adding a dependency, explain why existing dependencies are insufficient and how the new dependency affects installation and supported environments. For substantial dependency or architecture changes, discuss the proposal in an issue before implementing it, following the [contribution policy](../../api/contributing.md#choose-and-discuss-the-change).
