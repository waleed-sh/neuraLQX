# Documentation infrastructure

This file is for maintainers and is excluded from the published site. The authored
pages are intentionally empty skeletons; API content is generated from source.

## Build and preview

From the repository root:

```sh
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r docs/requirements.txt
.venv-docs/bin/python -m mkdocs serve
```

The release check is `.venv-docs/bin/python -m mkdocs build --strict`. Read the Docs
and CI use the same requirements and strict build. The GitHub workflow builds only
for published, stable releases tagged `v2.0.0` or newer (the `v` prefix is optional).
It does not run on pushes or pull requests. Prereleases and older release versions
skip the docs build. Only documentation dependencies are needed; API generation
never imports neuraLQX or installs the simulation stack.
Keep the direct dependency pins in `requirements.txt` and both `docs` dependency
lists in `../pyproject.toml` aligned when upgrading them.

## Navigation and appearance

`../mkdocs.yml` defines the authored page order. Each component has an `index.md`
overview and separate topic pages. Add pages there as the documentation grows.
The shared template in `overrides/templates/navigation.html` supports arbitrarily
nested sections on desktop and mobile, with the active section open by default.
Colours and layout adjustments live in `assets/stylesheets/theme.css`.

## Notebooks

Author tutorials in `tutorials/notebooks/` and add them to the navigation. Execute a
notebook explicitly in a simulation environment with neuraLQX and the appropriate
kernel installed, then save and commit the `.ipynb` **including its outputs**.
For example, after selecting the correct kernel:

```sh
jupyter nbconvert --to notebook --execute --inplace docs/tutorials/notebooks/first-simulation.ipynb
```

MkDocs uses `execute: false`, even on a clean RTD build. Its conversion cache only
saves rendering time; saved notebook outputs remain the source of truth. Notebook
downloads include those outputs. Re-execute only when intentionally updating the
example. The three initial notebooks contain no code cells.

## API generation

`hooks/api.py` discovers public packages and modules with an explicit `__all__`,
separates `neuralqx.experimental`, and generates pages and nested navigation in
memory on every build. The Python handler resolves public re-exports from source.
Private paths are omitted. Add exports to the Python package rather than editing
generated Markdown. The experimental namespace is ready for future additions.

`api/changelog.md` is the placeholder for the manually maintained changelog.

## Infrastructure checks

```sh
.venv-docs/bin/python -m unittest discover -s tests/docs -v
```

These checks cover namespace discovery, nested navigation, dependency alignment,
and rendering saved notebook output without executing code.
