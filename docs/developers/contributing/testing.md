---
title: Testing
---

# Testing

Tests establish the behaviour that a contribution promises. For scientific software, that includes the meaning of the numerical result, its approximation error, and the transformations under which it remains valid. A test suite should help a reviewer distinguish a correct implementation from one that merely runs.

New functionality must include tests. A bug fix should include a small regression test that fails for the original problem and passes with the correction. Documentation-only prose changes do not require artificial source tests, but changed examples must be executed and the documentation must build successfully.

## Running the relevant tests

neuraLQX uses pytest with the configuration in [pyproject.toml](https://github.com/waleed-sh/neuraLQX/blob/main/pyproject.toml). Test files are collected under `tests/` and use names beginning with `test_`. Start with the files that exercise the affected component.

```sh
python -m pytest tests/test_sampler.py -q
```

Use a test name or keyword to narrow the run while diagnosing a failure.

```sh
python -m pytest tests/test_sampler.py -k log_correction -q
```

Once the focused tests pass, run neighbouring tests when the change affects a shared contract. A sampler change can affect variational states and drivers. An operator change can affect expectations, gradients, and preconditioners. Choose that broader scope from the actual dependencies of the change.

```sh
JAX_PLATFORMS=cpu python -m pytest tests/test_sampler.py tests/test_vqs.py tests/test_driver.py -q
```

For broad changes, run the full suite from the repository root.

```sh
python -m pytest
```

A full run can require more time and resources than a focused one. Some tests start additional processes or exercise device layouts. If your environment cannot run part of the required coverage, state that limitation and the exact commands you did run. Do not report a skipped hardware path as tested.

## Numerical references and regression tests

Use a reference that is independent of the implementation being tested. On a small Hilbert space, direct enumeration or a dense matrix can establish operator actions, normalized expectations, overlaps, and gradients. For a proposal rule, a finite transition matrix can verify detailed balance exactly. For a constrained space, check validity and reachability separately.

Avoid constructing the expected answer by repeating the same algorithm with slightly different variable names. Two matching implementations can share the same error. A regression test should isolate the condition responsible for the failure, such as an asymmetric proposal, a non-Born sampling density, a complex conjugation, or an exact wavefunction node.

Check numerical values and gradients separately when both are part of the interface. A forward value can be correct while a cached or stopped derivative is wrong. Compare automatic derivatives with a direct small-system expression or finite differences over several step sizes. For complex parameters, specify the real-coordinate or holomorphic convention being tested.

Tolerances should reflect dtype, conditioning, and approximation controls. Prefer explicit absolute and relative tolerances. Explain unusual tolerances in a comment and demonstrate that they distinguish the intended result from a meaningful error. Do not increase a tolerance solely because a test failed.

## Randomness and sampling tests

Use explicit JAX keys or fixed seeds and keep tests independent of execution order. A reproducible seed is useful for debugging, but it does not turn a statistically fragile assertion into a reliable one.

Where possible, test proposal probabilities, corrections, support, and normalization on a small enumerated space. Reserve long Monte Carlo runs for properties that require sampling, and choose acceptance bounds or confidence intervals from the expected statistical error. A single successful trajectory is not evidence of ergodicity or correct equilibrium weights.

For importance sampling, test the physical numerator and normalization together. Include different proposal distributions, arbitrary common density normalizations, and support edge cases relevant to the implementation. Retain the density of the batch actually generated when testing adaptive sampling. The [sampler extension guides](../samplers/index.md) give examples of these distinctions.

## JAX, dtypes, and execution modes

Exercise compiled execution when a function is intended for JIT. Check eager and compiled results where that comparison can expose tracing errors. Test relevant real and complex dtypes, and include differentiation when the function participates in variational gradients or SR.

Cover shape boundaries that affect the algorithm, such as singleton batches, a final incomplete chunk, heterogeneous local dimensions, or zero connections. Test empty input only when the interface supports it or specifies an error. A function should not silently accept a shape that changes the meaning of an axis.

Distributed and accelerator claims require checks on those execution paths. A CPU result alone does not establish correct collective reductions or device placement. When testing performance, separate compilation from execution, synchronise device results, and record the workload and hardware. Keep timing assertions out of ordinary correctness tests when machine load would make them unreliable. Use the [profiling guides](../profiling/index.md) for measurement procedures.

The registered markers are `gpu`, `integration`, `jaxjit`, and `slow`. Apply them consistently. Registration describes a test category and does not itself skip a test or provide the required hardware. A hardware-dependent test must check its prerequisites and skip explicitly when they are unavailable. Select marker groups deliberately, for example:

```sh
python -m pytest -m "not gpu and not slow"
```

Use fixtures to share meaningful setup and parametrisation to cover cases that exercise distinct behaviours. Keep failures local and understandable. Clean up temporary files and processes, and avoid relying on a developer's datasets, credentials, current directory contents, or network access for ordinary unit tests.

## Documentation checks

Run both the infrastructure tests and the strict build in the documentation environment.

```sh
python -m unittest discover -s tests/docs -v
python -m mkdocs build --strict
```

The infrastructure checks cover API discovery, nested navigation, dependency alignment, and rendering saved notebook outputs without executing cells. The build checks navigation and page references, but it does not execute the Python examples or establish that every API anchor refers to the intended object. Follow changed links in the rendered site and inspect the relevant API pages.

Execute each complete new or changed example in the documented environment. Check output shapes, assertions, displayed values, and imports. For notebooks, run them intentionally during development and save their outputs. The documentation build must continue to render those outputs without executing the notebook again.

When changing layout, navigation, figures, or equations, inspect the rendered result. Check narrow layouts for overflowing equations or tables and verify dark-mode readability when appearance changes. Restore any temporary browser settings and stop temporary preview servers after checking the site.

## Reporting validation in a pull request

State which commands passed, which failed, and which could not be run. Include the backend and relevant dependency versions when they affect interpretation. If the change fixes a numerical error, describe the independent reference and the case that failed before the fix.

For a new feature, explain how the tests cover the public behaviour, important edge cases, and relevant integration points. For a documentation change, identify the examples you executed and confirm the strict build result. This gives reviewers a concrete basis for assessing the contribution and its remaining limitations.
