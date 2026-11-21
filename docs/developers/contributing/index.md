---
title: Contributing
---

# Contributing

This section describes how to make a contribution understandable, testable, and maintainable within neuraLQX. Start with [Contributing to neuraLQX](../../api/contributing.md) for the issue, fork, and pull-request workflow. The [environment guide](environment.md) provides installation and formatting commands, and the [testing guide](testing.md) explains the evidence expected with a change.

The same standards apply whether a contribution was written manually or with AI assistance. The author is responsible for its correctness and explanation. The project's [AI policy](../../api/contributing.md#using-ai-tools) also requires disclosure in the pull request.

## Source code and scientific contracts

Before implementing an extension, identify the interface it belongs to. The guides for [Hilbert spaces](../hilbert/index.md), [operators](../operators/index.md), [estimators](../estimators/index.md), [samplers](../samplers/index.md), [networks](../networks/index.md), and [graphs](../graphs/index.md) describe the contracts that other components rely on. Reuse an existing extension point when it expresses the intended behaviour.

Choose names that describe the physical quantity or computational responsibility. Keep a function focused on one task and make data dependencies explicit. For numerical kernels, document array shapes, axis order, local-label conventions, dtype assumptions, and the meaning of complex conjugation. State whether an input is a configuration label, a basis index, a log amplitude, a probability, or an unnormalised density when that distinction matters.

Use comments to explain a reason that is not apparent from the operations. A stability shift, a frozen sampling measure, a support restriction, or an approximation tolerance often needs a short mathematical explanation. Repeating the code in words adds little. If a derivation is substantial, put the full explanation in the relevant guide and retain a concise reference near the implementation.

Follow the repository's [Ruff configuration](https://github.com/waleed-sh/neuraLQX/blob/main/pyproject.toml). It uses a line length of 100, double-quoted strings, and one import per line. Add useful type annotations at interfaces, but do not use an annotation as a substitute for documenting a shape or semantic constraint. Keep unrelated formatting changes out of a scientific fix so that reviewers can identify what changed numerically.

For code intended to run under JAX transformations, keep dynamic numerical values in array arguments or pytree fields. Avoid hidden random generators, stateful host operations, and Python decisions based on traced values. The [struct guide](../structs/index.md) explains the distinction between numerical fields and static metadata. A change that claims compatibility with JIT, differentiation, or device partitioning must test the relevant behaviour.

## Docstrings

Document new public functions, classes, and methods at their definitions. Internal helpers also need a docstring or comment when their role, assumptions, or mathematics would otherwise be unclear. Start with a sentence describing what the object does, then explain the contract in the order a caller needs it.

Use Google-style sections for new and substantially revised docstrings. The documentation renderer is configured for this format. `Args`, `Returns`, `Raises`, `Attributes`, and `Examples` are useful where they apply. Describe defaults through their behaviour, identify which errors are checked, and distinguish conditions the implementation validates from assumptions the caller must satisfy. Do not add empty sections or copy the same explanation into every helper.

The following complete example demonstrates the style. It defines a local example function, rather than introducing a new neuraLQX API.

```python
import jax
import jax.numpy as jnp
import numpy as np

def normalized_born_weights(log_amplitudes: jax.Array, axis: int = -1) -> jax.Array:
    """Normalize Born weights along one configuration axis.

    Uses a softmax of twice the real log amplitude, so a common finite
    additive shift does not change the resulting probabilities.

    Args:
        log_amplitudes: Real or complex log amplitudes with at least one axis.
            Each normalized slice must contain a finite real log amplitude.
            Negative infinity may represent zero amplitude. NaN and positive
            infinity are outside this function's input contract.
        axis: Configuration axis to normalize. Other axes identify independent
            distributions. Keep this argument static when using JIT.

    Returns:
        Real nonnegative weights with the input shape. Each slice sums to one
        along the selected axis, up to floating-point precision.

    Raises:
        ValueError: The input has no axes or axis is out of range.

    Notes:
        Numerical input validity is a caller responsibility. The shape checks
        do not detect an entirely zero wavefunction or nonfinite input values.
    """
    values = jnp.asarray(log_amplitudes)
    if values.ndim == 0 or not -values.ndim <= axis < values.ndim:
        raise ValueError("axis must identify an input dimension")
    return jax.nn.softmax(2 * values.real, axis=axis)

logs = jnp.array([0.0, jnp.log(2.0)])
weights = jax.jit(normalized_born_weights)(logs)
np.testing.assert_allclose(weights, [0.2, 0.8], rtol=1e-6)
```

For classes, explain the purpose and construction choices before listing attributes. For operators and estimators, give the mathematical action or measured quantity when the name alone is insufficient. Explain approximation controls and support assumptions explicitly. Use a small example with real imports and complete setup when it helps a caller choose the API.

When adding an object to the public interface, update the appropriate package exports and `__all__`. API pages are generated from those declarations and source docstrings. They are not maintained as separate handwritten copies. Check the resulting page and link the first useful mention of the object from its guide to the generated API reference.

## Documentation changes

Documentation should explain what a reader is trying to calculate, which components are involved, and how the example implements that calculation. Introduce concepts before relying on them. Use connected paragraphs and descriptive headings, with tables or lists only where they make relationships easier to follow.

Keep scientific-use material in [Core concepts](../../concepts/index.md) and implementation contracts in the [Developer guide](../index.md). A new section or page needs an entry in `mkdocs.yml` at the appropriate level. Use relative Markdown links for internal pages, and link named API objects when their reference helps the reader understand available arguments or behaviour. Avoid repeating a complete explanation when a focused link provides the necessary context.

Write “Hilbert space”, “gauge invariant”, and “spin network” with spaces. Avoid em dashes, semicolons, emojis, promotional claims, and catchy titles. Explain the actual scientific or computational meaning. Keep terminology consistent across prose, figure labels, and docstrings.

Examples must use the current public interface and run from the setup shown. Include imports, parameters, random seeds, and any required construction steps. Execute complete examples before submitting them and check the displayed outputs. Clearly identify fragments that only illustrate a signature or an operation within an already explained calculation.

Use equations where they resolve an ambiguity, such as a normalization, a proposal correction, a gradient convention, or an operator action. Define the symbols and connect the equation to the code. Check the rendered mathematics and break long expressions across lines when necessary. Any figure should explain a relationship that is harder to understand from the text alone.

## Notebook contributions

Tutorial notebooks are published from their saved outputs. Execute a notebook intentionally when developing or updating that example, inspect the result, and save the notebook with the outputs needed by the reader. Record the relevant environment, seed, and computational requirements, particularly for simulations that need an accelerator or substantial runtime.

Documentation builds must not execute notebook cells. Do not change the MkDocs notebook setting to rerun examples during every build. Remove accidental debugging output, private paths, credentials, and unrelated generated files before committing. Keep output useful and reasonably sized so that the notebook can be reviewed and loaded comfortably.

## Preparing the change for review

Read the source diff, tests, docstrings, and rendered guide as parts of one contribution. A new option should use the same name and default everywhere. A claimed invariant should have a corresponding argument or test. A benchmark claim should identify the workload, hardware, dtype, and measurement method.

If a change alters a public contract, explain the compatibility impact and any migration steps in the pull request and relevant guide. For a substantial feature, return to the agreed issue scope when the design changes. The [testing guide](testing.md) describes how to present validation results so reviewers can assess both the implementation and its practical limits.
