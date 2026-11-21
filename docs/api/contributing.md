---
title: Contributing to neuraLQX
---

# Contributing to neuraLQX

Contributions to neuraLQX can improve the scientific implementation, make an example easier to follow, or establish a numerical result more carefully through tests. Documentation corrections and focused regression tests are valuable contributions in their own right. You do not need to add a new model or algorithm to help the project.

Development takes place in the [neuraLQX GitHub repository](https://github.com/waleed-sh/neuraLQX). This page describes the path from a proposal to a pull request. The developer guides explain the [development environment](../developers/contributing/environment.md), [source code and documentation conventions](../developers/contributing/index.md), and [testing expectations](../developers/contributing/testing.md).

## Choose and discuss the change

Search the [issue tracker](https://github.com/waleed-sh/neuraLQX/issues) before starting, since someone may already be investigating the same problem. A useful bug report includes a small reproducer, what happened, what you expected, and the package and hardware configuration needed to reproduce the behaviour. For a numerical discrepancy, explain the reference calculation and the tolerances involved.

Small corrections, clear documentation improvements, and focused bug fixes can proceed directly to a pull request. If an existing issue describes the work, leave a short comment explaining that you intend to address it so that other contributors can coordinate.

**Discuss substantial features in a GitHub issue before implementing them.** This includes new public interfaces, major algorithms, additional dependencies, and changes that span several subsystems. Explain the scientific use case, the proposed interface, alternatives considered, and how the result could be tested. Seek agreement on the scope and an implementation direction before producing a large patch.

The amount of code that can be generated quickly does not determine what can be reviewed responsibly. In particular, a contribution expected to add thousands of lines should be discussed and divided into coherent stages. This applies equally to manually written and AI-assisted work. Avoid combining a new algorithm, unrelated cleanup, and a broad refactor in one pull request.

## Fork the repository and create a branch

Use GitHub's **Fork** button on the [public repository](https://github.com/waleed-sh/neuraLQX). Clone your fork, add the original repository as `upstream`, and create a branch from its current `main`. Replace `YOUR_USERNAME` in the clone command with your GitHub username, and choose a branch name that describes your change.

```sh
git clone https://github.com/YOUR_USERNAME/neuraLQX.git
cd neuraLQX
git remote add upstream https://github.com/waleed-sh/neuraLQX.git
git fetch upstream
git switch -c improve-sampler-guide upstream/main
```

Your fork is the `origin` remote. The project repository is `upstream`. Develop on your own branch and target the project's `main` branch when opening the pull request.

Next, prepare a [development environment](../developers/contributing/environment.md). Source and test changes use an editable installation with development dependencies. A documentation-only change can use the smaller documentation environment, although examples that import neuraLQX must still be checked in an environment containing the package.

## Develop the implementation, tests, and explanation together

A new feature needs tests that establish its behaviour. A bug fix needs a regression test that exposes the original failure where a practical reproducer is available. For scientific code, include the mathematical or numerical reference behind the test. A successful import or a finite output alone does not establish that an estimator, constraint, or operator is correct.

Update the relevant documentation when users need to know about a new interface, changed behaviour, new option, or limitation. Public objects need docstrings describing their contracts, including shapes, conventions, and assumptions that affect a calculation. Examples should run with the documented imports and setup. The [source and documentation guide](../developers/contributing/index.md) shows the expected docstring format and explains how API pages are generated.

Run the relevant checks while developing, and broaden them when a change affects shared behaviour. Record the exact commands and results in the pull request, including any failures or hardware paths you could not exercise. The [testing guide](../developers/contributing/testing.md) covers numerical references, stochastic tests, JAX transformations, and documentation verification.

## Using AI tools

You may use AI tools to help write, revise, explain, or test a contribution. **Disclose that use in the pull request**, including which tool was used and what it helped produce. A short statement is sufficient, for example, “I used an AI coding assistant to draft the test parametrisation and edit the accompanying explanation. I reviewed the changes and ran the commands listed below.” Describe your actual use rather than copying this statement unchanged.

You remain responsible for everything you submit, including generated code, tests, documentation, equations, references, and dependencies. Read the complete diff, understand the implementation, and be able to explain its assumptions and limitations. Verify the tests yourself and report their actual results. An AI-generated test that repeats the implementation's mistake is not independent validation.

Check the origin and suitability of incorporated material, and respect the project's [license](https://github.com/waleed-sh/neuraLQX/blob/main/LICENSE) and any applicable third-party terms. Do not submit code or explanations you cannot evaluate, or references you have not checked. AI assistance does not reduce the need for a focused scope, an issue discussion for substantial work, or careful responses during review.

## Prepare and open the pull request

Inspect both the diff and the working tree before committing. Include the source, tests, and documentation belonging to the change. Exclude virtual environments, generated sites, caches, local results, and unrelated formatting changes.

```sh
git diff
git status --short
```

Stage the intended files with `git add`. For the example documentation branch above, a change limited to the sampler overview could be committed and pushed as follows. Adjust the file selection and commit message to the work you actually performed.

```sh
git add docs/concepts/samplers/index.md
git commit -m "Clarify the sampler overview"
git push --set-upstream origin improve-sampler-guide
```

Open a pull request from that branch in your fork to `main` in `waleed-sh/neuraLQX`. Use a title that states the change. In the description, explain the problem, the resulting behaviour, the main design choices, and how you verified them. Link the related issue and describe any compatibility changes. Include the AI disclosure when applicable.

A draft pull request is useful when you want feedback on an agreed implementation before it is complete. State what remains unfinished. When ready for review, the code, tests, and relevant documentation should form a coherent change that a reviewer can assess without reconstructing the conversation that led to it.

Respond to review comments with explanations and focused updates. Push further commits to the same branch to update the pull request. If `main` has moved and integration changes are needed, fetch `upstream` and resolve them on your contribution branch, then rerun the affected checks. Do not assume that a successful documentation preview or one test command covers every consequence of the update.
