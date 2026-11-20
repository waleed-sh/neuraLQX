---
title: Installation
summary: Install neuraLQX with pip and choose the dependencies for your workflow.
---

# Installation

neuraLQX is available on [PyPI](https://pypi.org/project/neuralqx/) and requires **Python 3.11 or newer**. The standard installation includes the libraries needed to run simulations on a CPU. Optional extras add accelerator dependencies, profiling tools, or the tools used to develop the library and build its documentation.

If this is your first installation, start with a fresh Python environment. If you already have an environment for your simulations, activate it and continue with [installing the package](#install-with-pip).

## Prepare an environment

A virtual environment keeps the dependencies for your simulations separate from those used by other projects. On Linux or macOS, create and activate one with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Use a Python interpreter that meets the version requirement when creating the environment. Once it is active, `python -m pip` installs packages for that environment's interpreter. The same approach works with an environment you manage through another tool.

Hardware and operating system support also depend on JAX. Consult the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for its supported platforms, especially when setting up an accelerator or working outside Linux and macOS.

## Install with pip

Install the latest release, or update an existing installation, with:

```bash
python -m pip install --upgrade neuralqx
```

Pip installs neuraLQX together with its dependencies, including JAX, Flax, and Optax. This is the usual starting point for a [CPU simulation](../distributed/cpu.md). You can then [check the installation](#check-the-installation) and continue to the [quick start](quickstart.md).

To install a particular release, specify its version explicitly:

```bash
python -m pip install "neuralqx==2.0.0"
```

Pinning the neuraLQX version is useful when reproducing a calculation or sharing an environment with collaborators. For a complete record of the software used in a study, also keep the dependency versions from that environment.

## Choose optional extras

Extras install additional dependencies alongside the same neuraLQX package. The names in square brackets select the tools you need, and several extras can be combined by separating their names with commas.

### GPU and TPU support

For NVIDIA GPUs on Linux, the `cuda` extra requests JAX's CUDA dependencies:

```bash
python -m pip install --upgrade "neuralqx[cuda]"
```

Follow the [single GPU guide](../distributed/single-gpu.md) for the installation steps and checks needed for your hardware. The [multiple GPU](../distributed/multi-gpu.md) and [multiple node](../distributed/multi-node.md) guides cover setup for larger runs. Those pages are the starting point for choosing a compatible driver and runtime configuration.

For a Linux environment with access to a TPU, the `tpu` extra requests JAX's TPU dependencies:

```bash
python -m pip install --upgrade "neuralqx[tpu]"
```

Use the [JAX TPU installation instructions](https://docs.jax.dev/en/latest/installation.html#google-cloud-tpu) for the corresponding platform setup. Both accelerator extras add their hardware dependencies on Linux.

### NVIDIA profiling tools

The `profile-nvidia` extra adds the Python bindings used for NVIDIA device metrics and NVTX annotations. You can install it together with the CUDA extra when you want to run and profile GPU simulations:

```bash
python -m pip install --upgrade "neuralqx[cuda,profile-nvidia]"
```

See [profiling for scientists](../developers/profiling/for-scientists.md) for examining a simulation, or [traces and backends](../developers/profiling/traces.md) for the profiling integrations. The profiling extra supplies measurement tools, while the CUDA extra supplies the accelerator dependencies.

### Development and documentation tools

The `dev` extra adds testing, formatting, type checking, and packaging tools. The `docs` extra adds MkDocs, the documentation theme, notebook rendering, and API reference generation. If you need both sets of tools, install them together:

```bash
python -m pip install --upgrade "neuralqx[dev,docs]"
```

You can select `neuralqx[dev]` or `neuralqx[docs]` individually when you only need one set. These extras are useful when contributing to the project. The [development environment guide](../developers/contributing/environment.md) covers working on a local checkout and using those tools with the source code.

## Check the installation

Start Python in the environment where you installed neuraLQX, then check the package version and the devices visible to JAX:

```python
import jax
import neuralqx as nqx

print(nqx.__version__)
print(jax.devices())
```

The first line of output identifies your neuraLQX version. The second lists the devices available to the numerical backend. A CPU installation typically reports a device such as `CpuDevice(id=0)`. If you are setting up a GPU, use the device checks in the [single GPU guide](../distributed/single-gpu.md) before starting a larger calculation.

You can also ask pip to check that the installed packages satisfy one another's declared dependencies:

```bash
python -m pip check
```

With the installation ready, the [quick start](quickstart.md) introduces your first calculation. The [core concepts](../concepts/index.md) explain the main components, and the [tutorials](../tutorials/index.md) provide a route into longer examples.
