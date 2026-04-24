<div align="center">
  <img src="https://raw.githubusercontent.com/waleed-sh/neuraLQX/main/docs/_static/_logo.png" width="420">
</div>

<div align="center">

# neuraLQX
**High-performance variational simulations for canonical Loop Quantum Gravity - built on [NetKet](https://www.github.com/netket/netket) & [JAX](https://github.com/jax-ml/jax).**

[![PyPI](https://img.shields.io/pypi/v/neuraLQX.svg)](https://pypi.org/project/neuraLQX/)
![Python](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fwaleed-sh%2FneuraLQX%2Fmain%2Fpyproject.toml)
[![License](https://img.shields.io/github/license/waleed-sh/neuraLQX.svg)](https://github.com/waleed-sh/neuraLQX/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/neuralqx/badge/?version=latest)](https://neuralqx.readthedocs.io/en/latest/?badge=latest)

</div>

---

neuraLQX is an open-source Python package for variational canonical Loop Quantum Gravity.
It lets you work directly with LQG-native building blocks, graphs, Hilbert spaces, gauge groups, constraints, and projectors,
while leveraging the battle-tested and state of the art variational backend of NetKet.

Under the hood, neuraLQX builds on NetKet and JAX, making fast Monte Carlo methods, automatic differentiation, and scalable optimisation
available in an API that speaks the language of LQG.



## Installation

neuraLQX requires **Python ≥ 3.11**.

### Stable release (PyPI)

```bash
pip install --upgrade neuralqx
```

### From source (editable)

```bash
git clone https://www.github.com/waleed-sh/neuralqx
cd neuralqx
pip install -e .
```

### Optional extras

#### Developer / contributor dependencies

```bash
pip install --upgrade "neuralqx[dev]"
```

#### GPU support (Linux only)

```bash
pip install --upgrade "neuralqx[cuda]"
```

#### MPI support (for versions prior to v1.1.0)

```bash
mpicc --showme:link
pip install --upgrade "neuralqx[mpi]"
```

#### Docs

```bash
pip install --upgrade "neuralqx[docs]"
```

#### Profiling

```bash
pip install --upgrade "neuralqx[profile]"
```

## Getting help & contributing

- **Questions / ideas:** GitHub Discussions  
  https://github.com/waleed-sh/neuraLQX/discussions

- **Bug reports:** GitHub Issues  
  https://github.com/waleed-sh/neuraLQX/issues

- **Contributing:** contributor guide  
  https://neuralqx.readthedocs.io/en/latest/contribute.html


## License

This package is licensed under the [Apache License 2.0](https://github.com/waleed-sh/neuraLQX/blob/main/LICENSE).
