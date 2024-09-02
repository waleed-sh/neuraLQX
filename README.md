<div align="center">
<img src="neuralqx/utils/base/logo_WIP.png" alt="logo" width="600"></img>
</div>

<hr>




[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Static Badge](https://img.shields.io/badge/linting-pylint-blue)](https://github.com/pylint-dev/pylint)

## Description

neuraLQX is an open-source Python package designed for high-performance computing, tailored specifically for simulating 
canonical loop quantum gravity systems. Built on top of [NetKet](https://github.com/netket/netket.git), neuraLQX offers a complete and user-friendly 
environment for researchers and developers in the field.

**Note:** The package is currently under development and will be released at a later date. Once available, it can be 
installed via pip. Check back here for the release data later.

**Release Date:** TBA

### Key Features

**Modular and Customizable Design:**
- **Flexible Quantum Models:** Easily construct custom quantum models, including various geometric and quantum 
operators.
- **Custom Graphs:** Design and implement custom graphs to fit specific needs.
- **Quantum Constraints:** Ready-to-use quantum constraints such as Gauß and Euclidean Hamilton constraints, with the 
ability to solve these using neural networks (see [NQS](https://www.science.org/doi/10.1126/science.aag2302)).
- **Pre-trained Neural Networks:** Includes pre-trained novel neural networks capable of solving constraints across 
different models, saving time and effort.

**Advanced State Characterization and Analysis:**
- **State Characterization Tools:** Analyze states using tools like coloring operators, N-point functions and Rényi
entropy.
- **Multiple Hilbert Spaces:** Support for different Hilbert spaces (e.g. gauge-invariant subspaces).
- **Gauge Groups and Spacetime Dimensions:** Choose from different gauge groups (e.g., U(1)<sup>3</sup>, SU(2)) and 
spacetime dimensions, providing flexibility for diverse simulations.
- **Standard Models of Interest:** Implementations of some commonly used models, such as the torus universe, are 
included for convenience.

**High-Performance Computing (HPC) Friendly and Ease of Use:**
- **Parallelization Support:** Optimized for HPC environments, allowing for efficient parallel computations.
- **Seamless Installation and Setup:** Modular designed enabling the bypass technical difficulties, allowing you to 
focus on simulating systems rather than dealing with implementation issues.

<hr>

## Examples

The package is still under development. However, its functionality has already been put to test. You can read about 
some systems explored using neuraLQX in this [paper](https://arxiv.org/abs/2405.00661) and this 
[paper](https://arxiv.org/abs/2402.10622) too.

You can also see some more in depth examples which showcase the intended usage of the package in our 
[tutorials page](/docs/Tutorials/). 

<hr>

## __License__

This package will be available under the Apache 2.0 license. You can read about the Apache 2.0 license [here](/LICENSE).