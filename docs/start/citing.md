---
title: Citing neuraLQX
summary: Acknowledge neuraLQX, its foundations, and the software used in your calculations.
---

# Citing neuraLQX

If you use neuraLQX in a paper, thesis, or other research output, please cite the software and acknowledge the projects that helped make it possible. Citations give readers a way to find the implementation behind your results and recognise the research and maintenance that support scientific software.

Please include the neuraLQX software reference and both NetKet publications below. JAX and Flax provide the numerical and neural network foundations of the current library, and their references are included here as well. You can expand each section to copy its BibTeX, or [generate the references from Python](#generate-citations-from-python).

## Cite neuraLQX

Use the following software reference for neuraLQX 2.0.0. Record the version used for your calculations so that readers can connect your results to the corresponding implementation. You can inspect the installed version with `nqx.__version__`, as shown in the [installation check](installation.md#check-the-installation).

??? quote "neuraLQX 2.0.0 · BibTeX"

    ```bibtex
    @software{neuralqx2026github,
      author = {Sherif, Waleed},
      title = {{neuraLQX}: a high-performance simulations toolkit for variational loop quantum gravity},
      url = {http://github.com/waleed-sh/neuraLQX},
      version = {2.0.0},
      year = {2026},
    }
    ```

## Acknowledge NetKet

neuraLQX began as a project built on [NetKet](https://netket.readthedocs.io/en/latest/), and earlier versions relied on its simulation infrastructure. Version 2.0.0 provides its own core infrastructure and is now largely independent of NetKet. Its API nevertheless remains closely aligned with NetKet's approach and draws inspiration from its design.

We therefore continue to ask users to cite **both NetKet publications**. This recognises NetKet's role in the origins and development of neuraLQX, including calculations performed with the current independent infrastructure. It also acknowledges the ideas and conventions that continue to shape the library.

The two references are [NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems](https://doi.org/10.21468/SciPostPhysCodeb.7) and [NetKet: A machine learning toolkit for many-body quantum systems](https://doi.org/10.1016/j.softx.2019.100311). NetKet's own [citation guide](https://netket.readthedocs.io/en/latest/cite.html) provides further context.

??? quote "NetKet publications · BibTeX"

    ```bibtex
    @article{netket2:2019,
        title={NetKet: A machine learning toolkit for many-body quantum systems},
        author={Carleo, Giuseppe and Choo, Kenny and Hofmann, Damian and Smith, James ET and Westerhout, Tom and Alet, Fabien and Davis, Emily J and Efthymiou, Stavros and Glasser, Ivan and Lin, Sheng-Hsuan and Mauri, Marta and Mazzola, Guglielmo and Pereira, Christian B and Vicentini, Filippo},
        journal={SoftwareX},
        volume={10},
        pages={100311},
        year={2019},
        publisher={Elsevier},
        doi={10.1016/j.softx.2019.100311}
    }

    @article{netket3:2022,
        title={NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems},
        author={Vicentini, Filippo and Hofmann, Damian and Szabó, Attila and Wu, Dian and Roth, Christopher and Giuliani, Clemens and Pescia, Gabriel and Nys, Jannes and Vargas-Calderón, Vladimir and Astrakhantsev, Nikita and Carleo, Giuseppe},
        journal={SciPost Phys. Codebases},
        pages={7},
        year={2022},
        doi={10.21468/SciPostPhysCodeb.7}
    }
    ```

## Cite the numerical libraries

[JAX](https://docs.jax.dev/en/latest/) provides array computation, compilation, automatic differentiation, and the foundations for [parallel simulations](../distributed/jax-parallelism.md). Please acknowledge it when reporting calculations performed with neuraLQX.

??? quote "JAX · BibTeX"

    ```bibtex
    @misc{jax2018github,
      title = {{{JAX}}: Composable Transformations of {{Python}}+{{NumPy}} Programs},
      author = {Bradbury, James and Frostig, Roy and Hawkins, Peter and Johnson, Matthew James and Leary, Chris and Maclaurin, Dougal and Necula, George and Paszke, Adam and VanderPlas, Jake and Wanderman-Milne, Skye and Zhang, Qiao},
      year = {2018}
    }
    ```

[Flax](https://flax.readthedocs.io/en/latest/) supplies the neural network framework used by neuraLQX's built-in models. Include its reference when your calculations use those models or other Flax networks, as discussed in the [neural network guide](../concepts/networks/index.md).

??? quote "Flax · BibTeX"

    ```bibtex
    @misc{flax2020github,
      title = {Flax: A Neural Network Library and Ecosystem for JAX},
      author = {Heek, Jonathan and Levskaya, Anselm and Oliver, Avital and Ritter, Marvin and Rondepierre, Bertrand and Steiner, Andreas and van Zee, Marc},
      year = {2024}
    }
    ```

## Generate citations from Python

The same references are available through `nqx.cite`. It is a citation selector that you can print directly. With no additional selection, it produces the neuraLQX software entry:

```python
import neuralqx as nqx

print(nqx.cite)
```

Select additional projects by chaining their names. For example, this produces the neuraLQX reference together with both NetKet publications:

```python
print(nqx.cite.netket)
```

To include JAX and Flax as well, use:

```python
print(nqx.cite.netket.jax.flax)
```

Each selection retains the neuraLQX entry. Repeated selections are included once, and the references are returned in a consistent order. `nqx.cite.all` is a shorthand for all five software references currently provided by the selector. These entries are bundled with neuraLQX, so generating the NetKet references does not require a NetKet installation.

You can save the selected entries to a BibTeX file for your manuscript:

```python
from pathlib import Path

references = str(nqx.cite.netket.jax.flax)
Path("references.bib").write_text(references + "\n", encoding="utf-8")
```

This writes `references.bib` in your current working directory. Choose a different filename if you already maintain a bibliography there. The selector provides software references. Add the relevant scientific publications for the physical models and algorithms you use, such as [stochastic reconfiguration](../concepts/optimisers/sr.md), when preparing your bibliography.

## Acknowledgement in a manuscript

An acknowledgement can state which software performed the calculations and how the related projects contributed. For a calculation using neuraLQX's built-in Flax models, you could adapt the following LaTeX text to your work:

```latex
The simulations presented in this work were performed using
neuraLQX~\cite{neuralqx2026github}, a high performance simulations
toolkit for variational canonical loop quantum gravity.
The design and API of neuraLQX are inspired by
NetKet~\cite{netket2:2019,netket3:2022}, on which earlier versions
of the library were based. Numerical computation and automatic
differentiation were provided by JAX~\cite{jax2018github}, and
neural network models were implemented using Flax~\cite{flax2020github}.
```

Include the software version and any details needed to reproduce your calculation in the methods section or accompanying code. The [public API reference](../api/public/index.md) provides the definitions of the interfaces used in your simulation.
