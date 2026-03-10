# Copyright (c) 2026 The neuraLQX Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import FrozenSet
from typing import Dict


class _Citation:
    """
    Immutable citation selector for neuraLQX.

    Attribute chaining accumulates citations:
        neuralqx.cite
        neuralqx.cite.netket
        neuralqx.cite.netket.mpi
        neuralqx.cite.netket.mpi.jax.flax
        (or any combination of the above)
    """

    __slots__ = (
        "netket",
        "mpi",
        "jax",
        "flax",
        "_keys",
    )

    _ENTRIES: Dict[str, str] = {
        "neuralqx": r"""
@software{neuralqx2026github,
  author = {Sherif, Waleed},
  title = {{neuraLQX}: a high-performance simulations toolkit for loop quantum gravity},
  url = {http://github.com/waleed-sh/neuraLQX},
  version = {1.1.0},
  year = {2026},
}
""".strip(),
        "netket2": r"""
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
""".strip(),
        "netket3": r"""
@article{netket3:2022,
    title={NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems},
    author={Vicentini, Filippo and Hofmann, Damian and Szabó, Attila and Wu, Dian and Roth, Christopher and Giuliani, Clemens and Pescia, Gabriel and Nys, Jannes and Vargas-Calderón, Vladimir and Astrakhantsev, Nikita and Carleo, Giuseppe},
    journal={SciPost Phys. Codebases},
    pages={7},
    year={2022},
    doi={10.21468/SciPostPhysCodeb.7}
}
""".strip(),
        "mpi": r"""
@article{mpi4jax:2021,
    title={mpi4jax: Zero-copy MPI communication of JAX arrays},
    author={Häfner, Dion and Vicentini, Filippo},
    journal={Journal of Open Source Software},
    volume={6},
    number={65},
    pages={3419},
    year={2021},
    doi={10.21105/joss.03419}
}
""".strip(),
        "jax": r"""
@misc{jax2018github,
  title = {{{JAX}}: Composable Transformations of {{Python}}+{{NumPy}} Programs},
  author = {Bradbury, James and Frostig, Roy and Hawkins, Peter and Johnson, Matthew James and Leary, Chris and Maclaurin, Dougal and Necula, George and Paszke, Adam and VanderPlas, Jake and Wanderman-Milne, Skye and Zhang, Qiao},
  year = {2018}
}
""".strip(),
        "flax": r"""
@misc{flax2020github,
  title = {Flax: A Neural Network Library and Ecosystem for JAX},
  author = {Heek, Jonathan and Levskaya, Anselm and Oliver, Avital and Ritter, Marvin and Rondepierre, Bertrand and Steiner, Andreas and van Zee, Marc},
  year = {2024}
}
""".strip(),
    }

    _GROUPS: Dict[str, FrozenSet[str]] = {
        "netket": frozenset({"netket2", "netket3"}),
        "mpi": frozenset({"mpi"}),
        "jax": frozenset({"jax"}),
        "flax": frozenset({"flax"}),
        "all": frozenset({"neuralqx", "netket2", "netket3", "mpi", "jax", "flax"}),
    }

    def __init__(self, keys: FrozenSet[str] | None = None):
        # always include neuraLQX by default
        self._keys: FrozenSet[str] = keys or frozenset({"neuralqx"})

    def __getattr__(self, name: str) -> "_Citation":
        """Attribute chaining adds citation groups."""
        if name not in self._GROUPS:
            raise AttributeError(f"No such citation group: '{name}'")

        new_keys = self._keys | self._GROUPS[name]
        return _Citation(new_keys)

    def __repr__(self) -> str:
        """Render BibTeX entries in a stable, deterministic order."""
        ordered = [self._ENTRIES[k] for k in self._ENTRIES if k in self._keys]
        return "\n\n".join(ordered)

    def __str__(self) -> str:
        return self.__repr__()


cite = _Citation()
