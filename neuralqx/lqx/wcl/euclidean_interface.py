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

from typing import Optional

from humanize import scientific

from ..abstract_lqx_interface import AbstractLqxInterface
from neuralqx.hilbert import AbstractHilbertInterface
from neuralqx.gauge_groups import AbstractGaugeGroup

from .core import LqxWCL1D
from .core import LqxWCL3D
from .core import LqxWCL4D


class LqxWCL(AbstractLqxInterface):

    def __init__(
        self,
        H: AbstractHilbertInterface,
        gauge_group: AbstractGaugeGroup,
        *,
        lazy_load: bool = True,
        spacetime_dimensions: int,
        computational: bool = True,
    ):
        # create the model
        if spacetime_dimensions == 3:
            if H.gauge_dimensions == 1:
                mn = "U(1) BF-Theory"
                self._model = LqxWCL1D(
                    hilbert=H,
                    graph=H.graph,
                    gauge_group=gauge_group,
                    computational=computational,
                    spacetime_dimensions=spacetime_dimensions,
                    model_name=mn,
                )
            elif H.gauge_dimensions == 3:
                mn = "(2+1)-Euclidean LQG"
                self._model = LqxWCL3D(
                    hilbert=H,
                    graph=H.graph,
                    gauge_group=gauge_group,
                    computational=computational,
                    spacetime_dimensions=spacetime_dimensions,
                    model_name=mn,
                    is_4d=False,
                )
            else:
                raise NotImplementedError
        elif spacetime_dimensions == 4:
            mn = "(3+1)-Abelian LQG"
            self._model = LqxWCL4D(
                hilbert=H,
                graph=H.graph,
                gauge_group=gauge_group,
                computational=computational,
                spacetime_dimensions=spacetime_dimensions,
                model_name=mn,
                is_4d=True,
            )
        else:
            raise NotImplementedError(
                f"There are no WCL models implemented for spacetime dimensions "
                f"`{spacetime_dimensions}`."
            )

        # THEN init the super class
        super().__init__(H, gauge_group, lazy_load=lazy_load, model_name=mn)

    def volume(
        self,
        vertex: int,
        *,
        computational: bool = True,
        jax: bool = True,
    ):
        """
        Returns the volume operator of the underlying wrapped model.

        :param vertex: the vertex the volume operator should act on
        :param computational: if True, the volume operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :return:
        """
        return self.model.volume(vertex, computational=computational, jax=jax)

    def area(
        self,
        surface: list,
        *,
        computational: bool = True,
        jax: bool = True,
    ):
        """
        Returns the area operator of the underlying wrapped model.

        :param surface: the surface the area operator acts on. This should be a list composed of
          edges in their raw representation as provided from the AbstractGraph.edges property
        :param computational: if True, the area operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :return:
        """
        return self.model.area(surface, computational=computational, jax=jax)

    def minimal_loop_holonomy(
        self,
        loop: list,
        *,
        computational: bool = True,
        jax: bool = True,
    ):
        """
        Returns the minimal loop holonomy operator of the underlying wrapped model.

        :param loop: the loop the operator should act on. This should be one of the loops provided
          by the AbstractGraph.dressed_minimal_loops()
        :param computational: if True, the operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :return:
        """
        return self.model.minimal_loop_holonomy(
            loop, computational=computational, jax=jax
        )

    def holonomy(
        self,
        edge: list,
        *,
        computational: bool = True,
        jax: bool = True,
    ):
        """
        Returns the minimal loop holonomy operator of the underlying wrapped model.

        :param edge: the edge the operator should act on. This should be one of the loops provided
          by the AbstractGraph.edges
        :param computational: if True, the operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :return:
        """
        return self.model.holonomy(edge, computational=computational, jax=jax)

    def curvature_constraint(
        self,
        *,
        computational: bool = True,
        jax: bool = True,
    ):
        """
        Returns the curvature (flatness) constraint for the underlying wrapped model.

        :param computational: if True, the operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :return:
        """
        return self.model.curvature_constraint(computational=computational, jax=jax)

    def thiemann_quadratic_constraint(
        self,
        computational: Optional[bool] = None,
        jax: Optional[bool] = True,
        **kwargs,
    ):
        """
        Returns the quadratic constraint implemented using Thiemann's regularised quantum
        Hamilton constraint

        :param computational: if True, the operator is implemented as a computational
          operator
        :param jax: if True, and a computational operator implementation is requested, the operator
          is returned as a JAX computational operator
        :return:
        """

        return self.model.thiemann_quadratic_constraint(
            computational=computational, jax=jax, **kwargs
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"gauge_invariant_model={self.hilbert.is_gauge_invariant}, "
            f"hilbert_dimensions={scientific(self.hilbert_dimensions)}, "
            f"gauge_dimensions={self.gauge_dimensions}, "
            f"model_name={self.model_name}, "
            f"spacetime_dimensions={self.model.spacetime_dimensions}, "
            f"is_computational={self.model.is_computational}"
            f")"
        )
