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


"""Projection adapter for explicit functions and user-defined frameworks."""

from typing import Any

from neuralqx.utils.errors import NeuralqxNotImplementedError
from neuralqx.utils.struct import field

from .base import AbstractModelFramework


class ProjectedFramework(AbstractModelFramework):
    """Keep fixed projector arrays separate from a framework's trainable state.

    Native Linen and NNX models use native module wrappers. This adapter
    covers explicit apply functions and custom framework implementations.
    Their parameter tree is unchanged. Base model state and the projector
    are stored under separate keys in the non-trainable state.
    """

    base: AbstractModelFramework = field(pytree=False, compare=False)
    """Underlying framework, including its exact connection evaluator."""

    projector: Any = field(pytree=False, compare=False)
    """Initial projector recipe, used only when constructing variable state."""

    @property
    def name(self):
        """Framework name with the projection wrapper made explicit."""
        return f"projected.{self.base.name}"

    @property
    def has_projector(self):
        """Projection can introduce exact amplitude nodes."""
        return True

    def __hash__(self):
        return hash((type(self), self.base.jit_cache_key(), id(self.projector)))

    def init(self, key, input_shape, *, dtype):
        """Initialize the base variables and attach fixed projection data."""
        variables = self.base.init(key, input_shape, dtype=dtype)
        params, state = self.base.split_variables(variables)
        return self.merge_variables(
            params, {"base": state, "projector": self.projector}
        )

    def split_variables(self, variables):
        """Separate trainable base parameters from model and projector state."""
        if "base" not in variables or "projector" not in variables:
            from neuralqx.nn.projectors.utils import projector_state

            params, state = self.base.split_variables(variables)
            return params, {"base": state, "projector": projector_state(self.projector)}
        params, state = self.base.split_variables(variables["base"])
        return params, {"base": state, "projector": variables["projector"]}

    def merge_variables(self, parameters, model_state=None):
        """Restore the base variable layout without changing parameter metadata."""
        from neuralqx.nn.projectors.utils import projector_state

        state = {} if model_state is None else model_state
        if "projector" not in state:
            state = {"base": state, "projector": self.projector}
        return {
            "base": self.base.merge_variables(parameters, state.get("base")),
            "projector": projector_state(state.get("projector", self.projector)),
        }

    def apply_variables(
        self, variables, states, *, mutable=False, training_kwargs=None
    ):
        """Project a pure base evaluation using runtime projector arrays.

        Updating mutable collections is ambiguous for a projection that
        evaluates several configurations. Freeze them for wavefunction
        evaluation and update model state explicitly outside this call.
        """
        if mutable is not False and mutable is not None:
            raise NeuralqxNotImplementedError(
                "Projected framework evaluation requires immutable collections."
            )

        def evaluate(x):
            return self.base.apply_variables(
                variables["base"], x, training_kwargs=training_kwargs
            )

        return variables["projector"].project(evaluate, states)

    def apply_connected(
        self,
        variables,
        sources,
        targets,
        owners,
        *,
        chunk_size=None,
        training_kwargs=None,
    ):
        """Delegate exact connection reuse before applying the projection."""

        def evaluate(x):
            return self.base.apply_variables(
                variables["base"], x, training_kwargs=training_kwargs
            )

        def connected(source, target, owner, *, chunk_size=None):
            return self.base.apply_connected(
                variables["base"],
                source,
                target,
                owner,
                chunk_size=chunk_size,
                training_kwargs=training_kwargs,
            )

        return variables["projector"].project_connected(
            evaluate,
            connected,
            sources,
            targets,
            owners,
            chunk_size=chunk_size,
        )
