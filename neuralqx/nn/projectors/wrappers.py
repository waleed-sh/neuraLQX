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


"""Native model wrappers preserving parameters and connected evaluation."""

from dataclasses import dataclass
from functools import wraps
from typing import Any

from flax import linen
from flax import nnx

from neuralqx.utils.errors import NeuralqxTypeError

from .base import Projector
from .utils import connected_evaluator
from .utils import projector_state


class ProjectedNNX(nnx.Module):
    """An NNX wavefunction with a fixed projection of its amplitudes.

    The original module remains available as ``model``. Its parameters and
    their partition metadata are preserved, including shared parameter
    objects. Projector arrays live in a separate non-parameter variable.
    Splitting with ``nnx.Param`` therefore differentiates only the network.
    Both ordinary and connected evaluations use the same projection.
    """

    model: nnx.Module
    """Underlying initialized model, retained without copying."""

    projection: nnx.Variable
    """Fixed projector pytree, included in non-trainable model state."""

    def __init__(self, model, projector):
        """Wrap an initialized NNX model with a Projector instance."""
        if not isinstance(model, nnx.Module) or not isinstance(projector, Projector):
            raise NeuralqxTypeError(
                "ProjectedNNX requires an NNX module and a Projector."
            )
        self.model = model
        self.projection = nnx.Variable(projector_state(projector))

    @property
    def projector(self):
        """Current immutable projector, including its fixed array data."""
        return self.projection.get_value()

    def __call__(self, states, **kwargs):
        """Return projected log amplitudes using the underlying model's options."""
        return self.projector.project(lambda x: self.model(x, **kwargs), states)

    def connected_log_value(self, sources, targets, owners, *, chunk_size=None):
        """Project exact connected values using the underlying model's capability."""
        return self.projector.project_connected(
            self.model,
            connected_evaluator(self.model),
            sources,
            targets,
            owners,
            chunk_size=chunk_size,
        )


class ProjectedLinen(linen.Module):
    """A Linen wavefunction with projector data in a separate collection.

    The base model is a normal Linen submodule. Parameters initialize through
    ``init`` and retain their native metadata. The ``projection`` collection
    stores fixed projector data alongside any other model collections.
    An existing base model's variables need the corresponding submodule
    scope when transferred into this wrapper. Evaluation uses immutable
    collections so row expansions and chunking cannot change model state.
    """

    model: linen.Module
    """Underlying Linen module, also allowed to be a bridged NNX model."""

    projector: Projector
    """Recipe for the projector state initialized in the projection collection."""

    def setup(self):
        """Create the non-trainable projector variable."""
        if not isinstance(self.model, linen.Module) or not isinstance(
            self.projector, Projector
        ):
            raise NeuralqxTypeError(
                "ProjectedLinen requires a Linen module and a Projector."
            )
        self._projection = self.variable(
            "projection", "data", lambda: projector_state(self.projector)
        )

    def __call__(self, states, **kwargs):
        """Return projected log amplitudes with initialized, immutable collections."""
        if self.is_initializing():
            # Linen creates parameters as a side effect. Initialize them before
            # a projector enters a raw JAX map or rematerialized chunk.
            self.model(states.reshape((-1, states.shape[-1]))[:1], **kwargs)
        model, variables = self.model.clone(parent=None), self.model.variables
        return self._projection.value.project(
            lambda x: model.apply(variables, x, **kwargs), states
        )

    def connected_log_value(self, sources, targets, owners, *, chunk_size=None):
        """Retain the wrapped module's exact connected-state evaluator."""
        if self.is_initializing():
            self.model(sources[:1])
        model, variables = self.model.clone(parent=None), self.model.variables

        def evaluate(x):
            return model.apply(variables, x)

        method = getattr(model, "connected_log_value", None)
        if callable(method):

            def connected(source, target, owner, *, chunk_size=None):
                return model.apply(
                    variables,
                    source,
                    target,
                    owner,
                    chunk_size=chunk_size,
                    method=method,
                )

        else:
            connected = connected_evaluator(evaluate)
        return self._projection.value.project_connected(
            evaluate,
            connected,
            sources,
            targets,
            owners,
            chunk_size=chunk_size,
        )


@dataclass(frozen=True, eq=False)
class ProjectedCallable:
    """Projection of an explicit ``apply(parameters_or_variables, states)`` function.

    Call this object like the original apply function. When passed to a
    variational state, its adapter keeps projector arrays in model state
    rather than capturing them as executable constants. Supply the same
    ``init_fn`` and ``expects_variables`` convention as for the base function.
    """

    model: Any
    """Original explicit apply function."""

    projector: Projector
    """Fixed projection of the function's wavefunction amplitudes."""

    def __call__(self, parameters, states, **kwargs):
        """Evaluate the projected function with explicit parameters or variables."""
        return self.projector.project(
            lambda x: self.model(parameters, x, **kwargs), states
        )


def wrap_model(model, projector):
    """Select a wrapper once, before tracing or constructing a variational state."""
    from neuralqx.utils.frameworks.base import AbstractModelFramework

    if isinstance(model, type) or hasattr(model, "_neuralqx_projected_constructor"):

        @wraps(model, updated=())
        def construct(*args, **kwargs):
            return wrap_model(model(*args, **kwargs), projector)

        construct._neuralqx_projected_constructor = (model, projector)
        return construct
    if isinstance(model, nnx.Module):
        return ProjectedNNX(model, projector)
    if isinstance(model, linen.Module):
        return ProjectedLinen(model, projector)
    if isinstance(model, AbstractModelFramework):
        from neuralqx.utils.frameworks.projected import ProjectedFramework

        return ProjectedFramework(model, projector)
    if callable(model):
        return ProjectedCallable(model, projector)
    raise NeuralqxTypeError(
        f"Cannot project unsupported model type {type(model).__name__}."
    )
