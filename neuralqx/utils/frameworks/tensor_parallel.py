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


"""Framework composition that preserves tensor layouts at every model boundary."""

from typing import Any

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from neuralqx.jax.tensor_parallel import tensor_axis_name
from neuralqx.jax.tensor_parallel import tensor_mesh
from neuralqx.jax.tensor_parallel import tensor_partition_count
from neuralqx.nn.parallel.plan import place_parameter
from neuralqx.nn.parallel.plan import plan_parameters
from neuralqx.nn.parallel.rules import model_rules
from neuralqx.utils.struct import field

from .base import AbstractModelFramework


class TensorParallelFramework(AbstractModelFramework):
    """Compose a framework with immutable, compiler-managed tensor layouts.

    Manual sample/operator dispatch retains complete tensor shapes inside this
    adapter. Constraints apply only to the automatic tensor axis. JAX therefore
    differentiates the same global function for all combinations of work axes.
    Connected-state reuse shares exactly the same parameter plan.
    """

    base: Any = field(pytree=False, compare=False)
    rules: tuple = field(pytree=False, default=())
    generic: bool = field(pytree=False, default=False)
    required_patterns: tuple = field(pytree=False, default=())

    @property
    def name(self):
        return self.base.name

    @property
    def has_projector(self):
        return self.base.has_projector

    def __hash__(self):
        return hash(
            (
                type(self),
                self.base.jit_cache_key(),
                self.rules,
                self.generic,
                self.required_patterns,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, TensorParallelFramework):
            return NotImplemented
        return (
            self.base.jit_cache_key() == other.base.jit_cache_key()
            and self.rules == other.rules
            and self.generic == other.generic
            and self.required_patterns == other.required_patterns
        )

    def partition_plan(self, parameters):
        """Return the validated layout and replication decisions for parameters."""
        return plan_parameters(
            parameters,
            self.rules,
            generic=self.generic,
            required_patterns=self.required_patterns,
        )

    def split_variables(self, variables):
        return self.base.split_variables(variables)

    def merge_variables(self, parameters, model_state=None):
        return self.base.merge_variables(parameters, model_state)

    def init(self, key, input_shape, *, dtype):
        # Shape tracing avoids allocating a full model before its layout is
        # known. Out-shardings partition random initialization itself.
        def initialize(key):
            return self.base.init(key, input_shape, dtype=dtype)

        abstract = jax.eval_shape(initialize, key)
        parameters, state = self.split_variables(abstract)
        plan = self.partition_plan(parameters)
        replicated = NamedSharding(tensor_mesh(), P())
        layouts = self.merge_variables(
            jax.tree.map(lambda spec: NamedSharding(tensor_mesh(), spec), plan.specs),
            jax.tree.map(lambda _: replicated, state),
        )
        return jax.jit(initialize, out_shardings=layouts)(key)

    def place_variables(self, variables):
        parameters, state = self.split_variables(variables)
        parameters = self.partition_plan(parameters).place(parameters)
        replicated = NamedSharding(tensor_mesh(), P())
        state = jax.tree.map(
            lambda x: (
                place_parameter(x, replicated)
                if isinstance(x, (jax.Array, jax.ShapeDtypeStruct))
                else x
            ),
            state,
        )
        return self.merge_variables(parameters, state)

    def _constrain(self, variables):
        parameters, state = self.split_variables(variables)
        return self.merge_variables(
            self.partition_plan(parameters).constrain(parameters), state
        )

    def apply_variables(
        self, variables, states, *, mutable=False, training_kwargs=None
    ):
        return self.base.apply_variables(
            self._constrain(variables),
            states,
            mutable=mutable,
            training_kwargs=training_kwargs,
        )

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
        return self.base.apply_connected(
            self._constrain(variables),
            sources,
            targets,
            owners,
            chunk_size=chunk_size,
            training_kwargs=training_kwargs,
        )


def wrap_tensor_parallel(framework, variables, model=None, explicit_rules=()):
    from flax import nnx
    from flax.nnx.bridge import ToLinen

    if isinstance(model, ToLinen):
        # Native Linen facades execute the same NNX architecture. Recover its
        # rules from shapes alone, retaining paired contractions in both APIs.
        constructor = model.nnx_class
        kwargs = dict(model.kwargs)
        args = model.args
        skip_rng = model.skip_rng

        def construct():
            options = dict(kwargs)
            if not skip_rng:
                options["rngs"] = nnx.Rngs(0)
            return constructor(*args, **options)

        with nnx.use_eager_sharding(False):
            model = nnx.eval_shape(construct)
    rules = {}
    generic = not isinstance(model, nnx.Module)
    if not generic:
        rules = {
            "/".join(map(str, path)): spec
            for path, spec in model_rules(
                model, tensor_axis_name(), tensor_partition_count()
            ).items()
        }
    rules.update(dict(explicit_rules))
    wrapped = TensorParallelFramework(
        framework,
        tuple(rules.items()),
        generic=generic,
        required_patterns=tuple(dict(explicit_rules)),
    )
    return wrapped, None if variables is None else wrapped.place_variables(variables)


__all__ = ["TensorParallelFramework"]
