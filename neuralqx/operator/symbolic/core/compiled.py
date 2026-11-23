# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
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


"""Native executable operators produced by symbolic lowering."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.operator.computational import ComputationalOperator
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError
from neuralqx.utils.struct import field


class CompiledOperator(ComputationalOperator):
    """A symbolic kernel integrated with native batching, streaming and algebra.

    ``fn`` accepts a flat configuration and returns padded row connections.
    Kernel functions and their shape metadata are static struct fields.
    Unused slots are replaced by the input configuration so evaluating a
    wavefunction on padding is safe on constrained Hilbert spaces.
    """

    _name: str = field(static=True)
    _fn: Callable = field(static=True)
    _hermitian: bool = field(static=True)
    _dtype_val: np.dtype = field(static=True)
    _max_conn_size: int = field(static=True)
    _diagonal: bool = field(static=True, default=False)
    _support: tuple[tuple[int, ...], ...] | None = field(static=True, default=None)
    _adjoint_fn: Callable | None = field(static=True, default=None)

    def __init__(
        self,
        hilbert,
        *,
        name=None,
        fn=None,
        is_hermitian=False,
        dtype=None,
        max_conn_size=1,
        is_diagonal=False,
        support=None,
        adjoint_factory=None,
        **struct_fields,
    ):
        # Struct.replace reconstructs objects using their declared field names.
        # Public construction keeps the familiar compiler factory keywords.
        values = dict(
            hilbert=hilbert,
            _name=name,
            _fn=fn,
            _hermitian=is_hermitian,
            _dtype_val=dtype,
            _max_conn_size=max_conn_size,
            _diagonal=is_diagonal,
            _support=support,
            _adjoint_fn=adjoint_factory,
        )
        values.update(struct_fields)
        values["_name"] = str(values["_name"])
        values["_dtype_val"] = np.dtype(values["_dtype_val"])
        values["_max_conn_size"] = int(values["_max_conn_size"])
        values["_hermitian"] = bool(values["_hermitian"])
        values["_diagonal"] = bool(values["_diagonal"])
        if values["_support"] is not None:
            values["_support"] = tuple(tuple(sites) for sites in values["_support"])
        self._init_fields(**values)

    def __post_init__(self):
        super().__post_init__()
        if not callable(self._fn):
            raise NeuralqxTypeError(
                "fn must be a JAX-compatible connectivity function."
            )
        if self._max_conn_size < 1:
            raise NeuralqxValueError("max_conn_size must be positive.")

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype_val

    @property
    def is_hermitian(self):
        return self._hermitian

    @property
    def is_diagonal(self):
        return self._diagonal

    @property
    def max_conn_size(self):
        return self._max_conn_size

    def _get_conn_padded_kernel(self, state):
        targets, amplitudes = self._fn(state)
        targets, amplitudes = jnp.asarray(targets), jnp.asarray(
            amplitudes, dtype=self.dtype
        )
        return jnp.where((amplitudes != 0)[..., None], targets, state), amplitudes

    @property
    def adjoint(self):
        if self._adjoint_fn is not None:
            return self._adjoint_fn(self)
        if self.is_hermitian:
            return self
        from .adjoint import adjoint_operator

        return adjoint_operator(self)

    def with_adjoint(self, operator: ComputationalOperator) -> ComputationalOperator:
        """Pair this kernel with an explicitly supplied reverse operator.

        Use this for non-Hermitian rules whose update support would require
        too many local candidates in the generic adjoint. Both operators must
        use the same Hilbert space. The caller supplies the mathematical
        adjoint, including complex conjugation of row matrix elements.
        """
        if not isinstance(operator, ComputationalOperator):
            raise NeuralqxTypeError(
                "The explicit adjoint must be a compiled ComputationalOperator."
            )
        if operator.hilbert != self.hilbert:
            raise NeuralqxValueError(
                "The explicit adjoint must use the same Hilbert space."
            )
        from .adjoint import ExplicitAdjointOperator

        return ExplicitAdjointOperator(self.hilbert, self, operator)


_DYNAMIC_CLASS_CACHE: dict[tuple[type, str], type] = {}


def create_compiled_operator(
    hilbert: Any,
    *,
    name: str,
    fn: Callable,
    is_hermitian: bool,
    dtype: Any,
    max_conn_size: int,
    operator_type: type = ComputationalOperator,
    connection_method: str = "_get_conn_padded_kernel",
    operator_factory: Callable | None = None,
    is_diagonal: bool = False,
    support: tuple[tuple[int, ...], ...] | None = None,
) -> Any:
    """Construct a native operator or a developer-registered target.

    A custom factory receives the Hilbert space and the standard kernel
    construction keywords. Native target subclasses can inherit the compiled
    struct implementation. Other target families use a small callable adapter.
    """
    if not isinstance(connection_method, str) or not connection_method.strip():
        raise NeuralqxValueError("connection_method must be a non-empty string.")
    method = connection_method.strip()
    if not method.isidentifier():
        raise NeuralqxValueError("connection_method must be a valid Python identifier.")
    if not isinstance(operator_type, type):
        raise NeuralqxTypeError("operator_type must be a class.")
    kwargs = dict(
        name=name,
        fn=fn,
        is_hermitian=is_hermitian,
        dtype=dtype,
        max_conn_size=max_conn_size,
    )
    if operator_factory is not None:
        return operator_factory(hilbert, **kwargs)
    if operator_type in (ComputationalOperator, CompiledOperator):
        if method != "_get_conn_padded_kernel":
            raise NeuralqxValueError("Native targets must use _get_conn_padded_kernel.")
        return CompiledOperator(
            hilbert, **kwargs, is_diagonal=is_diagonal, support=support
        )
    if issubclass(operator_type, CompiledOperator):
        return operator_type(
            hilbert, **kwargs, is_diagonal=is_diagonal, support=support
        )
    key = operator_type, method
    cls = _DYNAMIC_CLASS_CACHE.get(key)
    if cls is None:
        if issubclass(operator_type, ComputationalOperator):
            if method != "_get_conn_padded_kernel":
                raise NeuralqxValueError(
                    "Native targets must use _get_conn_padded_kernel."
                )
            cls = type(
                "Compiled_" + operator_type.__name__,
                (CompiledOperator, operator_type),
                {"__module__": __name__, "__init__": CompiledOperator.__init__},
            )
        else:
            cls = type(
                "Compiled_" + operator_type.__name__,
                (_CallableAdapter, operator_type),
                {"__module__": __name__, method: lambda self, x: self._fn(x)},
            )
            jax.tree_util.register_pytree_node_class(cls)
        _DYNAMIC_CLASS_CACHE[key] = cls
    try:
        if issubclass(cls, CompiledOperator):
            return cls(hilbert, **kwargs, is_diagonal=is_diagonal, support=support)
        return cls(hilbert, **kwargs)
    except TypeError as exc:
        raise NeuralqxTypeError(
            f"Failed to instantiate compiled operator target {operator_type!r}: {exc}. "
            "Register a factory for targets with a different constructor."
        ) from exc


class _CallableAdapter:
    """Developer extension for targets outside the native struct hierarchy."""

    def __init__(self, hilbert, *, name, fn, is_hermitian, dtype, max_conn_size):
        super().__init__(hilbert)
        self._name, self._fn = name, fn
        self._hermitian, self._dtype_val = is_hermitian, np.dtype(dtype)
        self._max_conn_size = int(max_conn_size)

    name = property(lambda self: self._name)
    dtype = property(lambda self: self._dtype_val)
    is_hermitian = property(lambda self: self._hermitian)
    max_conn_size = property(lambda self: self._max_conn_size)

    def tree_flatten(self):
        return (), (
            self.hilbert,
            self._name,
            self._fn,
            self._hermitian,
            self._dtype_val,
            self._max_conn_size,
        )

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        hilbert, name, fn, hermitian, dtype, capacity = auxiliary
        return cls(
            hilbert,
            name=name,
            fn=fn,
            is_hermitian=hermitian,
            dtype=dtype,
            max_conn_size=capacity,
        )


__all__ = ["CompiledOperator", "create_compiled_operator"]
