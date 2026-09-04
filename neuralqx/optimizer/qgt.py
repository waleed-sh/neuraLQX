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


"""Quantum geometric tensor implementations."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import jax
from neuralqx.jax.differentiation import real_coordinate_size
from neuralqx.jax.differentiation import resolve_differentiation_mode
from neuralqx.jax.differentiation import split_tree_to_real
from neuralqx.jax.streaming import apply_batched
from neuralqx.jax.tree import tree_axpy
from neuralqx.jax.tree import tree_cast_like
from neuralqx.jax.tree import tree_size
from neuralqx.utils.frameworks import AbstractModelFramework
from neuralqx.utils.struct import field
from neuralqx.vqs.estimators.utils import flatten_samples

from .linear_operator import LinearOperator


class QGTOnTheFly(LinearOperator):
    """Matrix-free quantum geometric tensor from VQS samples.

    The implementation applies the centered Jacobian Gram matrix without
    materialising the Jacobian:

    ``S v = J^H (J v - mean(J v)) / n + diag_shift * v``.

    Samples keep their native neuraLQX layout externally and are flattened only
    inside the matvec. This preserves compatibility with sample sharding and
    chunked model evaluation.
    """

    framework: AbstractModelFramework = field(pytree=False, compare=False)
    parameters: Any
    model_state: Any = field(default=None)
    samples: Any = field(default=None)
    sample_weights: Any = field(default=None)
    hilbert_size: int = field(static=True, default=0)
    chunk_size: int | None = field(static=True, default=None)
    training_kwargs: tuple[tuple[str, Any], ...] = field(static=True, default=())
    holomorphic: bool | None = field(static=True, default=None)
    mode: str = field(static=True, default="auto")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "training_kwargs",
            _hashable_training_kwargs(self.training_kwargs),
        )
        if self.samples is None:
            raise ValueError("QGTOnTheFly requires samples.")
        if self.hilbert_size <= 0:
            object.__setattr__(
                self, "hilbert_size", int(jnp.asarray(self.samples).shape[-1])
            )
        if self.mode not in {"real", "complex", "holomorphic"}:
            raise ValueError(
                "QGTOnTheFly mode must be 'real', 'complex', or "
                f"'holomorphic', got {self.mode!r}."
            )

        if self.sample_weights is not None:
            expected = jnp.asarray(self.samples).shape[:-1]
            if jnp.asarray(self.sample_weights).shape != expected:
                raise ValueError(
                    "sample_weights must match samples.shape[:-1]; "
                    f"got {jnp.asarray(self.sample_weights).shape} and expected {expected}."
                )
            if not bool(jnp.all(jnp.isfinite(jnp.asarray(self.sample_weights)))):
                raise ValueError("sample_weights must be finite.")
            total_weight = jnp.sum(jnp.asarray(self.sample_weights))
            if not bool(jnp.isfinite(total_weight)) or float(total_weight) == 0.0:
                raise ValueError("sample_weights must have a finite nonzero sum.")

    @classmethod
    def from_vstate(
        cls,
        vstate: Any,
        *,
        samples: Any | None = None,
        sample_weights: Any | None = None,
        chunk_size: int | None = None,
        holomorphic: bool | None = None,
        mode: str | None = None,
        diag_shift: Any = 0.0,
        diag_scale: Any | None = None,
    ) -> QGTOnTheFly:
        """Build an on-the-fly QGT from a variational state's current samples."""
        if diag_scale is not None:
            raise NotImplementedError("QGTOnTheFly does not support diag_scale yet.")
        samples = vstate.samples if samples is None else samples
        flat_samples, _ = flatten_samples(samples, vstate.hilbert.size)

        def logpsi(parameters: Any) -> jax.Array:
            variables = vstate.merge_variables_kernel(parameters, vstate.model_state)
            return apply_batched(
                vstate.apply_variables_kernel,
                variables,
                flat_samples,
                chunk_size=vstate.chunk_size if chunk_size is None else chunk_size,
            )

        resolved_mode = resolve_differentiation_mode(
            logpsi,
            vstate.parameters,
            mode=mode,
            holomorphic=holomorphic,
        )
        return cls(
            diag_shift=diag_shift,
            framework=vstate.framework,
            parameters=vstate.parameters,
            model_state=vstate.model_state,
            samples=samples,
            sample_weights=sample_weights,
            hilbert_size=int(vstate.hilbert.size),
            chunk_size=vstate.chunk_size if chunk_size is None else chunk_size,
            training_kwargs=_hashable_training_kwargs(
                getattr(vstate, "training_kwargs", {})
            ),
            holomorphic=holomorphic,
            mode=resolved_mode,
        )

    @property
    def n_parameters(self) -> int:
        """Number of scalar parameter entries."""
        if self.mode == "holomorphic":
            return tree_size(self.parameters)
        return real_coordinate_size(self.parameters)

    def __matmul__(self, vector: Any) -> Any:
        if hasattr(vector, "ndim"):
            if vector.ndim != 1:
                raise ValueError("Dense QGT matvec expects a rank-1 vector.")
            coordinate_parameters, _ = _coordinate_parameters(
                self.parameters,
                self.mode,
            )
            flat_params, unravel = ravel_pytree(coordinate_parameters)
            if vector.size != flat_params.size:
                raise ValueError(
                    f"Vector size {vector.size} does not match "
                    f"parameter size {flat_params.size}."
                )
            if self.mode == "holomorphic":
                result = _qgt_matvec_tree(self, unravel(vector))
            else:
                result = _qgt_matvec_coordinates(self, unravel(vector))
            return ravel_pytree(result)[0]

        if jax.tree_util.tree_structure(vector) != jax.tree_util.tree_structure(
            self.parameters
        ):
            raise ValueError(
                "QGT vector pytree must match the parameter pytree structure."
            )
        return _qgt_matvec_tree(self, vector)

    def to_dense(self) -> jax.Array:
        """Materialise the dense QGT matrix.

        This is intended for tests and small models. Large-scale runs should use
        matrix-free iterative solves.
        """
        coordinate_parameters, _ = _coordinate_parameters(self.parameters, self.mode)
        flat_params, unravel = ravel_pytree(coordinate_parameters)
        eye = jnp.eye(flat_params.size, dtype=flat_params.dtype)

        def column(vector):
            if self.mode == "holomorphic":
                result = _qgt_matvec_tree(self, unravel(vector))
            else:
                result = _qgt_matvec_coordinates(self, unravel(vector))
            return ravel_pytree(result)[0]

        columns_t = jax.vmap(column)(eye)
        return columns_t.T

    def __repr__(self) -> str:
        return (
            f"QGTOnTheFly(n_parameters={self.n_parameters}, "
            f"diag_shift={self.diag_shift}, mode={self.mode!r})"
        )


@jax.jit
def _qgt_matvec_tree(qgt: QGTOnTheFly, vector: Any) -> Any:
    if qgt.mode != "holomorphic":
        coordinate_vector, reassemble = _vector_to_coordinates(vector, qgt.parameters)
        result = _qgt_matvec_coordinates(qgt, coordinate_vector)
        return reassemble(result)

    vector = tree_cast_like(vector, qgt.parameters)
    flat_samples, _ = flatten_samples(qgt.samples, qgt.hilbert_size)
    flat_weights = _flatten_weights(qgt)

    def logpsi(parameters: Any) -> jax.Array:
        variables = qgt.framework.merge_variables(parameters, qgt.model_state)

        def apply_variables(variables: Any, states: Any) -> jax.Array:
            return qgt.framework.apply_variables(
                variables,
                states,
                training_kwargs=dict(qgt.training_kwargs),
            )

        return apply_batched(
            apply_variables,
            variables,
            flat_samples,
            chunk_size=qgt.chunk_size,
        )

    _values, jvp_values = jax.jvp(logpsi, (qgt.parameters,), (vector,))
    if flat_weights is None:
        centered = jvp_values - jnp.mean(jvp_values)
        cotangent = centered / jvp_values.size
    else:
        mean = jnp.sum(flat_weights * jvp_values)
        cotangent = flat_weights * (jvp_values - mean)

    _primal, pullback = jax.vjp(logpsi, qgt.parameters)
    result = jax.tree_util.tree_map(
        jnp.conjugate,
        pullback(jnp.conjugate(cotangent))[0],
    )
    return tree_axpy(qgt.diag_shift, vector, result)


@jax.jit
def _qgt_matvec_coordinates(qgt: QGTOnTheFly, coordinate_vector: Any) -> Any:
    coordinate_parameters, reassemble = _coordinate_parameters(qgt.parameters, qgt.mode)
    coordinate_vector = tree_cast_like(coordinate_vector, coordinate_parameters)
    flat_samples, _ = flatten_samples(qgt.samples, qgt.hilbert_size)
    flat_weights = _flatten_weights(qgt)

    def logpsi_coordinates(parameters: Any) -> jax.Array:
        variables = qgt.framework.merge_variables(
            reassemble(parameters), qgt.model_state
        )

        def apply_variables(variables: Any, states: Any) -> jax.Array:
            return qgt.framework.apply_variables(
                variables,
                states,
                training_kwargs=dict(qgt.training_kwargs),
            )

        values = apply_batched(
            apply_variables,
            variables,
            flat_samples,
            chunk_size=qgt.chunk_size,
        )
        if qgt.mode == "real":
            return jnp.real(values)
        if qgt.mode == "complex":
            return jnp.stack((jnp.real(values), jnp.imag(values)), axis=-1)
        return values

    _values, jvp_values = jax.jvp(
        logpsi_coordinates,
        (coordinate_parameters,),
        (coordinate_vector,),
    )
    if flat_weights is None:
        centered = jvp_values - jnp.mean(jvp_values, axis=0, keepdims=True)
        cotangent = centered / flat_samples.shape[0]
    else:
        weight_shape = (flat_weights.shape[0],) + (1,) * (jvp_values.ndim - 1)
        weights = flat_weights.reshape(weight_shape)
        mean = jnp.sum(weights * jvp_values, axis=0, keepdims=True)
        cotangent = weights * (jvp_values - mean)

    _primal, pullback = jax.vjp(logpsi_coordinates, coordinate_parameters)
    result = pullback(cotangent)[0]
    return tree_axpy(qgt.diag_shift, coordinate_vector, result)


def _coordinate_parameters(parameters: Any, mode: str):
    if mode == "holomorphic":
        return parameters, lambda value: value
    return split_tree_to_real(parameters)


def _vector_to_coordinates(vector: Any, parameters: Any):
    del parameters
    return split_tree_to_real(vector)


def _flatten_weights(qgt: QGTOnTheFly) -> jax.Array | None:
    if qgt.sample_weights is None:
        return None
    weights = jnp.asarray(qgt.sample_weights).reshape((-1,))
    total = jnp.sum(weights)
    return weights / total


def _hashable_training_kwargs(kwargs: Any) -> tuple[tuple[str, Any], ...]:
    if kwargs is None:
        return ()
    return tuple(sorted(dict(kwargs).items()))


def qgt_onthefly(vstate: Any, **kwargs: Any) -> QGTOnTheFly:
    """Convenience constructor mirroring the optimizer-level object name."""
    return QGTOnTheFly.from_vstate(vstate, **kwargs)


__all__ = ["QGTOnTheFly", "qgt_onthefly"]
