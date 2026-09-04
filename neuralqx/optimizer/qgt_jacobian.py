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


"""Dense-Jacobian quantum geometric tensor implementations."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import jax
from neuralqx.jax.differentiation import resolve_differentiation_mode
from neuralqx.jax.differentiation import split_tree_to_real
from neuralqx.jax.streaming import apply_batched
from neuralqx.jax.tree import tree_any_complex
from neuralqx.utils.struct import field
from neuralqx.vqs.estimators.utils import flatten_samples

from .linear_operator import LinearOperator


class QGTJacobianDense(LinearOperator):
    """Dense parameter-space QGT backed by a centered sample Jacobian.

    ``QGTOnTheFly`` is the scalable default for matrix-free iterative solves.
    This representation is the fast path for small and medium dense SR solves:
    it builds the centered Jacobian once and lets direct solvers form
    ``JᴴJ + λI`` in compiled dense linear algebra.
    """

    jacobian: Any
    parameters: Any = None
    mode: str = field(static=True, default="real")

    def __post_init__(self) -> None:
        jacobian = jnp.asarray(self.jacobian)
        if jacobian.ndim != 2:
            raise ValueError(
                "QGTJacobianDense expects a rank-2 Jacobian with shape "
                "(n_samples, n_parameters)."
            )
        if self.mode not in {"real", "complex", "holomorphic"}:
            raise ValueError(
                "QGTJacobianDense mode must be 'real', 'complex', or "
                f"'holomorphic', got {self.mode!r}."
            )

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
    ) -> QGTJacobianDense:
        """Build a dense-Jacobian QGT from a variational state."""
        if diag_scale is not None:
            raise NotImplementedError(
                "QGTJacobianDense does not support diag_scale yet."
            )
        samples = vstate.samples if samples is None else samples
        flat_samples, _ = flatten_samples(samples, vstate.hilbert.size)
        flat_weights = _flatten_weights(sample_weights, samples)
        model_chunk_size = vstate.chunk_size if chunk_size is None else chunk_size

        def logpsi(parameters: Any) -> jax.Array:
            variables = vstate.merge_variables_kernel(parameters, vstate.model_state)
            return apply_batched(
                vstate.apply_variables_kernel,
                variables,
                flat_samples,
                chunk_size=model_chunk_size,
            )

        resolved_mode = resolve_differentiation_mode(
            logpsi,
            vstate.parameters,
            mode=mode,
            holomorphic=holomorphic,
        )
        jacobian = _dense_centered_jacobian(
            vstate.apply_variables_kernel,
            vstate.merge_variables_kernel,
            vstate.parameters,
            vstate.model_state,
            flat_samples,
            flat_weights,
            model_chunk_size,
            resolved_mode,
        )
        return cls(
            jacobian=jacobian,
            parameters=vstate.parameters,
            diag_shift=diag_shift,
            mode=resolved_mode,
        )

    @property
    def n_parameters(self) -> int:
        """Number of scalar parameter entries."""
        return int(self.jacobian.shape[-1])

    def __matmul__(self, vector: Any) -> Any:
        if hasattr(vector, "ndim"):
            if vector.ndim != 1:
                raise ValueError("Dense QGT matvec expects a rank-1 vector.")
            if vector.size != self.n_parameters:
                raise ValueError(
                    f"Vector size {vector.size} does not match "
                    f"parameter size {self.n_parameters}."
                )
            return _dense_jacobian_matvec(
                self.jacobian,
                self.diag_shift,
                vector,
                self.mode,
            )
        flat, reassemble = _vector_to_coordinates(vector, self.parameters, self.mode)
        result = _dense_jacobian_matvec(
            self.jacobian,
            self.diag_shift,
            flat,
            self.mode,
        )
        return reassemble(result)

    def to_dense(self) -> jax.Array:
        """Materialise ``JᴴJ + λI``."""
        return _dense_jacobian_matrix(self.jacobian, self.diag_shift, self.mode)

    def _solve(
        self,
        solve_fn: Any,
        rhs: Any,
        *,
        x0: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.mode == "holomorphic" or not tree_any_complex(self.parameters):
            return super()._solve(solve_fn, rhs, x0=x0, **kwargs)
        if not _is_dense_solver(self, solve_fn, kwargs):
            return super()._solve(solve_fn, rhs, x0=x0, **kwargs)

        rhs_coordinates, reassemble = _vector_to_coordinates(
            rhs,
            self.parameters,
            self.mode,
        )
        if x0 is not None:
            x0, _ = _vector_to_coordinates(x0, self.parameters, self.mode)
        result = solve_fn(
            self.to_dense(),
            rhs_coordinates,
            x0=x0,
            **kwargs,
        )
        if isinstance(result, tuple) and len(result) == 2:
            solution, info = result
            return reassemble(solution), info
        return reassemble(result)

    def __repr__(self) -> str:
        return (
            f"QGTJacobianDense(n_parameters={self.n_parameters}, "
            f"n_samples={self.jacobian.shape[0]}, diag_shift={self.diag_shift}, "
            f"mode={self.mode!r})"
        )


@partial(
    jax.jit,
    static_argnames=("apply_variables", "merge_variables", "chunk_size", "mode"),
)
def _dense_centered_jacobian(
    apply_variables: Any,
    merge_variables: Any,
    parameters: Any,
    model_state: Any,
    flat_samples: jax.Array,
    flat_weights: jax.Array | None,
    chunk_size: int | None,
    mode: str,
) -> jax.Array:
    flat_parameters, unravel = ravel_pytree(parameters)

    def logpsi_flat(flat: jax.Array) -> jax.Array:
        params = unravel(flat)
        variables = merge_variables(params, model_state)
        return apply_batched(
            apply_variables,
            variables,
            flat_samples,
            chunk_size=chunk_size,
        )

    if mode == "holomorphic":
        jacobian = jax.jacfwd(logpsi_flat, holomorphic=True)(flat_parameters)
    else:
        coordinate_parameters, reassemble = split_tree_to_real(parameters)
        flat_coordinates, unravel_coordinates = ravel_pytree(coordinate_parameters)

        def logpsi_coordinates(flat: jax.Array) -> jax.Array:
            params = reassemble(unravel_coordinates(flat))
            variables = merge_variables(params, model_state)
            values = apply_batched(
                apply_variables,
                variables,
                flat_samples,
                chunk_size=chunk_size,
            )
            if mode == "real":
                return jnp.real(values)
            return jnp.stack((jnp.real(values), jnp.imag(values)), axis=-1)

        jacobian = jax.jacfwd(logpsi_coordinates)(flat_coordinates)

    if flat_weights is None:
        centered = jacobian - jnp.mean(jacobian, axis=0, keepdims=True)
        n_samples = jnp.asarray(flat_samples.shape[0], dtype=jnp.real(jacobian).dtype)
        scaled = centered / jnp.sqrt(n_samples)
        if mode == "complex":
            return scaled.reshape((-1, scaled.shape[-1]))
        return scaled

    weights = flat_weights.reshape((-1,))
    weight_shape = (weights.shape[0],) + (1,) * (jacobian.ndim - 1)
    shaped_weights = weights.reshape(weight_shape)
    mean = jnp.sum(shaped_weights * jacobian, axis=0, keepdims=True)
    scaled = (jacobian - mean) * jnp.sqrt(shaped_weights)
    if mode == "complex":
        return scaled.reshape((-1, scaled.shape[-1]))
    return scaled


@partial(jax.jit, static_argnames=("mode",))
def _dense_jacobian_matrix(
    jacobian: jax.Array, diag_shift: Any, mode: str
) -> jax.Array:
    matrix = jnp.conjugate(jacobian).T @ jacobian
    if mode != "holomorphic":
        matrix = jnp.real(matrix).astype(jnp.real(jacobian).dtype)
    shift = jnp.asarray(diag_shift, dtype=matrix.dtype)
    return matrix + shift * jnp.eye(matrix.shape[0], dtype=matrix.dtype)


@partial(jax.jit, static_argnames=("mode",))
def _dense_jacobian_matvec(
    jacobian: jax.Array,
    diag_shift: Any,
    vector: jax.Array,
    mode: str,
) -> jax.Array:
    result = jnp.conjugate(jacobian).T @ (jacobian @ vector)
    if mode != "holomorphic":
        result = jnp.real(result).astype(jnp.asarray(vector).dtype)
    return result + jnp.asarray(diag_shift, dtype=result.dtype) * vector


def _vector_to_coordinates(vector: Any, parameters: Any, mode: str):
    if mode == "holomorphic" or not tree_any_complex(parameters):
        flat, unravel = ravel_pytree(vector)
        return flat, unravel
    coordinates, reassemble_coordinates = split_tree_to_real(vector)
    flat, unravel_coordinates = ravel_pytree(coordinates)

    def reassemble(flat_value: jax.Array) -> Any:
        return reassemble_coordinates(unravel_coordinates(flat_value))

    return flat, reassemble


def _is_dense_solver(
    operator: QGTJacobianDense, solve_fn: Any, kwargs: dict[str, Any]
) -> bool:
    from .solver import auto as auto_solver
    from .solver import cholesky
    from .solver import pinv
    from .solver import pinv_smooth
    from .solver import solve
    from .solver._api import solver_base
    from .solver._api import solver_config

    base = solver_base(solve_fn)
    if base in {solve, cholesky, pinv, pinv_smooth}:
        return True
    if base is not auto_solver:
        return False
    merged = solver_config(solve_fn) | dict(kwargs)
    dense_threshold = int(merged.get("dense_threshold", 2048))
    return operator.n_parameters <= dense_threshold


def _flatten_weights(sample_weights: Any | None, samples: Any) -> jax.Array | None:
    if sample_weights is None:
        return None
    expected = jnp.asarray(samples).shape[:-1]
    weights = jnp.asarray(sample_weights)
    if weights.shape != expected:
        raise ValueError(
            "sample_weights must match samples.shape[:-1]; "
            f"got {weights.shape} and expected {expected}."
        )
    if not bool(jnp.all(jnp.isfinite(weights))):
        raise ValueError("sample_weights must be finite.")
    flat = weights.reshape((-1,))
    total = jnp.sum(flat)
    if not bool(jnp.isfinite(total)) or float(total) == 0.0:
        raise ValueError("sample_weights must have a finite nonzero sum.")
    return flat / total


def qgt_jacobian_dense(vstate: Any, **kwargs: Any) -> QGTJacobianDense:
    """Convenience constructor for :class:`QGTJacobianDense`."""
    return QGTJacobianDense.from_vstate(vstate, **kwargs)


__all__ = ["QGTJacobianDense", "qgt_jacobian_dense"]
