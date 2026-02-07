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


from typing import Any
from typing import Optional
from typing import Union

import numpy as np

import jax
import jax.numpy as jnp

from neuralqx.debug import errors_only

from ._types import Order, Backend

from ._base import _rank_digits_base
from ._base import _choose_return_dtype
from ._base import _unrank_numbers_base
from ._base import _rank_digits_base_jax
from ._base import _unrank_numbers_base_jax

from ._constrained_index import _reconstruct_from_reduced_free_values
from ._constrained_index import _extract_reduced_free_values
from ._constrained_index import _space_is_gauge_fixed
from ._constrained_index import _extract_reduced_free_values_jax
from ._constrained_index import _reconstruct_from_reduced_free_values_jax

from ._netket import _infer_netket_site_order
from ._netket import _local_digits_to_values
from ._netket import _local_values_to_digits
from ._netket import _local_values_to_digits_jax
from ._netket import _local_digits_to_values_jax

from ._array_normalise import _as_numpy
from ._array_normalise import _ensure_2d_states
from ._array_normalise import _ensure_1d_numbers
from ._array_normalise import _ensure_2d_states_jax
from ._array_normalise import _ensure_1d_numbers_jax


def _is_jax_value(x: Any) -> bool:
    """Return True if x is a JAX value (Array or Tracer)."""
    arr_t = getattr(jax, "Array", None)
    if arr_t is None:
        return isinstance(x, jax.core.Tracer) or hasattr(x, "__jax_array__")
    return isinstance(x, (arr_t, jax.core.Tracer)) or hasattr(x, "__jax_array__")


# @partial(jax.jit, static_argnums=(0, 2))
def _states_to_numbers_jax(space: Any, states: Any, order: Order) -> jax.Array:
    """JAX backend for states_to_numbers. Returns int64 with shape states.shape[:-1]."""
    st = jnp.asarray(states, dtype=space.dtype)
    orig_shape = st.shape
    st2d, was_single = _ensure_2d_states_jax(st)

    local_states = space.allowed_basis_states
    L = int(local_states.length)

    if _space_is_gauge_fixed(space):
        free_vals2d = _extract_reduced_free_values_jax(space, st2d)
        digits2d = _local_values_to_digits_jax(local_states, free_vals2d)
        nums1d = _rank_digits_base_jax(digits2d, L, "C")
    else:
        digits2d = _local_values_to_digits_jax(local_states, st2d)
        nums1d = _rank_digits_base_jax(digits2d, L, order)

    if was_single:
        return nums1d[0]
    return nums1d.reshape(orig_shape[:-1])


# @partial(jax.jit, static_argnums=(0, 2))
def _numbers_to_states_jax(space: Any, numbers: Any, order: Order) -> jax.Array:
    """JAX backend for numbers_to_states. Returns states with shape numbers.shape + (N,)."""
    nums1d, was_scalar, orig_shape = _ensure_1d_numbers_jax(numbers)

    local_states = space.allowed_basis_states
    L = int(local_states.length)

    if hasattr(space, "gauge_fixing") and hasattr(space, "reimpose_gauge_fixing"):
        gf = space.gauge_fixing
        D = int(space.gauge_dimensions) * int(len(gf.free))
        digits2d = _unrank_numbers_base_jax(nums1d, L, D, "C")
        free_vals2d = _local_digits_to_values_jax(local_states, digits2d)
        sigma2d = _reconstruct_from_reduced_free_values_jax(space, free_vals2d)

        N = int(space.size)
        if was_scalar:
            return sigma2d[0]
        return sigma2d.reshape((*orig_shape, N))

    N = int(space.size)
    digits2d = _unrank_numbers_base_jax(nums1d, L, N, order)
    vals2d = _local_digits_to_values_jax(local_states, digits2d).astype(space.dtype)

    if was_scalar:
        return vals2d[0]
    return vals2d.reshape((*orig_shape, N))


@errors_only(tag="STATES_TO_NUMBERS")
def states_to_numbers(
    space: Any,
    states: Any,
    *,
    backend: Backend = "auto",
    return_dtype: Union[str, np.dtype, None] = "auto",
    validate: bool = False,
    netket_order: Optional[Order] = None,
) -> Any:
    """
    Convert basis states to sequential numbers, matching NetKet's convention.

    Works even when NetKet cannot index the space, by using a purely arithmetic
    rank/unrank on:
      - the full state (unconstrained direct-product spaces), or
      - the reduced "free-variable" parameterization (gauge-fixed constrained spaces).

    :param space: A core Hilbert space (AbstractHilbertSpace) or compatible object.
    :param states: Array-like of shape (N,) or (..., N) containing local basis values.
    :param backend: "auto" | "netket" | "python".
        - "netket": call space.hilbert.states_to_numbers directly.
        - "python": always use arithmetic ranking (works when not indexable).
        - "auto": use "netket" when it works, else fallback to "python".
    :param return_dtype: "auto" | "int64" | "object" (python big-int).
        "auto" returns int64 when safe, else object.
    :param validate: If True, perform extra checks (may be slow):
        - unconstrained: validates values are in local basis.
        - gauge-fixed: also validates gauge invariance (space.check_states if present).
    :param netket_order: Optional override for site-order ("C" or "F"). If None, inferred
        from NetKet using a tiny N=2 reference Hilbert.

    :return: numbers with shape (...) matching the batch shape of `states` (without trailing N).
    """

    if backend == "jax" or (backend == "auto" and _is_jax_value(states)):
        local_states = space.allowed_basis_states
        order = netket_order or _infer_netket_site_order(local_states)
        return _states_to_numbers_jax(space, states, order=order)

    states_np = _as_numpy(states)
    orig_shape = states_np.shape
    states2d, was_single = _ensure_2d_states(states_np)

    # optional validation for GI spaces
    if validate and _space_is_gauge_fixed(space) and hasattr(space, "check_states"):
        ok = _as_numpy(space.check_states(jnp.asarray(states2d)))
        if not bool(np.all(ok)):
            raise ValueError(
                "states_to_numbers(validate=True): input contains non gauge-invariant states."
            )

    if backend in ("auto", "netket"):
        try:
            states_np = _as_numpy(states)
            orig_shape = states_np.shape

            # NetKet generally expects (N,) or (B,N). If we get higher-rank, flatten batch dims
            if states_np.ndim == 1:
                out = space.hilbert.states_to_numbers(states_np)
                out_np = _as_numpy(out)
                # netKet returns scalar (0-d) for single state, return Python/NumPy scalar
                return out_np.item() if out_np.ndim == 0 else out_np.reshape(()).item()

            # flatten leading batch dims to (B,N)
            st2d, _ = _ensure_2d_states(states_np)
            out = space.hilbert.states_to_numbers(st2d)
            out_np = _as_numpy(out).reshape((st2d.shape[0],))

            # reshape back to original batch shape (orig_shape[:-1])
            return out_np.reshape(orig_shape[:-1])

        except Exception:
            if backend == "netket":
                raise
            # else fallback to python below

    local_states = space.allowed_basis_states
    # L = int(_as_numpy(local_states.all_states()).shape[0])
    L = int(local_states.length)

    order = netket_order or _infer_netket_site_order(local_states)

    if _space_is_gauge_fixed(space):
        # reduced (free-variable) representation: digits length = G*F
        free_vals2d = _extract_reduced_free_values(space, states2d)
        digits2d = _local_values_to_digits(local_states, free_vals2d)
        D = int(digits2d.shape[1])
        out_dtype = _choose_return_dtype(L, D, return_dtype)

        # reduced space uses stable C-order
        nums1d = _rank_digits_base(digits2d, L, "C", out_dtype)
    else:
        # full direct-product representation: digits length = N
        digits2d = _local_values_to_digits(local_states, states2d)
        N = int(digits2d.shape[1])
        out_dtype = _choose_return_dtype(L, N, return_dtype)
        nums1d = _rank_digits_base(digits2d, L, order, out_dtype)

    # reshape back
    if was_single:
        return nums1d[0]

    # orig batch shape = orig_shape[:-1]
    out_shape = orig_shape[:-1]
    return nums1d.reshape(out_shape)


@errors_only(tag="NUMBERS_TO_STATES")
def numbers_to_states(
    space: Any,
    numbers: Any,
    *,
    backend: Backend = "auto",
    validate: bool = False,
    netket_order: Optional[Order] = None,
) -> "jnp.ndarray":
    """
    Convert sequential numbers to basis states, matching NetKet's convention.

    Works even when NetKet cannot index the space, by using arithmetic unranking.

    :param space: A core Hilbert space (AbstractHilbertSpace) or compatible object.
    :param numbers: Scalar or array-like of integers.
    :param backend: "auto" | "netket" | "python" (see states_to_numbers).
    :param validate: If True, check that numbers are in range for the chosen mapping.
    :param netket_order: Optional override for site-order ("C" or "F"). If None, inferred.

    :return: JAX array of states with shape (..., N) and dtype=space.dtype.
    """

    if backend == "jax" or (backend == "auto" and _is_jax_value(numbers)):
        local_states = space.allowed_basis_states
        order = netket_order or _infer_netket_site_order(local_states)
        return _numbers_to_states_jax(space, numbers, order=order)

    nums1d, was_scalar = _ensure_1d_numbers(numbers)
    orig_shape = _as_numpy(numbers).shape

    if backend in ("auto", "netket"):
        try:
            nums1d, was_scalar = _ensure_1d_numbers(numbers)
            orig_shape = _as_numpy(numbers).shape
            N = int(space.size)

            if was_scalar:
                # call NetKet with scalar so it can return (N,) directly
                out = space.hilbert.numbers_to_states(int(nums1d[0]))
                out_j = jnp.asarray(out, dtype=space.dtype)

                # if NetKet returns (N,), keep it, if it returns (1,N), unwrap it
                if out_j.ndim == 1:
                    return out_j
                if out_j.ndim == 2 and out_j.shape[0] == 1:
                    return out_j[0]
                # last-resort: reshape to (N,)
                return out_j.reshape((N,))

            # non-scalar numbers: call NetKet with flat (B,)
            out = space.hilbert.numbers_to_states(nums1d.astype(np.int64))
            out_j = jnp.asarray(out, dtype=space.dtype)

            # ensure (B,N)
            if out_j.ndim == 1:
                # rare, but if N==1 NetKet might return (B,)
                out_j = out_j.reshape((nums1d.shape[0], 1))
            elif out_j.ndim != 2:
                out_j = out_j.reshape((nums1d.shape[0], N))

            # reshape back to (*orig_shape, N)
            return out_j.reshape((*orig_shape, N))

        except Exception:
            if backend == "netket":
                raise
            # else fallback to python below

    local_states = space.allowed_basis_states
    # L = int(_as_numpy(local_states.all_states()).shape[0])
    L = int(local_states.length)

    order = netket_order or _infer_netket_site_order(local_states)

    if _space_is_gauge_fixed(space):
        # reduced space: D = G*F
        gf = space.gauge_fixing
        D = int(space.gauge_dimensions) * int(len(gf.free))

        if validate:
            # numbers must be < L**D
            maxn = pow(L, D)
            for n in nums1d:
                if int(n) < 0 or int(n) >= maxn:
                    raise ValueError(
                        "numbers_to_states(validate=True): number out of range for reduced "
                        "GI space."
                    )

        # reduced space uses stable C-order
        digits2d = _unrank_numbers_base(nums1d, L, D, "C")
        free_vals2d = _local_digits_to_values(local_states, digits2d)
        sigma = _reconstruct_from_reduced_free_values(space, free_vals2d)

        # reshape output
        if was_scalar:
            return sigma[0]
        # orig numbers shape + trailing N
        out_shape = (*orig_shape, int(space.size))
        return sigma.reshape(out_shape)

    # unconstrained direct-product space: N = space.size
    N = int(space.size)

    if validate:
        maxn = pow(L, N)
        for n in nums1d:
            if int(n) < 0 or int(n) >= maxn:
                raise ValueError(
                    "numbers_to_states(validate=True): number out of range for "
                    "direct-product space."
                )

    digits2d = _unrank_numbers_base(nums1d, L, N, order)
    vals2d = _local_digits_to_values(local_states, digits2d)

    out_np = vals2d.astype(space.allowed_basis_states.dtype, copy=False)

    if was_scalar:
        return jnp.asarray(out_np.reshape((N,)), dtype=space.dtype)

    out_shape = (*orig_shape, N)
    return jnp.asarray(out_np.reshape(out_shape), dtype=space.dtype)
