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

from collections.abc import Callable
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from netket.utils import get_afun_if_module, mpi
from netket.utils.types import Array, PyTree
from netket.hilbert import DiscreteHilbert


def to_array_numpy(
    hilbert: DiscreteHilbert,
    apply_fun: Callable[[PyTree, Array], Array],
    variables: PyTree,
    *,
    normalize: bool = True,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """
    NumPy-backed variant of `to_array`:

      ψ[i] = exp(apply_fun(variables, basis_state_i))

    for all basis states of `hilbert`, evaluated in JAX on chunks,
    but stored directly in a NumPy array of length `hilbert.n_states`.

    Differences from the original JAX `to_array`:
      - Returns a NumPy ndarray (not a JAX DeviceArray).
      - Internally processes states in chunks and never creates a
        giant JAX array of shape (n_states,).
      - Currently supports only single-process (mpi.n_nodes == 1)
        and no experimental sharding.

    It works for both real and complex amplitudes: dtype is inferred from
    the first chunk.
    """

    if not hilbert.is_indexable:
        raise RuntimeError("The hilbert space is not indexable")

    if mpi.n_nodes != 1:
        raise NotImplementedError(
            "to_array_numpy currently supports only mpi.n_nodes == 1 "
            "(no MPI splitting)."
        )

    # no sharding support here, keep semantics simple and predictable
    from netket.utils import config

    if getattr(config, "netket_experimental_sharding", False):
        raise NotImplementedError(
            "to_array_numpy does not support netket_experimental_sharding."
        )

    apply_fun = get_afun_if_module(apply_fun)

    n_states = int(hilbert.n_states)
    if chunk_size is None:
        # simple heuristic: up to 1e6 states per chunk, but not more than n_states
        chunk_size = min(1_000_000, n_states)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    # batched apply_fun: (variables, (batch, ...)) -> (batch,)
    def _apply_batch(variables, σ_batch):
        # σ_batch: shape (batch, ...) of basis states
        return jax.vmap(lambda σ: apply_fun(variables, σ))(σ_batch)

    apply_batch_jit = jax.jit(_apply_batch)

    vs: Optional[np.ndarray] = None
    norm_sq: float = 0.0

    for start in range(0, n_states, chunk_size):
        end = min(start + chunk_size, n_states)

        # indices for this chunk as JAX array
        idx_chunk = jnp.arange(start, end, dtype=jnp.int64)

        # basis states σ for these indices (JAX array)
        # we intentionally use numbers_to_states, not all_states, to avoid
        # building the full states table at once.
        sigma_chunk = hilbert.numbers_to_states(idx_chunk)

        # log ψ for this chunk (JAX array of shape (end-start,))
        log_psi_chunk = apply_batch_jit(variables, sigma_chunk)

        # ψ for this chunk (still JAX)
        psi_chunk = jnp.exp(log_psi_chunk)

        # Convert this chunk only to NumPy
        psi_chunk_np = np.asarray(psi_chunk)

        # allocate vs with the correct dtype on the first iteration
        if vs is None:
            vs = np.empty(n_states, dtype=psi_chunk_np.dtype)

        vs[start:end] = psi_chunk_np

        if normalize:
            norm_sq += float(np.sum(np.abs(psi_chunk_np) ** 2, dtype=np.float64))

    if vs is None:
        # hilbert.n_states == 0 (degenerate case)
        return np.empty(0, dtype=np.float64)

    if normalize:
        norm = np.sqrt(norm_sq)
        if norm > 0:
            vs /= norm

    return vs
