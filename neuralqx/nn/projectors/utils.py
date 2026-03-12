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


from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Callable
from typing import Any
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from neuralqx.graph import Graph
from neuralqx.utils.symmetries.types import _canon_edge

Array = jnp.ndarray


#
#
#  Low-level permutation utils


def _ensure_identity_first(perms: Array) -> Array:
    """Ensure the identity permutation exists and is in row 0 of a list of symmetries."""

    if perms.ndim != 2:
        raise ValueError(f"`perms` must be [G, N], got {perms.shape}.")

    # get shapes and ensure identity is present
    G, N = perms.shape
    id_perm = jnp.arange(N, dtype=perms.dtype)
    is_id = jnp.all(perms == id_perm[None, :], axis=1)

    has_id = bool(is_id.any())
    if has_id:
        idx = int(jnp.argmax(is_id))
        if idx == 0:
            return perms.astype(jnp.int32)

        # reorder so that row idx goes to front
        rest_idx = jnp.concatenate(
            [
                jnp.arange(0, idx, dtype=jnp.int32),
                jnp.arange(idx + 1, G, dtype=jnp.int32),
            ]
        )
        order = jnp.concatenate([jnp.array([idx], dtype=jnp.int32), rest_idx])
        return perms[order].astype(jnp.int32)
    else:
        return jnp.concatenate([id_perm[None, :], perms], axis=0).astype(jnp.int32)


def _ensure_identity_first_eager(perms: jnp.ndarray) -> jnp.ndarray:
    """Eager-only: ensure identity is present and first (OK to change leading dimension)."""

    # eager host array
    perms = np.asarray(perms)

    G, N = perms.shape
    id_perm = np.arange(N, dtype=perms.dtype)
    is_id = (perms == id_perm[None, :]).all(axis=1)
    if is_id.any():
        idx = int(np.argmax(is_id))
        if idx != 0:
            # swap to front
            perms[[0, idx]] = perms[[idx, 0]]
        return jnp.asarray(perms)
    else:
        perms2 = np.concatenate([id_perm[None, :], perms], axis=0)
        return jnp.asarray(perms2)


def _normalize_perms_and_chars_eager(
    perms: jnp.ndarray,
    characters: Optional[jnp.ndarray],
    *,
    require_identity_if_characters: bool = False,
    identity_character: complex = 1.0,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Ensure identity permutation is present and in row 0, and reorder/prepend `characters`
    identically. Operates eagerly (NumPy) on host.

    - If identity exists at row idx>0: swap row 0 <-> idx; reorder characters accordingly.
    - If identity is missing:
        * If `characters is None`: prepend identity row.
        * If `characters is not None`:
            - If `require_identity_if_characters` is True: raise (safer).
            - Else: prepend identity row and prepend `identity_character` to characters.
    """
    if perms.ndim != 2:
        raise ValueError(f"`perms` must be [G, N], got shape {perms.shape}.")

    p = np.asarray(perms)
    G, N = p.shape
    idp = np.arange(N, dtype=p.dtype)

    # where is the identity?
    is_id = (p == idp[None, :]).all(axis=1)
    if is_id.any():
        idx = int(np.argmax(is_id))
        if idx != 0:
            # swap rows
            p[[0, idx]] = p[[idx, 0]]
            if characters is not None:
                c = np.asarray(characters)
                if c.shape[0] != G:
                    raise ValueError(
                        f"`characters` length {c.shape[0]} must match G={G}."
                    )
                c[[0, idx]] = c[[idx, 0]]
                characters = jnp.asarray(c)
    else:
        # identity missing
        if characters is not None and require_identity_if_characters:
            raise ValueError(
                "Permutations are missing the identity but `characters` were provided. "
                "Either include the identity in your perms, or set "
                "`require_identity_if_characters=False` (and we will prepend with "
                f"identity_character={identity_character})."
            )
        # prepend identity
        p = np.concatenate([idp[None, :], p], axis=0)
        if characters is not None:
            c = np.asarray(characters)
            characters = jnp.asarray(np.concatenate([[identity_character], c], axis=0))

    return jnp.asarray(p, dtype=jnp.int32), characters


def _permute_last_axis(x: Array, perm_last_axis: Array) -> Array:
    """Permute only the last axis of x with permutation indices ``perm_last_axis``."""
    return jnp.take(x, perm_last_axis, axis=-1)


def _lift_edge_perm_to_levels(
    perm_edges: Array,
    n_edges: int,
    n_levels: int,
) -> Array:
    """
    Lift a permutation over edges (length n_edges) to a permutation over all
    DOFs (graph levels) (length n_edges * n_levels) assuming level-interleaved storage:

        [e0_l0, e1_l0, ..., e(M-1)_l0, e0_l1, e1_l1, ..., e(M-1)_l1, ...]
    """
    levels = jnp.arange(n_levels, dtype=jnp.int32)[:, None]
    dst_edges = perm_edges[None, :].astype(jnp.int32)
    lifted = (levels * n_edges) + dst_edges
    return lifted.reshape(-1)


def _maybe_lift_perms_to_full_size_dynamic(
    perms_edge: Array,
    N_full: int,
) -> Array:
    """
    Lift edge-level perms [G, M] to full-size [G, N_full] at runtime, using N_full inferred from
    sigma.shape[-1]
    """

    G, M = perms_edge.shape
    # compute L = N_full // M, assume exact divisibility (raise if not)
    # we keep the check on host for readability, if needed, can be made JAX-safe
    if N_full % M != 0:
        raise ValueError(
            f"Cannot lift edge-level perms of len={M} to full size N={N_full} (not divisible)."
        )
    L = N_full // M

    def lift_one(p):
        return _lift_edge_perm_to_levels(p, M, L)

    return jax.vmap(lift_one, in_axes=0)(perms_edge.astype(jnp.int32))


#
#
#  Graph/Symmetry to edge-index perms


def symmetry_to_index_perm(symmetry_obj, graph: Graph) -> Sequence[int]:
    """
    Convert a user symmetry object into an edge-index permutation using ``graph``.

    Requirements on ``graph``:
      - exposes ``edges`` (iterable of canonical edges)
      - exposes ``edge_to_index(edge) -> int``

    Requirements on ``symmetry_obj``:
      - defines ``map_edge(edge) -> edge'``
    """

    edges = getattr(graph, "edges", None)

    if edges is None:
        raise ValueError(
            "`graph` must expose `edges` (iterable of edges) for symmetry mapping."
        )

    if not hasattr(symmetry_obj, "map_edge"):
        raise ValueError("`symmetry_obj` must define `map_edge(edge)`.")

    perm = []
    map_edge: Callable[[Any], Any] = symmetry_obj.map_edge

    for e in edges:
        e_dst = _canon_edge(map_edge(e))
        j = graph.edge_to_index(e_dst)
        perm.append(int(j))

    return perm


def _build_edge_or_full_perms(
    *,
    symmetries: Optional[Iterable[object]],
    graph: Optional[object],
    index_perms: Optional[Iterable[Sequence[int]]],
) -> tuple[Array, Array]:
    """
    Returns (perms_full_or_empty, perms_edge_or_empty).

      - if user passed ``index_perms``, we sniff their length agreement later at call-site
      - if user passed ``symmetries + graph``, we compute edge-level perms here

    Exactly one of the two outputs will be non-empty
    """

    if index_perms is None and symmetries is None:
        raise ValueError("Provide either `index_perms` or (`symmetries` and `graph`).")

    if index_perms is not None:
        permlist = jnp.asarray([list(p) for p in index_perms], dtype=jnp.int32)

        # we dont know yet if these are edge-level or full-size, decide at call
        return permlist, jnp.empty((0, 0), dtype=jnp.int32)

    if graph is None:
        raise ValueError("When passing `symmetries`, you must also pass `graph`.")

    perms_edge = [symmetry_to_index_perm(s, graph) for s in symmetries]
    perms_edge = jnp.asarray(perms_edge, dtype=jnp.int32)

    return jnp.empty((0, 0), dtype=jnp.int32), perms_edge


#
#
#   Projection module (stable complex log-mean-exp)


def _project_log_psi(
    base_call,
    sigma: Array,
    perms_full: Array,
    characters: Optional[Array],
    *,
    conjugate_characters: bool = False,
    irrep_dim: Optional[float] = None,  # optional d_λ; numerically irrelevant for MC
    divide_by_group: bool = True,  # keep as True for stable magnitudes
) -> Array:
    """
    Compute: log ψ_proj(σ) = log( (c) * ∑_g χ(g) e^{logψ(g·σ)} ), with a real shift for stability.
    Where c is optionally (d_λ / |G|), or 1/|G| if divide_by_group=True, else 1.

    Notes:
      - Any global scalar factor cancels in MC sampling and local estimators.
      - We keep it here to match textbook projector formulas if desired.
    """

    def logpsi_for_perm(p_row):
        sig_p = jnp.take(sigma, p_row, axis=-1)
        return base_call(sig_p)

    # (G, B) complex
    logpsi_all = jax.vmap(logpsi_for_perm, in_axes=0)(perms_full)
    group_size = perms_full.shape[0]

    # Fast path for a single projector element.
    # Avoiding log(exp(logpsi)) keeps the original complex branch of logpsi unchanged.
    if group_size == 1:
        scale = jnp.asarray(1.0, dtype=logpsi_all.dtype)

        if characters is not None:
            chars = jnp.conj(characters) if conjugate_characters else characters
            chars = chars.astype(logpsi_all.dtype)
            scale = scale * chars[0]

        if divide_by_group:
            scale = scale / group_size
        if irrep_dim is not None:
            scale = scale * irrep_dim

        return logpsi_all[0] + jnp.log(scale)

    # stable shift on real part
    a = jnp.max(jnp.real(logpsi_all), axis=0, keepdims=True)
    amps = jnp.exp(logpsi_all - a)  # complex

    if characters is not None:
        chars = jnp.conj(characters) if conjugate_characters else characters
        chars = chars.astype(amps.dtype)
        amps = amps * chars[:, None]

    # sum over group
    sum_amp = jnp.sum(amps, axis=0)

    # (optional) global scalar prefactor (no numerical effect on MC)
    c = 1.0
    if divide_by_group:
        c = c / group_size
    if irrep_dim is not None:
        c = c * irrep_dim

    # log with shift re-added
    return jnp.log(c * sum_amp) + a.squeeze(0)
