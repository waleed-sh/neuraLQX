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
from typing import Iterable
from typing import Sequence
from typing import Tuple

from dataclasses import field

import flax.linen as nn

import jax.numpy as jnp

from ..utils import Array
from ..utils import _maybe_lift_perms_to_full_size_dynamic
from ..utils import _normalize_perms_and_chars_eager
from ..utils import _project_log_psi
from ..utils import _build_edge_or_full_perms

_IntTuple = Tuple[int, ...]
_PermTuple = Tuple[_IntTuple, ...]


#
#
#   Dynamic projector module


class GroupProjectorDynamic(nn.Module):
    """
    A dynamic projector that:
      - stores either edge-level perms OR full-size perms (identity-first enforced),
      - lifts at runtime (from edge to full) using σ.shape[-1] when needed,
      - applies a stable complex log-mean-exp projector each call
    """

    base: nn.Module
    """The base flax.linen implementation of the network."""

    perms_full: object = field(
        default=None, repr=False, compare=False, metadata={"pytree_node": False}
    )
    """The list of full permutations."""

    perms_edge: object = field(
        default=None, repr=False, compare=False, metadata={"pytree_node": False}
    )
    """The list of edge permutations."""

    characters: object = field(
        default=None, repr=False, compare=False, metadata={"pytree_node": False}
    )
    """Characters of the irreps."""

    # internals flags
    conjugate_characters: bool = False
    irrep_dim: Optional[float] = None
    divide_by_group: bool = True

    @nn.compact
    def __call__(self, sigma: Array, **kwargs) -> Array:

        # determine full-size perms for this input
        if self.perms_full is not None and self.perms_full.size > 0:
            # already normalised, no runtime shuffling
            perms_full = self.perms_full
        elif self.perms_edge is not None and self.perms_edge.size > 0:
            N = sigma.shape[-1]
            perms_full = _maybe_lift_perms_to_full_size_dynamic(self.perms_edge, N)
        else:
            raise ValueError(f"No permutations provided to {type(self).__name__}.")

        def base_call(x):
            return self.base(x, **kwargs)

        # characters dtype follows network dtype automatically inside _project_log_psi
        return _project_log_psi(
            base_call,
            sigma,
            perms_full,
            self.characters,
            conjugate_characters=self.conjugate_characters,
            irrep_dim=self.irrep_dim,
            divide_by_group=self.divide_by_group,
        )


def wrap_model(
    base_model: nn.Module,
    *,
    symmetries: Optional[Iterable[object]] = None,
    graph: Optional[object] = None,
    index_perms: Optional[Iterable[Sequence[int]]] = None,
    characters: Optional[Sequence[complex]] = None,
    conjugate_characters: bool = False,
    irrep_dim: Optional[float] = None,
    divide_by_group: bool = True,
) -> GroupProjectorDynamic:

    if index_perms is None and symmetries is None:
        raise ValueError("Provide either `index_perms` or (`symmetries` and `graph`).")

    chars_arr = None if characters is None else jnp.asarray(characters)

    perms_full_static, perms_edge_static = _build_edge_or_full_perms(
        symmetries=symmetries, graph=graph, index_perms=index_perms
    )

    # normalise and align (works for either edge- or full-size perms)
    if perms_full_static.size > 0:
        perms_full_norm, chars_aligned = _normalize_perms_and_chars_eager(
            perms_full_static,
            chars_arr,
            require_identity_if_characters=False,
            identity_character=1.0,
        )
        return GroupProjectorDynamic(
            base=base_model,
            perms_full=perms_full_norm,
            perms_edge=None,
            characters=chars_aligned,
            conjugate_characters=conjugate_characters,
            irrep_dim=irrep_dim,
            divide_by_group=divide_by_group,
        )
    else:
        perms_edge_norm, chars_aligned = _normalize_perms_and_chars_eager(
            perms_edge_static,
            chars_arr,
            require_identity_if_characters=False,
            identity_character=1.0,
        )
        return GroupProjectorDynamic(
            base=base_model,
            perms_full=None,
            perms_edge=perms_edge_norm,
            characters=chars_aligned,
            conjugate_characters=conjugate_characters,
            irrep_dim=irrep_dim,
            divide_by_group=divide_by_group,
        )


def projector(
    *,
    symmetries: Optional[Iterable[object]] = None,
    graph: Optional[object] = None,
    index_perms: Optional[Iterable[Sequence[int]]] = None,
    characters: Optional[Sequence[complex]] = None,
    conjugate_characters: bool = False,
    irrep_dim: Optional[float] = None,
    divide_by_group: bool = True,
):
    if index_perms is None and symmetries is None:
        raise ValueError("Provide either `index_perms` or (`symmetries` and `graph`).")

    chars_arr = None if characters is None else jnp.asarray(characters)
    perms_full_static, perms_edge_static = _build_edge_or_full_perms(
        symmetries=symmetries, graph=graph, index_perms=index_perms
    )

    # normalise eagerly now, store already-normalised in the closure
    if perms_full_static.size > 0:
        perms_full_norm, chars_aligned = _normalize_perms_and_chars_eager(
            perms_full_static,
            chars_arr,
            require_identity_if_characters=False,
            identity_character=1.0,
        )

        def _decorator(BaseModuleClass: type[nn.Module]):
            def _factory(*args, **kwargs) -> GroupProjectorDynamic:
                base_instance = BaseModuleClass(*args, **kwargs)
                return GroupProjectorDynamic(
                    base=base_instance,
                    perms_full=perms_full_norm,
                    perms_edge=None,
                    characters=chars_aligned,
                    conjugate_characters=conjugate_characters,
                    irrep_dim=irrep_dim,
                    divide_by_group=divide_by_group,
                )

            _factory.__name__ = BaseModuleClass.__name__
            _factory.__doc__ = BaseModuleClass.__doc__
            return _factory

        return _decorator
    else:
        perms_edge_norm, chars_aligned = _normalize_perms_and_chars_eager(
            perms_edge_static,
            chars_arr,
            require_identity_if_characters=False,
            identity_character=1.0,
        )

        def _decorator(BaseModuleClass: type[nn.Module]):
            def _factory(*args, **kwargs) -> GroupProjectorDynamic:
                base_instance = BaseModuleClass(*args, **kwargs)
                return GroupProjectorDynamic(
                    base=base_instance,
                    perms_full=None,
                    perms_edge=perms_edge_norm,
                    characters=chars_aligned,
                    conjugate_characters=conjugate_characters,
                    irrep_dim=irrep_dim,
                    divide_by_group=divide_by_group,
                )

            _factory.__name__ = BaseModuleClass.__name__
            _factory.__doc__ = BaseModuleClass.__doc__
            return _factory

        return _decorator
