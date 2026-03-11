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

"""U(1)-specific Hilbert state enumerator classes."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import TYPE_CHECKING

import numpy as np

from neuralqx.hilbert.utils.index._abstract import HilbertStateEnumerator

from ._constrained_index import _extract_reduced_free_values
from ._constrained_index import _space_is_gauge_fixed
from ._netket import _infer_netket_site_order
from ._netket import _local_digits_to_values
from ._netket import _local_values_to_digits
from .mapping import numbers_to_states as _u1_numbers_to_states
from .mapping import states_to_numbers as _u1_states_to_numbers
from .utils._array_normalise import _as_numpy
from .utils._array_normalise import _ensure_1d_numbers
from .utils._array_normalise import _ensure_2d_states
from .utils._base import _unrank_numbers_base

if TYPE_CHECKING:
    from neuralqx.hilbert.abstract_hilbert_core import AbstractHilbertSpace


def _finalise_explanation(
    *,
    single: bool,
    metadata: Dict[str, Any],
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a compact explanation payload for single or batched inputs."""
    if single:
        return {**metadata, **entries[0]}
    return {**metadata, "entries": entries}


class _U1StateEnumeratorBase(HilbertStateEnumerator["AbstractHilbertSpace"]):
    """Shared implementation for U(1) state enumerators."""

    _SCHEME_NAME = "u1"
    _ORDERING_CONTRACT = "U(1) mixed-radix ordering."

    @property
    def scheme_name(self) -> str:
        return self._SCHEME_NAME

    @property
    def ordering_contract(self) -> str:
        return self._ORDERING_CONTRACT

    @property
    def requires_full_precompute(self) -> bool:
        return False

    def supports_lazy_mode(self, core: "AbstractHilbertSpace") -> bool:
        # U(1) rank/unrank uses arithmetic mapping and does not require any
        # global precomputed sector table.
        return True

    def states_to_numbers(
        self,
        core: "AbstractHilbertSpace",
        states: Any,
        *,
        validate: bool = False,
        **kwargs: Any,
    ) -> Any:
        from neuralqx.hilbert.utils.index import states_to_numbers

        return states_to_numbers(self, core, states, validate=validate, **kwargs)

    def numbers_to_states(
        self,
        core: "AbstractHilbertSpace",
        numbers: Any,
        *,
        validate: bool = False,
        **kwargs: Any,
    ) -> Any:
        from neuralqx.hilbert.utils.index import numbers_to_states

        return numbers_to_states(self, core, numbers, validate=validate, **kwargs)

    def explain_state_to_number(
        self,
        core: "AbstractHilbertSpace",
        state: Any,
        *,
        validate: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Explain U(1) state -> number conversion for one state or a batch."""
        states_np = _as_numpy(state)
        states2d, was_single = _ensure_2d_states(states_np)
        numbers = _u1_states_to_numbers(core, states_np, validate=validate, **kwargs)
        nums1d, _ = _ensure_1d_numbers(numbers)

        local_states = core.allowed_basis_states
        base = int(local_states.length)
        is_gf = _space_is_gauge_fixed(core)

        order_override = kwargs.get("netket_order", None)
        order = (
            "C" if is_gf else (order_override or _infer_netket_site_order(local_states))
        )

        entries: List[Dict[str, Any]] = []
        for i in range(states2d.shape[0]):
            st = states2d[i]
            if is_gf:
                free_vals = _extract_reduced_free_values(core, st[None, :])[0]
                digits = _local_values_to_digits(local_states, free_vals[None, :])[0]
                entries.append(
                    {
                        "state": st.tolist(),
                        "number": int(nums1d[i]),
                        "base": base,
                        "order": "C",
                        "reduced_free_values": free_vals.tolist(),
                        "reduced_digits": digits.astype(int).tolist(),
                    }
                )
            else:
                digits = _local_values_to_digits(local_states, st[None, :])[0]
                entries.append(
                    {
                        "state": st.tolist(),
                        "number": int(nums1d[i]),
                        "base": base,
                        "order": order,
                        "digits": digits.astype(int).tolist(),
                    }
                )

        metadata = {
            "scheme_name": self.scheme_name,
            "ordering_contract": self.ordering_contract,
            "requires_full_precompute": self.requires_full_precompute,
            "supports_lazy_mode": self.supports_lazy_mode(core),
            "gauge_fixed": is_gf,
            "validate": bool(validate),
            "backend": kwargs.get("backend", "auto"),
        }
        return _finalise_explanation(
            single=was_single, metadata=metadata, entries=entries
        )

    def explain_number_to_state(
        self,
        core: "AbstractHilbertSpace",
        number: Any,
        *,
        validate: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Explain U(1) number -> state conversion for one number or a batch."""
        nums1d, was_scalar = _ensure_1d_numbers(number)
        states = _u1_numbers_to_states(core, number, validate=validate, **kwargs)
        states_np = _as_numpy(states)
        states2d, _ = _ensure_2d_states(states_np)

        local_states = core.allowed_basis_states
        base = int(local_states.length)
        is_gf = _space_is_gauge_fixed(core)

        order_override = kwargs.get("netket_order", None)
        order = (
            "C" if is_gf else (order_override or _infer_netket_site_order(local_states))
        )

        entries: List[Dict[str, Any]] = []
        for i in range(nums1d.shape[0]):
            n = int(nums1d[i])
            st = states2d[i]
            if is_gf:
                n_free = int(core.gauge_dimensions) * int(len(core.gauge_fixing.free))
                digits = _unrank_numbers_base(
                    np.asarray([n], dtype=object), base, n_free, "C"
                )[0]
                free_vals = _local_digits_to_values(local_states, digits[None, :])[0]
                entries.append(
                    {
                        "number": n,
                        "decoded_state": st.tolist(),
                        "base": base,
                        "order": "C",
                        "reduced_digits": digits.astype(int).tolist(),
                        "reduced_free_values": _as_numpy(free_vals)
                        .astype(object)
                        .tolist(),
                    }
                )
            else:
                n_sites = int(core.size)
                digits = _unrank_numbers_base(
                    np.asarray([n], dtype=object), base, n_sites, order
                )[0]
                vals = _local_digits_to_values(local_states, digits[None, :])[0]
                entries.append(
                    {
                        "number": n,
                        "decoded_state": st.tolist(),
                        "base": base,
                        "order": order,
                        "digits": digits.astype(int).tolist(),
                        "decoded_values": _as_numpy(vals).astype(object).tolist(),
                    }
                )

        metadata = {
            "scheme_name": self.scheme_name,
            "ordering_contract": self.ordering_contract,
            "requires_full_precompute": self.requires_full_precompute,
            "supports_lazy_mode": self.supports_lazy_mode(core),
            "gauge_fixed": is_gf,
            "validate": bool(validate),
            "backend": kwargs.get("backend", "auto"),
        }
        return _finalise_explanation(
            single=was_scalar,
            metadata=metadata,
            entries=entries,
        )


class U1UnconstrainedStateEnumerator(_U1StateEnumeratorBase):
    """Enumerator for unconstrained U(1) cores."""

    _SCHEME_NAME = "u1-unconstrained-direct-product"
    _ORDERING_CONTRACT = (
        "Direct-product mixed-radix ranking of local basis digits over all sites. "
        "Digit significance follows inferred NetKet site order ('C' or 'F'), "
        "unless netket_order is explicitly provided."
    )


class U1ConstrainedStateEnumerator(_U1StateEnumeratorBase):
    """Enumerator for constrained (gauge-fixed) U(1) cores."""

    _SCHEME_NAME = "u1-constrained-reduced-free-c-order"
    _ORDERING_CONTRACT = (
        "Gauge-fixed reduced parameterisation. Only free variables are ranked "
        "with C-order mixed-radix significance, concatenated by gauge copy."
    )
