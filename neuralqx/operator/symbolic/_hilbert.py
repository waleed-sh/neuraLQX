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


"""Native local-domain metadata shared by analysis and JAX lowering."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import jax
from neuralqx.hilbert.local import LocalRange
from neuralqx.utils.errors import NeuralqxTypeError
from neuralqx.utils.errors import NeuralqxValueError


def uniform_local_values(hilbert, *, max_values=65536):
    """Return a small shared basis for proofs, or None when it is unavailable.

    Missing metadata disables a specialization. It never changes the kernel's
    interpretation of a heterogeneous or large Hilbert space.
    """
    spaces = getattr(hilbert, "local_spaces", None)
    if spaces is None:
        return getattr(hilbert, "local_states", None)
    if not spaces or spaces[0].local_size > max_values:
        return None
    if any(space != spaces[0] for space in spaces[1:]):
        return None
    return np.asarray(spaces[0].all_values())


def domain_descriptor(space):
    """Describe a local basis without materializing arithmetic ranges."""
    if isinstance(space, LocalRange):
        return (
            "range",
            np.asarray(space.start).item(),
            np.asarray(space.step).item(),
            int(space.local_size),
        )
    values = np.asarray(
        space.values if hasattr(space, "values") else space.all_values()
    )
    return ("explicit", tuple(values.tolist()))


def illegal_local_positions(hilbert, state):
    """Check host diagnostic samples without dispatching one JAX op per site."""
    invalid = []
    for index, (space, value) in enumerate(
        zip(hilbert.local_spaces, np.asarray(state).reshape(-1), strict=False)
    ):
        domain = domain_descriptor(space)
        if domain[0] == "range":
            _, start, step, count = domain
            if np.issubdtype(space.dtype, np.integer):
                offset = int(value) - start
                position = offset // step
                valid = (
                    value == int(value) and offset % step == 0 and 0 <= position < count
                )
            else:
                position = (value - start) / step
                valid = (
                    np.isfinite(position)
                    and 0 <= round(position) < count
                    and np.isclose(value, start + step * round(position))
                )
        else:
            valid = np.any(np.asarray(domain[1]) == value)
        if not valid:
            invalid.append(index)
    return tuple(invalid)


def shift_mod_metadata(hilbert):
    """Describe cyclic local-index shifts for every native flat site."""
    spaces = getattr(hilbert, "local_spaces", None)
    if spaces is None:
        raise NeuralqxTypeError("Modular shifts require native per-site local spaces.")
    domains = tuple(domain_descriptor(space) for space in spaces)
    if not domains:
        raise NeuralqxValueError("Modular shifts require a nonempty Hilbert space.")
    first = domains[0]
    if (
        all(domain == first for domain in domains)
        and first[0] == "range"
        and first[2] == 1
        and first[1] == int(first[1])
    ):
        return {
            "shift_mod_spec": {
                "version": "uniform_integer_wrap_v1",
                "state_min": int(first[1]),
                "mod_span": first[3],
            }
        }
    return {"shift_mod_spec": {"version": "site_domains_v2", "domains": domains}}


@dataclass(frozen=True)
class SiteWrapDomains:
    """Static per-site bases used by the reference JAX lowering."""

    domains: tuple

    def shift(self, site, value, delta):
        unique = tuple(dict.fromkeys(self.domains))
        groups = tuple(unique.index(domain) for domain in self.domains)

        def branch(domain):
            def apply(args):
                value, delta = args
                if domain[0] == "range":
                    _, start, step, count = domain
                    if jnp.issubdtype(value.dtype, jnp.integer):
                        index = (value.astype(jnp.int64) - start) // step
                    else:
                        index = jnp.rint((value - start) / step).astype(jnp.int64)
                    target = (index + jnp.asarray(delta, dtype=index.dtype)) % count
                    return (start + step * target).astype(value.dtype)
                values = jnp.asarray(domain[1], dtype=value.dtype)
                index = jnp.argmax(values == value)
                target = (index + jnp.asarray(delta, dtype=index.dtype)) % len(
                    domain[1]
                )
                return values[target]

            return apply

        if len(unique) == 1:
            return branch(unique[0])((value, delta))
        group = jnp.asarray(groups, dtype=jnp.int32)[site]
        return jax.lax.switch(
            group, tuple(branch(domain) for domain in unique), (value, delta)
        )


def resolve_wrap_metadata(metadata):
    spec = metadata["shift_mod_spec"]
    if spec["version"] == "uniform_integer_wrap_v1":
        return int(spec["state_min"]), int(spec["mod_span"])
    if spec["version"] == "site_domains_v2":
        return SiteWrapDomains(tuple(spec["domains"])), 1
    raise NeuralqxValueError(
        f"Unsupported shift_mod_spec version: {spec['version']!r}."
    )
