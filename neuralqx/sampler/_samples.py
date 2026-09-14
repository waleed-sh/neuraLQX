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


"""Sample-count and chain-shape helpers."""

from __future__ import annotations

import warnings


def device_count_for_samples(*, sample_sharding: bool, device_count: int) -> int:
    """Number of devices participating in sample-axis work."""
    return int(device_count) if sample_sharding else 1


def resolve_n_chains(
    *,
    n_chains: int | None,
    n_chains_per_device: int | None,
    sample_sharding: bool,
    device_count: int,
) -> int:
    """Resolve total chain count, rounding up to device divisibility when needed."""
    n_devices = device_count_for_samples(
        sample_sharding=sample_sharding,
        device_count=device_count,
    )
    if n_chains is not None and n_chains_per_device is not None:
        raise ValueError("Specify either n_chains or n_chains_per_device, not both.")
    if n_chains_per_device is None and n_chains is None:
        n_chains_per_device = 16
    if n_chains_per_device is not None:
        if n_chains_per_device <= 0:
            raise ValueError("n_chains_per_device must be positive.")
        return int(n_chains_per_device) * n_devices
    if n_chains is None or n_chains <= 0:
        raise ValueError("n_chains must be positive.")
    chains_per_device = max((int(n_chains) + n_devices - 1) // n_devices, 1)
    resolved = chains_per_device * n_devices
    if resolved != int(n_chains):
        warnings.warn(
            f"Using n_chains={resolved} so chains divide evenly across "
            f"{n_devices} sample device(s); requested n_chains={n_chains}.",
            UserWarning,
            stacklevel=3,
        )
    return resolved


def n_chains_per_device(
    *,
    n_chains: int,
    sample_sharding: bool,
    device_count: int,
) -> int:
    """Return chains hosted by each sample device."""
    n_devices = device_count_for_samples(
        sample_sharding=sample_sharding,
        device_count=device_count,
    )
    per_device, remainder = divmod(int(n_chains), n_devices)
    if remainder:
        raise RuntimeError("n_chains must be divisible by sample device count.")
    return per_device


def n_batches(
    *,
    n_chains: int,
    sample_sharding: bool,
    device_count: int,
) -> int:
    """Leading batch size visible to the current process."""
    if sample_sharding:
        return int(n_chains)
    return n_chains_per_device(
        n_chains=int(n_chains),
        sample_sharding=sample_sharding,
        device_count=device_count,
    )


def adjusted_sample_count(n_samples: int, n_chains: int) -> tuple[int, int]:
    """Return ``(adjusted_samples, chain_length)`` nearest-divisible by chains."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_chains <= 0:
        raise ValueError("n_chains must be positive.")
    lower = (int(n_samples) // int(n_chains)) * int(n_chains)
    upper = lower + int(n_chains)
    if lower == 0:
        adjusted = upper
    elif int(n_samples) - lower < upper - int(n_samples):
        adjusted = lower
    else:
        adjusted = upper
    return adjusted, adjusted // int(n_chains)


def chain_length_from_samples(
    n_samples: int,
    n_chains: int,
    *,
    warn: bool = True,
) -> int:
    """Resolve chain length from a total sample request."""
    adjusted, chain_length = adjusted_sample_count(n_samples, n_chains)
    if warn and adjusted != int(n_samples):
        warnings.warn(
            f"n_samples={n_samples} is not divisible by n_chains={n_chains}; "
            f"using n_samples={adjusted} and chain_length={chain_length}.",
            UserWarning,
            stacklevel=3,
        )
    return chain_length


__all__ = [
    "adjusted_sample_count",
    "chain_length_from_samples",
    "device_count_for_samples",
    "n_batches",
    "n_chains_per_device",
    "resolve_n_chains",
]
