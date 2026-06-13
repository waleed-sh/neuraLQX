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


from __future__ import annotations

from collections.abc import Hashable
from collections.abc import Iterable


def normalize_labels(
    labels: Iterable[Hashable] | Hashable | int | None,
) -> tuple[Hashable, ...]:
    """Normalizes a label declaration to a tuple of labels.

    Args:
        labels: ``None``, an integer multiplicity, a single hashable label, or
            an iterable of labels.

    Returns:
        Tuple of normalized labels.

    Raises:
        ValueError: If an integer multiplicity is negative.
    """
    if labels is None:
        return ()
    if isinstance(labels, int):
        if labels < 0:
            raise ValueError("label multiplicity must be non-negative.")
        return tuple(range(labels))
    if isinstance(labels, str | bytes):
        return (labels,)
    try:
        return tuple(labels)  # type: ignore[arg-type]
    except TypeError:
        return (labels,)  # type: ignore[return-value]


__all__ = ["normalize_labels"]
