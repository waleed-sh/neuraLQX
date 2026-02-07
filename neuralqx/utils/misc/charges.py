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

import itertools


def generate_charge_vectors(cutoff: int, gauge_dimensions: int = 3) -> list:
    """
    Generate all discrete charge vectors within a symmetric cutoff.

    This helper enumerates the Cartesian product of the allowed charge set
    ``{-cutoff, ..., 0, ..., +cutoff}`` across ``gauge_dimensions`` components. The result is a list
    of tuples, each representing one possible charge vector.

    :param cutoff: Maximum absolute value of each charge component. Components range from
                   ``-cutoff`` to ``+cutoff`` (inclusive).
    :param gauge_dimensions: Number of components (dimensions) in each charge vector. Defaults to 3.
    :returns: List of tuples of length ``gauge_dimensions`` containing all possible charge vectors.
    """

    allowed_charges = range(-cutoff, cutoff + 1)

    combinations = itertools.product(allowed_charges, repeat=gauge_dimensions)

    return list(combinations)
