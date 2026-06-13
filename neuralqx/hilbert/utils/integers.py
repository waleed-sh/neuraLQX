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

import numpy as np

INDEX_LIMIT = int(np.iinfo(np.int32).max)


def product(values: tuple[int, ...]) -> int:
    """Multiplies integer factors into an arbitrary-size Python integer.

    Args:
        values: Integer factors to multiply.

    Returns:
        Exact Python integer product. The calculation intentionally avoids
        fixed-width NumPy integer overflow for very large Hilbert dimensions.
    """
    out = 1
    for value in values:
        out *= int(value)
    return out


__all__ = ["INDEX_LIMIT", "product"]
