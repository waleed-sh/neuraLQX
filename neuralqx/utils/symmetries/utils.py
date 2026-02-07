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

from typing import List, Iterable, Dict
import itertools


def _product_of_bucket_permutations(
    buckets: List[List[int]],
) -> Iterable[List[List[int]]]:
    """
    Yield all choices of a permutation within each bucket. For buckets [B1, B2, ...] yield
    [Perm(B1), Perm(B2), ...]
    """

    # gather all permutations per bucket
    per_bucket_perms = [list(itertools.permutations(b)) for b in buckets]

    # set off the generator
    for choice in itertools.product(*per_bucket_perms):
        yield [list(p) for p in choice]


def _vmap_name(vmap: Dict[int, int]) -> str:
    """
    Build cycle notation for vertex permutations (disjoint cycles)
    """

    cycles = []

    seen = set()

    for v in sorted(vmap):

        if v in seen or vmap[v] == v:
            seen.add(v)
            continue

        cur = v
        cyc = [cur]
        seen.add(cur)

        while vmap[cur] not in seen:
            cur = vmap[cur]
            cyc.append(cur)
            seen.add(cur)

        if len(cyc) > 1:
            cycles.append(tuple(cyc))

    if not cycles:
        return "auto_id"

    return "auto_" + "_".join("(" + ",".join(map(str, c)) + ")" for c in cycles)
