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

from dataclasses import dataclass
from dataclasses import field

from typing import Any
from typing import Dict
from typing import Tuple


@dataclass
class Stats:
    calls: int = 0
    inclusive_ns: int = 0
    exclusive_ns: int = 0
    min_ns: int = 0
    max_ns: int = 0
    flops: float = 0.0
    bytes: float = 0.0

    def update(
        self,
        inclusive_ns: int,
        exclusive_ns: int,
        flops: float = 0.0,
        bytes_: float = 0.0,
    ) -> None:
        self.calls += 1
        self.inclusive_ns += int(inclusive_ns)
        self.exclusive_ns += int(exclusive_ns)
        self.flops += float(flops)
        self.bytes += float(bytes_)
        if self.calls == 1:
            self.min_ns = int(inclusive_ns)
            self.max_ns = int(inclusive_ns)
        else:
            if inclusive_ns < self.min_ns:
                self.min_ns = int(inclusive_ns)
            if inclusive_ns > self.max_ns:
                self.max_ns = int(inclusive_ns)

    def merge(self, other: "Stats") -> None:
        if other.calls == 0:
            return
        if self.calls == 0:
            self.calls = other.calls
            self.inclusive_ns = other.inclusive_ns
            self.exclusive_ns = other.exclusive_ns
            self.min_ns = other.min_ns
            self.max_ns = other.max_ns
            self.flops = other.flops
            self.bytes = other.bytes
            return
        self.calls += other.calls
        self.inclusive_ns += other.inclusive_ns
        self.exclusive_ns += other.exclusive_ns
        self.min_ns = min(self.min_ns, other.min_ns) if other.min_ns else self.min_ns
        self.max_ns = max(self.max_ns, other.max_ns)
        self.flops += other.flops
        self.bytes += other.bytes

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "calls": self.calls,
            "inclusive_ns": self.inclusive_ns,
            "exclusive_ns": self.exclusive_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "flops": self.flops,
            "bytes": self.bytes,
        }
        # derived metrics are computed on demand downstream, keep raw numbers here
        return d


@dataclass
class ProfileNode:
    name: str
    cat: str = ""
    stats: Stats = field(default_factory=Stats)
    children: Dict[Tuple[str, str], "ProfileNode"] = field(default_factory=dict)

    def get_child(self, name: str, cat: str = "") -> "ProfileNode":
        key = (name, cat)
        node = self.children.get(key)
        if node is None:
            node = ProfileNode(name=name, cat=cat)
            self.children[key] = node
        return node

    def merge(self, other: "ProfileNode") -> None:
        self.stats.merge(other.stats)
        for k, child in other.children.items():
            if k not in self.children:
                self.children[k] = ProfileNode(name=child.name, cat=child.cat)
            self.children[k].merge(child)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cat": self.cat,
            "stats": self.stats.to_dict(),
            "children": [c.to_dict() for c in self.children.values()],
        }


@dataclass
class Frame:
    node: ProfileNode
    start_ns: int
    child_inclusive_ns_accum: int = 0
    flops: float = 0.0
    bytes: float = 0.0
