#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from neuralqx.profile._events import Stats, ProfileNode


def test_stats_update_min_max_and_sums():
    s = Stats()
    s.update(inclusive_ns=10, exclusive_ns=7, flops=1.0, bytes_=2.0)
    assert s.calls == 1
    assert s.inclusive_ns == 10
    assert s.exclusive_ns == 7
    assert s.min_ns == 10
    assert s.max_ns == 10
    assert s.flops == 1.0
    assert s.bytes == 2.0

    s.update(inclusive_ns=5, exclusive_ns=4, flops=0.5, bytes_=1.0)
    assert s.calls == 2
    assert s.inclusive_ns == 15
    assert s.exclusive_ns == 11
    assert s.min_ns == 5
    assert s.max_ns == 10
    assert s.flops == 1.5
    assert s.bytes == 3.0


def test_stats_merge_semantics():
    a = Stats()
    b = Stats()
    a.merge(b)
    assert a.calls == 0

    b.update(inclusive_ns=10, exclusive_ns=6, flops=1.0, bytes_=2.0)
    b.update(inclusive_ns=4, exclusive_ns=2, flops=3.0, bytes_=5.0)
    a.merge(b)
    assert a.calls == 2
    assert a.inclusive_ns == 14
    assert a.exclusive_ns == 8
    assert a.min_ns == 4
    assert a.max_ns == 10
    assert a.flops == 4.0
    assert a.bytes == 7.0


def test_profile_node_get_child_identity_and_merge():
    r1 = ProfileNode(name="root")
    c1a = r1.get_child("A", "cat1")
    c1b = r1.get_child("A", "cat1")
    assert c1a is c1b

    c1a.stats.update(inclusive_ns=10, exclusive_ns=8)
    c1a.get_child("B", "cat2").stats.update(inclusive_ns=3, exclusive_ns=3)

    r2 = ProfileNode(name="root")
    c2a = r2.get_child("A", "cat1")
    c2a.stats.update(inclusive_ns=7, exclusive_ns=7)
    c2a.get_child("C", "cat3").stats.update(inclusive_ns=2, exclusive_ns=1)

    r1.merge(r2)

    a = r1.get_child("A", "cat1").stats
    assert a.calls == 2
    assert a.inclusive_ns == 17

    a_node = r1.get_child("A", "cat1")
    assert ("B", "cat2") in [(n.name, n.cat) for n in a_node.children.values()]
    assert ("C", "cat3") in [(n.name, n.cat) for n in a_node.children.values()]

    d = r1.to_dict()
    assert d["name"] == "root"
    assert isinstance(d["children"], list)
