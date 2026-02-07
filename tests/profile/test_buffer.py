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


import threading
from neuralqx.profile._buffer import TraceBuffer


def test_trace_buffer_bounded_and_snapshot():
    b = TraceBuffer(max_events=3)
    b.append({"i": 0})
    b.append({"i": 1})
    b.append({"i": 2})
    assert len(b) == 3
    assert [e["i"] for e in b.snapshot()] == [0, 1, 2]

    b.append({"i": 3})
    assert len(b) == 3
    assert [e["i"] for e in b.snapshot()] == [1, 2, 3]


def test_trace_buffer_extend():
    b = TraceBuffer(max_events=5)
    b.extend([{"i": 1}, {"i": 2}])
    assert [e["i"] for e in b.snapshot()] == [1, 2]


def test_trace_buffer_thread_safety_smoke():
    b = TraceBuffer(max_events=1000)

    def worker(tid: int):
        for j in range(200):
            b.append({"tid": tid, "j": j})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(b) > 0
    snap = b.snapshot()
    assert all("tid" in e and "j" in e for e in snap)
