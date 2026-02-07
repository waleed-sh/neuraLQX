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


import logging
import re
import jax.numpy as jnp


def test_event_formats_key_value_fields(debug_enabled):
    dbg = debug_enabled

    dbg.event("E", tag="T", level=logging.INFO, a=1, b="x", c=jnp.ones((2, 3)))

    last = list(dbg._EVENT_BUFFER)[-1]
    assert "[T]" in last
    assert "E |" in last
    assert "a=1" in last
    assert "b='x'" in last
    assert "shape=(2, 3)" in last
    assert "dtype=" in last


def test_log_once_only_emits_once(debug_enabled):
    dbg = debug_enabled

    before = len(dbg._EVENT_BUFFER)
    dbg.log_once(logging.INFO, "k1", "hello", tag="ONCE")
    dbg.log_once(logging.INFO, "k1", "hello", tag="ONCE")  # should be ignored
    after = len(dbg._EVENT_BUFFER)

    # only one event appended
    assert after == before + 1
    assert "hello" in list(dbg._EVENT_BUFFER)[-1]


def test_dump_recent_events_replays_buffer(debug_enabled):
    dbg = debug_enabled
    dbg.event("A", tag="X")
    dbg.event("B", tag="X")
    before = len(dbg._EVENT_BUFFER)

    dbg.dump_recent_events(n=2, tag="DUMP")

    after = len(dbg._EVENT_BUFFER)
    # dump emits: header + 2 lines + footer = 4 additional emits
    assert after >= before + 4
    assert any("Recent events" in ln for ln in list(dbg._EVENT_BUFFER)[-10:])
    assert any("End recent events" in ln for ln in list(dbg._EVENT_BUFFER)[-10:])


def test_dump_stacks_emits_header_and_footer(debug_enabled):
    dbg = debug_enabled
    before = len(dbg._EVENT_BUFFER)
    dbg.dump_stacks(tag="STACKS")
    after = len(dbg._EVENT_BUFFER)
    assert after > before
    tail = "\n".join(list(dbg._EVENT_BUFFER)[-30:])
    assert "Thread stack dump" in tail
    assert "End thread stack dump" in tail
