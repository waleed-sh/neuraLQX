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
import time


def test_atexit_summary_emits_when_info_or_lower(debug_enabled):
    dbg = debug_enabled

    # record some stats via trace/errors_only
    @dbg.trace(tag="SUM", warn_slow_ms=None)
    def f():
        time.sleep(0.001)
        return 1

    @dbg.errors_only(tag="SUM", reraise=False)
    def g():
        raise RuntimeError("x")

    f()
    g()

    # Enable summary emission: requires verbosity <= INFO
    with dbg._STATE_LOCK:
        st = dbg._RT.settings
        dbg._RT.settings = st.__class__(
            enabled=True,
            verbosity=logging.INFO,
            log_dir=st.log_dir,
            session_id=st.session_id,
            rank=st.rank,
            world=st.world,
            pid=st.pid,
            logfile=st.logfile,
        )

    dbg._atexit_summary()

    tail = "\n".join(list(dbg._EVENT_BUFFER)[-80:])
    assert "=== Debug summary ===" in tail
    assert "=== End debug summary ===" in tail
    assert "calls=" in tail and "errors=" in tail
