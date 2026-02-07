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


import atexit
from pathlib import Path
import pytest

import neuralqx.debug as dbg


class DummyCfg:
    """
    Minimal cfg stand-in for neuralqx.cfg used by neuralqx.debug.

    Supports:
      - get(key): returns dynamic value (DEBUG / LOG_LEVEL)
      - get_static(key): returns log directory (the code uses the key "Leach Directory")
    """

    def __init__(self, *, debug: bool, log_level: str, log_dir: Path | None):
        self._vals = {"DEBUG": debug, "LOG_LEVEL": log_level}
        self._log_dir = log_dir

    def get(self, key: str):
        return self._vals.get(key)

    def get_static(self, key: str):
        if key == "Leach Directory":
            if self._log_dir is None:
                raise KeyError(key)
            return str(self._log_dir)
        raise KeyError(key)


def _reset_debug_state():
    # Close and remove handlers to avoid leaking file descriptors across tests.
    with dbg._STATE_LOCK:
        lg = dbg._RT.logger
        if lg is not None:
            for h in list(lg.handlers):
                try:
                    lg.removeHandler(h)
                except Exception:
                    pass
                try:
                    h.close()
                except Exception:
                    pass

        dbg._RT.initialised = False
        dbg._RT.logger = None
        dbg._RT.settings = None
        dbg._RT.file_handler = None
        dbg._RT.buffer_handler = None
        dbg._RT.hooks_installed = False
        dbg._RT.warnings_installed = False
        dbg._RT.faulthandler_enabled = False

    # reset global buffers/counters/stats
    dbg._EVENT_BUFFER.clear()
    dbg._ONCE_KEYS.clear()
    with dbg._STATS_LOCK:
        dbg._STATS_CALLS.clear()
        dbg._STATS_ERRORS.clear()
        dbg._STATS_TOTAL_NS.clear()
        dbg._STATS_MAX_NS.clear()

    with dbg._CALL_SEQ_LOCK:
        dbg._CALL_SEQ = 0

    # reset contextvars for this thread
    dbg._CALL_DEPTH.set(0)
    dbg._CALL_ID.set("")
    dbg._CALL_FUNC.set("")
    dbg._CALL_TAG.set("")


@pytest.fixture(autouse=True)
def _isolate_debug_state():
    _reset_debug_state()
    yield
    _reset_debug_state()


@pytest.fixture
def debug_enabled(tmp_path, monkeypatch):
    """
    Initialise debug enabled with a deterministic session id and patched hook installers.
    Returns the debug module.
    """
    cfg = DummyCfg(debug=True, log_level="DEBUG", log_dir=tmp_path / "logs")

    monkeypatch.setattr(dbg, "_get_cfg", lambda: cfg, raising=True)
    monkeypatch.setattr(dbg, "_mpi_rank_world", lambda: (0, 1), raising=True)
    monkeypatch.setattr(dbg, "_short_session_id", lambda: "abcdef0123", raising=True)

    # Don't actually install global hooks in unit tests.
    monkeypatch.setattr(dbg, "_install_hooks_best_effort", lambda: None, raising=True)
    monkeypatch.setattr(
        dbg, "_install_warning_hook_best_effort", lambda: None, raising=True
    )
    monkeypatch.setattr(
        dbg, "_enable_faulthandler_best_effort", lambda: None, raising=True
    )

    registered = []
    monkeypatch.setattr(
        atexit, "register", lambda fn: registered.append(fn), raising=True
    )

    logfile = dbg.initialise(force=True)
    assert logfile is not None
    return dbg


@pytest.fixture
def debug_disabled(tmp_path, monkeypatch):
    cfg = DummyCfg(debug=False, log_level="INFO", log_dir=tmp_path / "logs")
    monkeypatch.setattr(dbg, "_get_cfg", lambda: cfg, raising=True)
    monkeypatch.setattr(dbg, "_mpi_rank_world", lambda: (0, 1), raising=True)
    monkeypatch.setattr(dbg, "_short_session_id", lambda: "abcdef0123", raising=True)
    monkeypatch.setattr(dbg, "_install_hooks_best_effort", lambda: None, raising=True)
    monkeypatch.setattr(
        dbg, "_install_warning_hook_best_effort", lambda: None, raising=True
    )
    monkeypatch.setattr(
        dbg, "_enable_faulthandler_best_effort", lambda: None, raising=True
    )
    dbg.initialise(force=True)
    return dbg


def last_events(n=10):
    return list(dbg._EVENT_BUFFER)[-n:]
