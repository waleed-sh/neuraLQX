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


def test_parse_bool_variants(debug_disabled):
    dbg = debug_disabled
    assert dbg._parse_bool(True) is True
    assert dbg._parse_bool(False) is False
    assert dbg._parse_bool("1") is True
    assert dbg._parse_bool("true") is True
    assert dbg._parse_bool("YES") is True
    assert dbg._parse_bool("0") is False
    assert dbg._parse_bool("false") is False
    assert dbg._parse_bool("no") is False
    assert dbg._parse_bool("nonsense", default=True) is True
    assert dbg._parse_bool("nonsense", default=False) is False


def test_parse_level_variants(debug_disabled):
    dbg = debug_disabled
    assert dbg._parse_level("DEBUG") == logging.DEBUG
    assert dbg._parse_level("info") == logging.INFO
    assert dbg._parse_level("WaRnInG") == logging.WARNING
    assert dbg._parse_level(42) == 42
    assert dbg._parse_level("UNKNOWN", default=17) == 17


def test_read_settings_enabled_creates_logfile_path(tmp_path, monkeypatch):
    import neuralqx.debug as dbg

    class Cfg:
        def get(self, k):
            return {"DEBUG": True, "LOG_LEVEL": "INFO"}.get(k)

        def get_static(self, k):
            if k == "Leach Directory":
                return str(tmp_path / "logs")
            raise KeyError(k)

    monkeypatch.setattr(dbg, "_get_cfg", lambda: Cfg(), raising=True)
    monkeypatch.setattr(dbg, "_mpi_rank_world", lambda: (2, 5), raising=True)
    monkeypatch.setattr(dbg, "_short_session_id", lambda: "0123456789", raising=True)

    st = dbg.read_settings()
    assert st.enabled is True
    assert st.verbosity == logging.INFO
    assert st.rank == 2 and st.world == 5
    assert st.session_id == "0123456789"
    assert st.logfile is not None
    assert str(st.logfile).endswith("_rank2_0123456789.log")
    assert st.log_dir.exists()


def test_read_settings_disabled_has_no_logfile(tmp_path, monkeypatch):
    import neuralqx.debug as dbg

    class Cfg:
        def get(self, k):
            return {"DEBUG": False, "LOG_LEVEL": "DEBUG"}.get(k)

        def get_static(self, k):
            if k == "Leach Directory":
                return str(tmp_path / "logs")
            raise KeyError(k)

    monkeypatch.setattr(dbg, "_get_cfg", lambda: Cfg(), raising=True)
    monkeypatch.setattr(dbg, "_mpi_rank_world", lambda: (0, 1), raising=True)
    monkeypatch.setattr(dbg, "_short_session_id", lambda: "0123456789", raising=True)

    st = dbg.read_settings()
    assert st.enabled is False
    assert st.logfile is None


def test_initialise_writes_session_header_and_creates_file(debug_enabled):
    dbg = debug_enabled

    logfile = dbg.get_logfile()
    assert logfile is not None
    assert dbg.is_enabled() is True

    # The session header is emitted through the logger and buffered.
    buf = list(dbg._EVENT_BUFFER)
    assert any("SESSION" in ln and "NeuralQX Debug Session" in ln for ln in buf)

    # Ensure file exists and contains at least one SESSION line.
    # Flush handlers to force write-out.
    with dbg._STATE_LOCK:
        fh = dbg._RT.file_handler
    if fh is not None:
        fh.flush()

    assert logfile.exists()
    text = logfile.read_text(encoding="utf-8")
    assert "SESSION" in text


def test_initialise_disabled_mode_emits_nothing_and_no_logfile(debug_disabled):
    dbg = debug_disabled
    assert dbg.is_enabled() is False
    assert dbg.get_logfile() is None

    # Attempt to emit: should be gated by enabled=False.
    before = len(dbg._EVENT_BUFFER)
    dbg.event("hello", x=1)
    dbg.log_once(logging.INFO, key="k", msg="once")
    dbg.dump_recent_events()
    dbg.dump_stacks()
    after = len(dbg._EVENT_BUFFER)
    assert after == before


def test_refresh_settings_toggles_enabled_without_reinit(tmp_path, monkeypatch):
    import neuralqx.debug as dbg

    class Cfg:
        def __init__(self):
            self.debug = True
            self.level = "DEBUG"

        def get(self, k):
            return {"DEBUG": self.debug, "LOG_LEVEL": self.level}.get(k)

        def get_static(self, k):
            if k == "Leach Directory":
                return str(tmp_path / "logs")
            raise KeyError(k)

    cfg = Cfg()
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
    assert dbg.is_enabled() is True

    cfg.debug = False
    cfg.level = "ERROR"
    dbg.refresh_settings(reinit=False)
    assert dbg.is_enabled() is False

    # turning back on should re-enable emission with same logfile
    logfile_before = dbg.get_logfile()
    cfg.debug = True
    cfg.level = "INFO"
    dbg.refresh_settings(reinit=False)
    assert dbg.is_enabled() is True
    assert dbg.get_logfile() == logfile_before
