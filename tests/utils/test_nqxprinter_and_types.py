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


import importlib
import datetime as _dt

import pytest


class FixedDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return _dt.datetime(2020, 1, 2, 3, 4, 5)


class FakeCfg:

    def __init__(self, verbose=True):
        self._verbose = verbose

    def get(self, key):
        if key == "VERBOSE":
            return self._verbose
        return False


class FakeMPI:
    def __init__(self, is_master=True):
        self._is_master = is_master

    def is_global_master(self):
        return self._is_master


class FixedHashFn:
    def __init__(self):
        self.i = 0

    def __call__(self):
        self.i += 1
        return f"UID{self.i:03d}"


class DummyConsole:

    def __init__(self):
        self.printed = []

    def print(self, obj, *args, **kwargs):
        self.printed.append(obj)


class DummyStackEntry:

    def __init__(self, *, filename=None, lineno=None, function=None, frame=None):
        self.filename = filename
        self.lineno = lineno
        self.function = function
        self.frame = frame


class DummyFrame:
    def __init__(self, f_locals, co_name):
        self.f_locals = f_locals
        self.f_code = type("Code", (), {"co_name": co_name})


@pytest.fixture
def tmod():
    return importlib.import_module("neuralqx.utils.io.printing.types")


@pytest.fixture
def pmod():
    return importlib.import_module("neuralqx.utils.io.printing.nqxprinter")


@pytest.fixture
def fixed_time(monkeypatch, tmod, pmod):

    monkeypatch.setattr(tmod.datetime, "datetime", FixedDatetime, raising=True)
    monkeypatch.setattr(pmod.datetime, "datetime", FixedDatetime, raising=True)


@pytest.fixture
def make_printer(monkeypatch, pmod, tmod, fixed_time):

    def _make(*, verbose=True, is_master=True, hash_fn=None):
        monkeypatch.setattr(pmod, "cfg", FakeCfg(verbose=verbose), raising=True)
        monkeypatch.setattr(pmod, "_mpi", FakeMPI(is_master=is_master), raising=True)

        if hash_fn is None:
            hash_fn = FixedHashFn()
        handler = tmod.PrintHandler(hash_function=hash_fn)

        console = DummyConsole()
        printer = pmod.NQXPrinter(logger=handler, console=console)

        assert printer.logger is handler
        assert printer.console is console

        return printer, handler, console

    return _make


def test_printhandler_log_creates_entry_with_timestamp_caller_uid_via_patched_stack(
    tmod, monkeypatch, fixed_time
):

    fake_stack = [
        DummyStackEntry(filename="a", lineno=1, function="f0"),
        DummyStackEntry(filename="b", lineno=2, function="f1"),
        DummyStackEntry(filename="c", lineno=3, function="f2"),
        DummyStackEntry(filename="X.py", lineno=999, function="outer"),
    ]
    monkeypatch.setattr(tmod.inspect, "stack", lambda: fake_stack, raising=True)

    hf = FixedHashFn()
    h = tmod.PrintHandler(hash_function=hf)

    h.log("hello")

    assert len(h.logs) == 1
    e = h.logs[0]
    assert e.message == "hello"
    assert e.uid == "UID001"
    assert e.timestamp == "2020-01-02 03:04:05"
    assert e.caller == "X.py:999 in outer"


def test_printhandler_multiple_logs_append_and_uids_increment(
    tmod, monkeypatch, fixed_time
):
    fake_stack = [DummyStackEntry()] * 3 + [
        DummyStackEntry(filename="X.py", lineno=1, function="f")
    ]
    monkeypatch.setattr(tmod.inspect, "stack", lambda: fake_stack, raising=True)

    hf = FixedHashFn()
    h = tmod.PrintHandler(hash_function=hf)

    h.log("a")
    h.log("b")

    assert [e.message for e in h.logs] == ["a", "b"]
    assert [e.uid for e in h.logs] == ["UID001", "UID002"]


def test_printhandler_get_caller_info_fallback_unknown_when_stack_too_short(
    tmod, monkeypatch
):
    monkeypatch.setattr(tmod.inspect, "stack", lambda: [], raising=True)
    assert tmod.PrintHandler._get_caller_info() == "Unknown Caller"


def test_printentry_default_uid_is_nonempty_string(tmod):

    e = tmod.PrintEntry(timestamp="t", caller="c", message="m")
    assert isinstance(e.uid, str)
    assert len(e.uid) > 0


def test_printentry_explicit_uid_overrides_default(tmod):
    e = tmod.PrintEntry(timestamp="t", caller="c", message="m", uid="EXPLICIT")
    assert e.uid == "EXPLICIT"


def test_printer_uses_injected_console_and_logger(make_printer):
    printer, handler, console = make_printer(verbose=True, is_master=True)
    assert printer.logger is handler
    assert printer.console is console


def test_printer_print_is_noop_on_worker_even_if_verbose(make_printer):
    printer, handler, console = make_printer(verbose=True, is_master=False)

    printer.print("hello worker")

    assert console.printed == []
    assert handler.logs == []


def test_printer_print_is_noop_when_verbose_false(make_printer):
    printer, handler, console = make_printer(verbose=False, is_master=True)

    printer.print("hello quiet")

    assert console.printed == []
    assert handler.logs == []


def test_printer_print_emits_panel_and_logs_when_master_and_verbose(
    make_printer, monkeypatch, pmod
):
    printer, handler, console = make_printer(verbose=True, is_master=True)

    monkeypatch.setattr(
        pmod.NQXPrinter,
        "_get_current_caller_info",
        staticmethod(lambda: "MyClass.my_method()"),
        raising=True,
    )

    printer.print("hello <world>")

    assert len(console.printed) == 1
    panel = console.printed[0]

    assert getattr(panel, "title", None) == "neuraLQX Printer"
    assert "Logged by MyClass.my_method()" in str(getattr(panel, "subtitle", ""))

    assert len(handler.logs) == 1
    e = handler.logs[0]
    assert e.message == "hello <world>"
    assert e.timestamp == "2020-01-02 03:04:05"
    assert isinstance(e.uid, str) and len(e.uid) > 0


def test_create_message_panel_structure(make_printer, monkeypatch, pmod):
    printer, handler, console = make_printer(verbose=True, is_master=True)

    monkeypatch.setattr(
        pmod.NQXPrinter,
        "_get_current_caller_info",
        staticmethod(lambda: "C.f()"),
        raising=True,
    )

    panel = printer._create_message_panel("msg", "2020-01-02 03:04:05")

    assert getattr(panel, "title", None) == "neuraLQX Printer"
    assert "Logged by C.f()" in str(getattr(panel, "subtitle", ""))

    table = getattr(panel, "renderable", None)
    assert table is not None
    assert len(table.rows) >= 2


def test_get_current_caller_info_formats_class_and_function(pmod, monkeypatch):
    class SomeClass:
        pass

    dummy_self = SomeClass()

    stack = [
        DummyStackEntry(frame=DummyFrame({}, "lvl0")),
        DummyStackEntry(frame=DummyFrame({}, "lvl1")),
        DummyStackEntry(frame=DummyFrame({}, "lvl2")),
        DummyStackEntry(frame=DummyFrame({"self": dummy_self}, "method_name")),
    ]
    monkeypatch.setattr(pmod.inspect, "stack", lambda: stack, raising=True)

    s = pmod.NQXPrinter._get_current_caller_info()
    assert s == "SomeClass.method_name()"


def test_get_current_caller_info_unknown_when_stack_too_short(pmod, monkeypatch):
    monkeypatch.setattr(pmod.inspect, "stack", lambda: [1], raising=True)
    assert pmod.NQXPrinter._get_current_caller_info() == "UnknownCaller"


def test_display_logs_noop_on_worker(make_printer):
    printer, handler, console = make_printer(verbose=True, is_master=False)

    printer.display_logs()
    assert console.printed == []


def test_display_logs_shows_no_logs_panel_when_empty(make_printer):
    from rich.panel import Panel

    printer, handler, console = make_printer(verbose=True, is_master=True)

    handler.logs.clear()
    printer.display_logs()

    assert len(console.printed) == 1
    panel = console.printed[0]
    assert isinstance(panel, Panel)

    assert "No logs available" in str(panel.renderable)


def test_display_logs_prints_table_panel_with_entries(make_printer, monkeypatch, tmod):
    printer, handler, console = make_printer(verbose=True, is_master=True)

    fake_stack = [DummyStackEntry()] * 3 + [
        DummyStackEntry(filename="X.py", lineno=1, function="f")
    ]
    monkeypatch.setattr(tmod.inspect, "stack", lambda: fake_stack, raising=True)

    handler.logs.clear()
    handler.log("m1")
    handler.log("m2")

    printer.display_logs()

    assert len(console.printed) == 1
    panel = console.printed[0]
    assert getattr(panel, "title", None) == "Logged Messages"

    table = getattr(panel, "renderable", None)
    assert table is not None
    assert len(table.rows) == 2


def test_create_log_table_contains_expected_columns_and_row_count(
    make_printer, monkeypatch, tmod
):
    printer, handler, console = make_printer(verbose=True, is_master=True)

    fake_stack = [DummyStackEntry()] * 3 + [
        DummyStackEntry(filename="X.py", lineno=1, function="f")
    ]
    monkeypatch.setattr(tmod.inspect, "stack", lambda: fake_stack, raising=True)

    handler.logs.clear()
    handler.log("hello")

    table = printer._create_log_table()
    headers = [c.header for c in table.columns]

    assert headers == ["Timestamp", "Caller", "Message", "UID"]
    assert len(table.rows) == 1


def test_print_records_uid_and_message_via_handler(make_printer, monkeypatch, pmod):
    printer, handler, console = make_printer(verbose=True, is_master=True)

    monkeypatch.setattr(
        pmod.NQXPrinter,
        "_get_current_caller_info",
        staticmethod(lambda: "C.f()"),
        raising=True,
    )

    printer.print("msg")

    assert len(handler.logs) == 1
    e = handler.logs[0]
    assert e.message == "msg"
    assert isinstance(e.uid, str) and len(e.uid) > 0
