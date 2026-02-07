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
import json
import datetime as _dt

import pytest


class FakePrinter:
    def __init__(self):
        self.messages = []

    def print(self, msg):
        self.messages.append(msg)


class FakeAuthenticator:
    def generate_key_pair(self):
        return "PRK", "PUK", "PEM_PUBLIC_KEY"

    def sign_data(self, prk, data_str):
        assert prk == "PRK"
        return "SIG::" + str(hash(data_str))


class FixedDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return _dt.datetime(2020, 1, 2, 3, 4, 5)


class DummyConsole:
    def __init__(self):
        self.calls = []

    def print(self, obj):
        self.calls.append(obj)


def _set_mpi_master(lmod, monkeypatch, is_master: bool):
    monkeypatch.setattr(lmod._mpi, "is_global_master", lambda: is_master, raising=True)
    monkeypatch.setattr(lmod._mpi, "barrier", lambda: None, raising=True)


@pytest.fixture
def lmod():
    return importlib.import_module("neuralqx.utils.io.loggers.logger")


@pytest.fixture
def logger_factory(lmod, monkeypatch):
    monkeypatch.setattr(lmod, "__version__", "TESTVER", raising=True)
    monkeypatch.setattr(lmod, "datetime", FixedDatetime, raising=True)

    monkeypatch.setattr(lmod, "Authenticator", FakeAuthenticator, raising=True)
    monkeypatch.setattr(lmod, "NQXPrinter", FakePrinter, raising=True)

    dummy_console = DummyConsole()
    monkeypatch.setattr(lmod, "console", dummy_console, raising=True)

    def _make(seed="SEED", is_master=True, solver_seed="SOLVER_SEED"):
        _set_mpi_master(lmod, monkeypatch, is_master)
        lg = lmod.Logger(random_seed=seed, solver_seed=solver_seed)
        return lg, lmod

    return _make


def _extract_json_script(html: str, script_id: str) -> str:
    needle = f'<script id="{script_id}" type="application/json">'
    start = html.find(needle)
    assert start != -1, f"Missing JSON script tag: {script_id}"
    start += len(needle)
    end = html.find("</script>", start)
    assert end != -1, f"Unclosed script tag: {script_id}"
    return html[start:end]


def _assert_has_interactive_shell(html: str):
    assert 'class="nqx-shell"' in html
    assert 'id="nqx-app"' in html
    assert 'id="nqx-nav"' in html
    assert 'id="nqx-tbody"' in html
    assert 'id="nqx-section-search"' in html
    assert 'id="nqx-row-search"' in html

    assert 'id="nqx-foot-meta"' in html

    assert ":root" in html
    assert "--bg:" in html
    assert "--accent:" in html

    assert "(function()" in html or "(function ()" in html


def test_escape_html_and_format_value_escape(lmod):
    s = "<tag>&\"'"
    esc = lmod.escape_html(s)
    assert "&lt;tag&gt;" in esc
    assert "&amp;" in esc
    assert "&quot;" in esc

    out = lmod.format_value(s)
    assert out == esc


def test_dict_to_html_new_format_contains_shell_and_embeds_json(lmod, monkeypatch):
    class FakeStats:
        pass

    monkeypatch.setattr(lmod, "Stats", FakeStats, raising=True)

    data = {
        "Section <A>": {
            "k1": 123,
            "k2": 1.25,
            "k3": "x<y",
            "k4": [1, "a<b", 3],
            "k5": FakeStats(),
            "nested": {"child": "v<z"},
        }
    }

    html = lmod.dict_to_html(
        data, meta={"timestamp": "T", "hash": "H", "seed": "S", "version": "V"}
    )
    _assert_has_interactive_shell(html)

    data_json = _extract_json_script(html, "nqx-data")
    meta_json = _extract_json_script(html, "nqx-meta")

    payload = json.loads(data_json)
    meta = json.loads(meta_json)

    assert "Section <A>" in payload
    assert "k1" in payload["Section <A>"]

    assert payload["Section <A>"]["k3"] == "x<y"
    assert payload["Section <A>"]["k4"] == [1, "a<b", 3]
    assert "nested" in payload["Section <A>"]

    assert meta["timestamp"] == "T"
    assert meta["hash"] == "H"
    assert meta["seed"] == "S"
    assert meta["version"] == "V"

    assert "Section &lt;A&gt;" in html

    assert "x&lt;y" in html
    assert "v&lt;z" in html


def test_dict_to_html_nested_dict_value_should_be_present_in_json(lmod):
    data = {"Sec": {"parent": {"child": "val"}}}
    html = lmod.dict_to_html(data, meta={})
    payload = json.loads(_extract_json_script(html, "nqx-data"))
    assert payload["Sec"]["parent"]["child"] == "val"


def test_logger_default_schema_present(logger_factory):
    lg, _ = logger_factory(seed="S", is_master=True)
    log = lg.get_log()
    assert "Network Configs" in log
    assert "MPI" in log
    assert isinstance(log["Network Configs"], dict)


def test_add_field_inserts_under_parent_master(logger_factory):
    lg, _ = logger_factory(is_master=True)
    lg.add_field("Network Configs", "My Extra Field")
    assert "My Extra Field" in lg.get_log()["Network Configs"]
    assert lg.get_log()["Network Configs"]["My Extra Field"] is None


def test_add_field_nonexistent_parent_raises(logger_factory):
    lg, _ = logger_factory(is_master=True)
    with pytest.raises(KeyError, match=r"entry in the logger was not found"):
        lg.add_field("DOES_NOT_EXIST", "x")


def test_add_field_noop_on_worker(logger_factory):
    lg, _ = logger_factory(is_master=False)
    lg.add_field("Network Configs", "Worker Field")
    assert "Worker Field" not in lg.get_log()["Network Configs"]


def test_log_single_value_sets_field_master(logger_factory):
    lg, _ = logger_factory(is_master=True)
    lg.log("Network type", "Transformer<1>")
    assert lg.get_value_of("Network type") == "Transformer<1>"


def test_log_batch_sets_multiple_fields_master(logger_factory):
    lg, _ = logger_factory(is_master=True)
    keys = ["Network type", "Learning rate"]
    vals = ["CNN", 0.001]
    lg.log(keys, vals)
    assert lg.get_value_of("Network type") == "CNN"
    assert lg.get_value_of("Learning rate") == 0.001


def test_log_batch_length_mismatch_raises(logger_factory):
    lg, _ = logger_factory(is_master=True)
    with pytest.raises(ValueError, match="unequal amount of attributes and values"):
        lg.log(["a", "b"], [1])


def test_log_invalid_mixed_types_raises(logger_factory):
    lg, _ = logger_factory(is_master=True)
    with pytest.raises(ValueError, match="must either be lists.*or single items"):
        lg.log("Network type", ["CNN"])


def test_log_unknown_key_raises_keyerror_with_hint(logger_factory):
    lg, _ = logger_factory(is_master=True)
    with pytest.raises(KeyError, match=r"Try using addField"):
        lg.log("Totally Unknown Key", 1)


def test_log_noop_on_worker_even_for_unknown_key(logger_factory):
    lg, _ = logger_factory(is_master=False)
    lg.log("Totally Unknown Key", 1)


def test_get_value_of_parent_and_leaf_and_missing(logger_factory):
    lg, _ = logger_factory(is_master=True)
    parent = lg.get_value_of("Network Configs")
    assert isinstance(parent, dict)

    lg.log("Network type", "RBM")
    assert lg.get_value_of("Network type") == "RBM"

    with pytest.raises(KeyError, match="was not found in the logger"):
        lg.get_value_of("does-not-exist")


def test_sanitize_recursively_removes_none_and_empty_children(logger_factory):
    lg, _ = logger_factory(is_master=True)
    d = {"A": None, "B": {"B1": None, "B2": 3}, "C": {"C1": None}, "D": 0}
    clean = lg.sanitize(d)
    assert "A" not in clean
    assert clean["B"] == {"B2": 3}
    assert "C" not in clean
    assert clean["D"] == 0


def test_erase_parent_content_resets_original_removes_dynamic(logger_factory):
    lg, _ = logger_factory(is_master=True)
    lg.add_field("Network Configs", "Dynamic")
    lg.log(["Network type", "Dynamic"], ["X", 999])

    assert lg.get_value_of("Network type") == "X"
    assert lg.get_value_of("Dynamic") == 999

    lg.erase_parent_content("Network Configs")
    assert lg.get_log()["Network Configs"]["Network type"] is None
    assert "Dynamic" not in lg.get_log()["Network Configs"]


def test_erase_parent_content_noop_on_worker_even_if_parent_missing(logger_factory):
    lg, _ = logger_factory(is_master=False)
    lg.erase_parent_content("DOES_NOT_EXIST")


def test_display_log_master_prints_panel_and_sanitizes(logger_factory):
    lg, lmod = logger_factory(seed="S1", is_master=True)

    lg.log("Network type", "T<0>")
    lg.display_log()

    assert len(lmod.console.calls) == 1
    panel = lmod.console.calls[0]

    title = getattr(panel, "title", "")
    assert "neuraLQX Output Log" in title
    assert "hash:S1" in title
    assert "TESTVER" in title

    sanitized = lg.get_log()
    assert "Network Configs" in sanitized
    assert sanitized["Network Configs"]["Network type"] == "T<0>"


def test_display_log_worker_no_output(logger_factory):
    lg, lmod = logger_factory(is_master=False)

    lg.log("Network type", "X")
    lg.display_log()
    assert len(lmod.console.calls) == 0


def test_write_log_to_file_master_creates_signed_html_and_contains_app_shell(
    logger_factory, tmp_path
):
    lg, _ = logger_factory(seed="S2", is_master=True)

    lg.log("Network type", 'A<B&"')
    lg.write_log_to_file(str(tmp_path))

    expected = tmp_path / "S2_ModelData_02012020_030405.html"
    assert expected.exists()

    content = expected.read_text(encoding="utf-8")

    assert "<!DOCTYPE html>" in content
    _assert_has_interactive_shell(content)

    assert "SIG::" in content
    assert "PEM_PUBLIC_KEY" in content

    payload = json.loads(_extract_json_script(content, "nqx-data"))
    found = False
    for section in payload.values():
        if isinstance(section, dict) and section.get("Network type") == 'A<B&"':
            found = True
            break
    assert found, "Embedded JSON should contain the logged Network type value"

    assert "A&lt;B&amp;&quot;" in content

    assert any("Data exported to disk." in m for m in lg._p.messages)


def test_write_log_to_file_worker_should_not_create_file_or_print(
    logger_factory, tmp_path
):
    lg, _ = logger_factory(seed="S3", is_master=False)

    lg.write_log_to_file(str(tmp_path))

    assert not any(p.name.startswith("S3_ModelData_") for p in tmp_path.iterdir())
    assert lg._p.messages == []
