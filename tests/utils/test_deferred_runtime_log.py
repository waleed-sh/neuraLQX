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

import json

import pytest

from neuralqx.utils.io.runtime_loggers import DeferredRuntimeLog


def _item(mean: float, sigma: float = 0.0) -> dict:
    return {"Constraint": {"Mean": mean, "Sigma": sigma}}


def test_deferred_runtime_log_buffers_items_when_enabled():
    log = DeferredRuntimeLog()
    log.set_deferred_accumulation(True)

    log(0, _item(1.0, 0.1))
    log(1, _item(2.0, 0.2))

    assert len(log._pending_steps) == 2
    assert len(log._pending_items) == 2
    assert "Constraint" not in log._data


def test_deferred_runtime_log_flush_materializes_histories():
    log = DeferredRuntimeLog()
    log.set_deferred_accumulation(True)

    log(0, _item(1.0, 0.1))
    log(1, _item(2.0, 0.2))

    log.flush()

    assert len(log._pending_steps) == 0
    assert len(log._pending_items) == 0
    assert "Constraint" in log.data
    assert log.data["Constraint"]["Mean"].iters.tolist() == [0, 1]
    assert log.data["Constraint"]["Mean"].value.tolist() == pytest.approx([1.0, 2.0])


def test_deferred_runtime_log_getitem_triggers_materialization():
    log = DeferredRuntimeLog()
    log.set_deferred_accumulation(True)
    log(3, _item(5.0, 0.5))

    mean_hist = log["Constraint"]["Mean"]

    assert len(log._pending_steps) == 0
    assert mean_hist.iters.tolist() == [3]
    assert mean_hist.value.tolist() == pytest.approx([5.0])


def test_deferred_runtime_log_disabling_deferred_mode_materializes_pending():
    log = DeferredRuntimeLog()
    log.set_deferred_accumulation(True)
    log(0, _item(1.0, 0.1))

    log.set_deferred_accumulation(False)

    assert len(log._pending_steps) == 0
    assert "Constraint" in log.data
    assert log.data["Constraint"]["Mean"].iters.tolist() == [0]


def test_deferred_runtime_log_serialize_materializes_pending(tmp_path):
    log = DeferredRuntimeLog()
    log.set_deferred_accumulation(True)
    log(0, _item(1.0, 0.1))

    out = tmp_path / "runtime_snapshot"
    log.serialize(str(out))

    out_file = tmp_path / "runtime_snapshot.json"
    assert out_file.exists()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert "Constraint" in payload
    assert payload["Constraint"]["Mean"]["iters"] == [0]
