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


import os
import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    for k in list(os.environ.keys()):
        if k.startswith("NQX_PROFILE") or k.startswith("NQX_"):
            monkeypatch.delenv(k, raising=False)

    monkeypatch.setenv("NQX_PROFILE_DIR", str(tmp_path / "profiles"))
    yield


@pytest.fixture(autouse=True)
def _disable_project_cfg(monkeypatch):
    try:
        from neuralqx.profile import _cfg

        monkeypatch.setattr(_cfg, "_try_import_cfg", lambda: None, raising=True)
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    try:
        from neuralqx.profile import _rank

        _rank.get_rank_info.cache_clear()
    except Exception:
        pass

    try:
        from neuralqx.profile import _metrics

        _metrics.stop_metrics()
    except Exception:
        pass

    try:
        from neuralqx.profile import profiler as prof_mod

        prof_mod._PROFILER = None
    except Exception:
        pass

    try:
        from neuralqx.profile import _jax

        _jax._JAX = None
    except Exception:
        pass

    try:
        from neuralqx.profile import _nvtx

        _nvtx._NVTX = None
        _nvtx._NVTX_KIND = None
    except Exception:
        pass

    yield

    try:
        from neuralqx.profile import _metrics

        _metrics.stop_metrics()
    except Exception:
        pass
