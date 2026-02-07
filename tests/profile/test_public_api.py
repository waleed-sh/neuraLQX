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


def test_public_api_symbols_exist():
    import neuralqx.profile as p

    assert hasattr(p, "Profiler")
    assert hasattr(p, "get_profiler")
    assert hasattr(p, "profile")
    assert hasattr(p, "section")
    assert hasattr(p, "step")
    assert hasattr(p, "enabled")
    assert hasattr(p, "flush")


def test_public_flush_safe_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("NQX_PROFILE", "0")
    monkeypatch.setenv("NQX_PROFILE_DIR", str(tmp_path / "profiles"))

    import neuralqx.profile as p

    p.flush()
