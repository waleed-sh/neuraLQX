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


from neuralqx.profile import _rank
from neuralqx.utils.distributed import RuntimeInfo


def _runtime(rank: int, size: int, *, local_rank: int = 0, backend: str = "serial"):
    return RuntimeInfo(
        backend=backend,
        rank=rank,
        size=size,
        local_rank=local_rank,
        jax_available=True,
        multihost_utils_available=True,
    )


def test_serial_backend_when_single_process(monkeypatch):
    monkeypatch.setattr(
        _rank._dist,
        "runtime_info",
        lambda: _runtime(0, 1, backend="serial"),
        raising=True,
    )
    _rank.get_rank_info.cache_clear()

    info = _rank.get_rank_info()
    assert info.backend == "serial"
    assert info.rank == 0 and info.size == 1


def test_jax_backend_when_process_count_gt_1(monkeypatch):
    monkeypatch.setattr(
        _rank._dist, "runtime_info", lambda: _runtime(3, 8, backend="jax"), raising=True
    )
    _rank.get_rank_info.cache_clear()

    info = _rank.get_rank_info()
    assert info.backend == "jax"
    assert info.rank == 3 and info.size == 8


def test_local_rank_env_priority(monkeypatch):
    monkeypatch.setattr(
        _rank._dist,
        "runtime_info",
        lambda: _runtime(0, 1, backend="serial"),
        raising=True,
    )
    _rank.get_rank_info.cache_clear()

    monkeypatch.setenv("SLURM_LOCALID", "7")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")

    info = _rank.get_rank_info()
    assert info.local_rank == 1


def test_get_env_int_invalid_values_ignored(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "not_an_int")
    assert _rank._get_env_int(["LOCAL_RANK"], default=5) == 5
