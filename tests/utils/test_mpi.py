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


import os
import importlib
from types import SimpleNamespace

import pytest

import numpy as np

if os.environ.get("NQX_MPI_TESTS", "0") != "1":
    pytest.skip(
        "MPI tests are disabled by default. Set NQX_MPI_TESTS=1 to enable.",
        allow_module_level=True,
    )

core = importlib.import_module("neuralqx.utils.mpi.core")
if not getattr(core, "available", False):
    pytest.skip(
        "MPI not enabled/available in neuralqx (core.available is False). "
        "Enable MPI in config/env and ensure mpi4py is installed.",
        allow_module_level=True,
    )

helpers = importlib.import_module("neuralqx.utils.mpi.helpers")
primitives = importlib.import_module("neuralqx.utils.mpi.primitives")


class _MPISpy:
    SUM = "SUM"
    PROD = "PROD"
    MAX = "MAX"
    LOR = "LOR"
    LAND = "LAND"

    @staticmethod
    def Get_version():
        return (3, 1)

    @staticmethod
    def Get_library_version():
        return "FakeMPI 0.0"


class _CommSpy:

    def __init__(self, size: int = 4, rank: int = 0):
        self._size = int(size)
        self._rank = int(rank)
        self.barrier_called = False
        self.last_bcast_root = None
        self.last_scatter_arg = None

    def Get_size(self):
        return self._size

    def Get_rank(self):
        return self._rank

    def Barrier(self):
        self.barrier_called = True
        return None

    def Allreduce(self, sendbuf, recvbuf, op=None):
        sb = np.asarray(sendbuf)
        if op == _MPISpy.SUM:
            recvbuf[...] = sb * self._size
        elif op == _MPISpy.PROD:
            recvbuf[...] = sb**self._size
        elif op == _MPISpy.MAX:
            recvbuf[...] = sb
        elif op == _MPISpy.LOR:
            recvbuf[...] = sb.astype(bool)
        elif op == _MPISpy.LAND:
            recvbuf[...] = sb.astype(bool)
        else:
            recvbuf[...] = sb

    def Allgather(self, sendbuf, recvbuf):
        sb = np.asarray(sendbuf)
        tiled = np.broadcast_to(sb, (self._size,) + sb.shape)
        recvbuf[...] = tiled

    def Gather(self, sendbuf, recvbuf, root=0):
        sb = np.asarray(sendbuf)
        if recvbuf is None:
            return
        tiled = np.broadcast_to(sb, (self._size,) + sb.shape)
        recvbuf[...] = tiled

    def bcast(self, x, root=0):
        self.last_bcast_root = root
        return x

    def allgather(self, x):
        return [x] * self._size

    def gather(self, x, root=0):
        return [x] * self._size if self._rank == root else None

    def scatter(self, xs, root=0):
        self.last_scatter_arg = xs
        if xs is None:
            return "<recv>"
        return xs[self._rank]


@pytest.fixture
def mpi_core():
    return core


@pytest.fixture
def mpi_helpers():
    return helpers


@pytest.fixture
def mpi_primitives():
    return primitives


@pytest.fixture
def simulated_collectives(monkeypatch):

    comm = _CommSpy(size=4, rank=0)

    monkeypatch.setattr(primitives, "MPI", _MPISpy, raising=True)
    monkeypatch.setattr(primitives, "comm", comm, raising=True)
    monkeypatch.setattr(primitives, "n_nodes", 4, raising=True)
    monkeypatch.setattr(primitives, "rank", 0, raising=True)

    return comm


@pytest.mark.mpi
def test_core_public_surface_exports(mpi_core):
    for name in [
        "MPI",
        "comm",
        "available",
        "n_nodes",
        "rank",
        "node_number",
        "is_master",
        "barrier",
        "print0",
        "hostname",
        "all_hostnames",
        "ranks_per_node",
        "local_ranks",
        "detect_cpus_per_task",
    ]:
        assert hasattr(mpi_core, name), f"core missing {name}"


@pytest.mark.mpi
def test_core_rank_and_size_consistent(mpi_core):
    assert mpi_core.n_nodes >= 1
    assert 0 <= mpi_core.rank < mpi_core.n_nodes
    assert mpi_core.comm.Get_size() == mpi_core.n_nodes
    assert mpi_core.comm.Get_rank() == mpi_core.rank


@pytest.mark.mpi
def test_is_master_depends_on_rank(monkeypatch, mpi_core):
    monkeypatch.setattr(mpi_core, "rank", 0, raising=True)
    assert mpi_core.is_master() is True

    monkeypatch.setattr(mpi_core, "rank", 1, raising=True)
    assert mpi_core.is_master() is False


@pytest.mark.mpi
def test_env_global_rank_str_precedence(monkeypatch, mpi_core):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "5")
    monkeypatch.setattr(mpi_core, "rank", 0, raising=True)
    assert mpi_core._env_global_rank_str() == "5"


@pytest.mark.mpi
def test_is_global_master_invalid_env_falls_back_to_rank(monkeypatch, mpi_core):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "not_an_int")

    monkeypatch.setattr(mpi_core, "rank", 0, raising=True)
    assert mpi_core.is_global_master() is True

    monkeypatch.setattr(mpi_core, "rank", 3, raising=True)
    assert mpi_core.is_global_master() is False


@pytest.mark.mpi
def test_barrier_noop_when_single_rank(monkeypatch, mpi_core):
    class _BarrierExplodes:
        def Barrier(self):
            raise AssertionError("Barrier() should not be called in serial")

    monkeypatch.setattr(mpi_core, "n_nodes", 1, raising=True)
    monkeypatch.setattr(mpi_core, "comm", _BarrierExplodes(), raising=True)

    mpi_core.barrier()


@pytest.mark.mpi
def test_barrier_calls_comm_when_parallel(monkeypatch, mpi_core):
    comm = SimpleNamespace(called=False)

    def _bar():
        comm.called = True

    monkeypatch.setattr(mpi_core, "n_nodes", 2, raising=True)
    monkeypatch.setattr(mpi_core, "comm", SimpleNamespace(Barrier=_bar), raising=True)

    mpi_core.barrier()
    assert comm.called is True


@pytest.mark.mpi
def test_print0_only_on_master(monkeypatch, capsys, mpi_core):
    monkeypatch.setattr(mpi_core, "rank", 0, raising=True)
    mpi_core.print0("hello")
    out = capsys.readouterr().out
    assert "hello" in out

    monkeypatch.setattr(mpi_core, "rank", 1, raising=True)
    mpi_core.print0("nope")
    out = capsys.readouterr().out
    assert out == ""


@pytest.mark.mpi
def test_hostname_strips_hash_suffix(monkeypatch, mpi_core):
    class _MPIName:
        @staticmethod
        def Get_processor_name():
            return "node123#7"

    monkeypatch.setattr(mpi_core, "MPI", _MPIName, raising=True)
    assert mpi_core.hostname() == "node123"


@pytest.mark.mpi
def test_detect_cpus_per_task_priority(monkeypatch, mpi_core):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert mpi_core.detect_cpus_per_task() == 8

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    assert mpi_core.detect_cpus_per_task() == 6

    monkeypatch.setenv("OMP_NUM_THREADS", "six")
    assert mpi_core.detect_cpus_per_task() == 1


@pytest.mark.mpi
def test_helpers_tqdm_disables_on_nonzero_rank(monkeypatch, mpi_helpers):
    monkeypatch.setattr(mpi_helpers, "rank", 1, raising=True)
    bar = mpi_helpers.tqdm(range(3))
    try:
        assert getattr(bar, "disable", None) is True
    finally:
        bar.close()


@pytest.mark.mpi
def test_helpers_patches_tqdm_modules(mpi_helpers):
    import tqdm
    import tqdm.auto

    assert tqdm.tqdm is mpi_helpers.tqdm
    assert tqdm.auto.tqdm is mpi_helpers.tqdm


@pytest.mark.mpi
def test_check_mpi_disabled_returns_minimal_dict(monkeypatch, mpi_helpers):
    monkeypatch.setattr(mpi_helpers, "available", False, raising=True)
    info = mpi_helpers.check_mpi(return_dict=True, extended=False)
    assert info == {"mpi_enabled": "False"}


@pytest.mark.mpi
def test_check_mpi_enabled_return_dict_smoke(monkeypatch, mpi_helpers):

    monkeypatch.setattr(mpi_helpers, "available", True, raising=True)
    monkeypatch.setattr(mpi_helpers, "MPI", _MPISpy, raising=True)
    monkeypatch.setattr(mpi_helpers, "comm", _CommSpy(size=4, rank=0), raising=True)
    monkeypatch.setattr(
        mpi_helpers, "all_hostnames", lambda: ["A", "A", "B", "B"], raising=True
    )
    monkeypatch.setattr(mpi_helpers, "detect_cpus_per_task", lambda: 2, raising=True)
    monkeypatch.setattr(mpi_helpers, "is_master", lambda: True, raising=True)
    monkeypatch.setattr(
        mpi_helpers, "cfg", SimpleNamespace(get=lambda k: False), raising=True
    )
    monkeypatch.setenv("NETKET_MPI_AUTODETECT_LOCAL_GPU", "0")

    info = mpi_helpers.check_mpi(return_dict=True, extended=False)
    assert info["mpi4py_available"] == "True"
    assert info["n_ranks"] == "4"
    assert info["n_nodes"] == "2"
    assert info["cpus_per_task"] == "2"
    assert info["total_cpu_cores_used"] == "8"
    assert "mpi4py | MPI version" in info


@pytest.mark.mpi
def test_get_mpi_info_dict_disabled(monkeypatch, mpi_helpers):
    monkeypatch.setattr(mpi_helpers, "available", False, raising=True)
    d = mpi_helpers.get_mpi_info_dict()
    assert d == {"enabled": False, "n_nodes": 1, "ranks_per_node": 1}


@pytest.mark.mpi
def test_get_mpi_info_dict_enabled(monkeypatch, mpi_helpers):
    monkeypatch.setattr(mpi_helpers, "available", True, raising=True)
    monkeypatch.setattr(
        mpi_helpers, "all_hostnames", lambda: ["X", "X", "Y"], raising=True
    )
    monkeypatch.setattr(mpi_helpers, "local_ranks", lambda: 2, raising=True)
    d = mpi_helpers.get_mpi_info_dict()
    assert d["enabled"] is True
    assert d["n_nodes"] == 2
    assert d["ranks_per_node"] == 2


@pytest.mark.mpi
def test_mpi_sum_prod_max_branches(simulated_collectives, mpi_primitives):
    x = np.array([1.0, 2.0])
    out_sum = mpi_primitives.mpi_sum(x)
    assert np.allclose(out_sum, x * 4)

    out_prod = mpi_primitives.mpi_prod(np.array([2.0, 3.0]))
    assert np.allclose(out_prod, np.array([2.0, 3.0]) ** 4)

    out_max = mpi_primitives.mpi_max(np.array([5.0, -1.0]))
    assert np.allclose(out_max, np.array([5.0, -1.0]))


@pytest.mark.mpi
def test_mpi_sum_respects_out_buffer(simulated_collectives, mpi_primitives):
    x = np.array([1, 2], dtype=np.int64)
    out = np.empty_like(x)
    ret = mpi_primitives.mpi_sum(x, out=out)
    assert ret is out
    assert np.all(ret == np.array([4, 8]))


@pytest.mark.mpi
def test_mpi_mean_is_sum_div_n_nodes(simulated_collectives, mpi_primitives):
    x = np.array([3.0, 7.0])
    m = mpi_primitives.mpi_mean(x)
    assert np.allclose(m, x)


@pytest.mark.mpi
def test_mpi_any_all_returns_bool(simulated_collectives, mpi_primitives):
    x = np.array([1, 0, 1])
    a = mpi_primitives.mpi_any(x)
    b = mpi_primitives.mpi_all(x)
    assert a.dtype == np.bool_
    assert b.dtype == np.bool_
    assert np.all(a == np.array([True, False, True]))
    assert np.all(b == np.array([True, False, True]))


@pytest.mark.mpi
def test_mpi_bcast_calls_comm_when_parallel(
    monkeypatch, simulated_collectives, mpi_primitives
):
    comm = simulated_collectives
    y = mpi_primitives.mpi_bcast({"k": 1}, root=2)
    assert y == {"k": 1}
    assert comm.last_bcast_root == 2


@pytest.mark.mpi
def test_mpi_allgather_ndarray_shape(simulated_collectives, mpi_primitives):
    x = np.array([9, 10], dtype=np.int32)
    out = mpi_primitives.mpi_allgather(x)
    assert out.shape == (4, 2)
    assert np.all(out[0] == x)
    assert np.all(out[3] == x)


@pytest.mark.mpi
def test_mpi_allgather_python_object(simulated_collectives, mpi_primitives):
    out = mpi_primitives.mpi_allgather("hi")
    assert out == ["hi"] * 4


@pytest.mark.mpi
def test_mpi_gather_ndarray_root_gets_stack(
    monkeypatch, simulated_collectives, mpi_primitives
):
    x = np.array([1, 2, 3], dtype=np.float64)

    monkeypatch.setattr(mpi_primitives, "rank", 0, raising=True)
    out = mpi_primitives.mpi_gather(x, root=0)
    assert out.shape == (4, 3)
    assert np.all(out[2] == x)


@pytest.mark.mpi
def test_mpi_gather_ndarray_nonroot_returns_none(
    monkeypatch, simulated_collectives, mpi_primitives
):
    x = np.array([1, 2, 3], dtype=np.float64)

    monkeypatch.setattr(mpi_primitives, "rank", 1, raising=True)
    out = mpi_primitives.mpi_gather(x, root=0)
    assert out is None


@pytest.mark.mpi
def test_mpi_scatter_serial_requires_len1(monkeypatch, mpi_primitives):
    monkeypatch.setattr(mpi_primitives, "n_nodes", 1, raising=True)

    assert mpi_primitives.mpi_scatter([42], root=0) == 42
    with pytest.raises(ValueError, match="Serial scatter expects a length-1 sequence"):
        mpi_primitives.mpi_scatter([1, 2], root=0)


@pytest.mark.skipif(
    core.n_nodes < 2, reason="Run under mpiexec -n>1 to exercise real collectives."
)
@pytest.mark.mpi
def test_real_mpi_sum_matches_expected_across_ranks():
    x = np.array(core.rank + 1, dtype=np.int64)
    out = primitives.mpi_sum(x)
    expected = np.array((core.n_nodes * (core.n_nodes + 1)) // 2, dtype=np.int64)
    assert out == expected


@pytest.mark.skipif(
    core.n_nodes < 2, reason="Run under mpiexec -n>1 to exercise real gather/scatter."
)
@pytest.mark.mpi
def test_real_mpi_scatter_roundtrip():
    root = 0
    xs = list(range(core.n_nodes)) if core.rank == root else []
    y = primitives.mpi_scatter(xs, root=root)
    assert y == core.rank


@pytest.mark.skipif(
    core.n_nodes < 2, reason="Run under mpiexec -n>1 to exercise real gather."
)
@pytest.mark.mpi
def test_real_mpi_gather_root_only():
    root = 0
    x = np.array([core.rank], dtype=np.int64)
    gathered = primitives.mpi_gather(x, root=root)
    if core.rank == root:
        assert gathered.shape == (core.n_nodes, 1)
        assert np.all(gathered[:, 0] == np.arange(core.n_nodes))
    else:
        assert gathered is None


@pytest.mark.mpi
def test_core_uses_netket_comm_when_netket_mpi_enabled(mpi_core, monkeypatch):
    nk_mpi = pytest.importorskip("netket.utils.mpi")

    if os.environ.get("NETKET_MPI", "0") != "1":
        pytest.skip("NETKET_MPI != 1; NetKet communicator adoption is not expected.")

    nk_comm = getattr(nk_mpi, "MPI_py_comm", None)
    nk_rank = getattr(nk_mpi, "rank", None)
    nk_size = getattr(nk_mpi, "n_nodes", None)

    if nk_comm is None:
        pytest.skip(
            "NetKet MPI is importable but MPI_py_comm is None (NetKet MPI disabled?)."
        )

    assert mpi_core.comm is nk_comm, "neuraLQX core.comm is not NetKet's MPI_py_comm"

    assert mpi_core.rank == int(nk_rank)
    assert mpi_core.n_nodes == int(nk_size)
    assert mpi_core.comm.Get_rank() == mpi_core.rank
    assert mpi_core.comm.Get_size() == mpi_core.n_nodes


@pytest.mark.mpi
def test_core_netket_adoption_implies_no_dup_comm_side_effect(mpi_core):
    assert mpi_core.comm.Get_size() == mpi_core.n_nodes
    assert mpi_core.comm.Get_rank() == mpi_core.rank


@pytest.mark.mpi
def test_all_hostnames_returns_len_n_nodes_and_contains_own_hostname(mpi_core):
    hs = mpi_core.all_hostnames()
    assert isinstance(hs, (list, tuple))
    assert len(hs) == mpi_core.n_nodes
    assert mpi_core.hostname() in hs


@pytest.mark.mpi
def test_ranks_per_node_counts_sum_to_n_nodes(mpi_core):
    c = mpi_core.ranks_per_node()
    assert sum(c.values()) == mpi_core.n_nodes
    assert c[mpi_core.hostname()] >= 1


@pytest.mark.mpi
def test_local_ranks_matches_counter(mpi_core):
    c = mpi_core.ranks_per_node()
    assert mpi_core.local_ranks() == c[mpi_core.hostname()]


@pytest.mark.mpi
def test_primitives_serial_reductions_return_input(monkeypatch, mpi_primitives):
    monkeypatch.setattr(mpi_primitives, "n_nodes", 1, raising=True)

    x = np.array([1.0, 2.0])
    assert np.allclose(mpi_primitives.mpi_sum(x), x)
    assert np.allclose(mpi_primitives.mpi_prod(x), x)
    assert np.allclose(mpi_primitives.mpi_max(x), x)

    xb = np.array([1, 0, 1])
    assert np.all(mpi_primitives.mpi_any(xb) == np.array([True, False, True]))
    assert np.all(mpi_primitives.mpi_all(xb) == np.array([True, False, True]))


@pytest.mark.mpi
def test_primitives_serial_bcast_returns_input(monkeypatch, mpi_primitives):
    monkeypatch.setattr(mpi_primitives, "n_nodes", 1, raising=True)
    obj = {"a": 1}
    assert mpi_primitives.mpi_bcast(obj, root=0) is obj


@pytest.mark.mpi
def test_primitives_serial_allgather_ndarray_shape(monkeypatch, mpi_primitives):
    monkeypatch.setattr(mpi_primitives, "n_nodes", 1, raising=True)
    x = np.array([3, 4, 5], dtype=np.int32)
    out = mpi_primitives.mpi_allgather(x)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 3)
    assert np.all(out[0] == x)


@pytest.mark.mpi
def test_primitives_serial_allgather_python_object(monkeypatch, mpi_primitives):
    monkeypatch.setattr(mpi_primitives, "n_nodes", 1, raising=True)
    out = mpi_primitives.mpi_allgather("hi")
    assert out == ("hi",)


@pytest.mark.mpi
def test_primitives_serial_gather_ndarray_shape(monkeypatch, mpi_primitives):
    monkeypatch.setattr(mpi_primitives, "n_nodes", 1, raising=True)
    x = np.array([7, 8], dtype=np.int64)
    out = mpi_primitives.mpi_gather(x, root=0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 2)
    assert np.all(out[0] == x)


@pytest.mark.mpi
def test_primitives_serial_gather_python_object(monkeypatch, mpi_primitives):
    monkeypatch.setattr(mpi_primitives, "n_nodes", 1, raising=True)
    out = mpi_primitives.mpi_gather(123, root=0)
    assert out == (123,)


@pytest.mark.mpi
def test_helpers_tqdm_enabled_on_rank0(monkeypatch, mpi_helpers):
    monkeypatch.setattr(mpi_helpers, "rank", 0, raising=True)
    bar = mpi_helpers.tqdm(range(2))
    try:
        assert getattr(bar, "disable", None) in (False, None)
    finally:
        bar.close()
