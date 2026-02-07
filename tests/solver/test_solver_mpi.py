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

pytestmark = pytest.mark.skipif(
    not os.environ.get("NQX_MPI_TESTS"),
    reason="MPI tests require NQX_MPI_TESTS=1 and NQX_MPI=1",
)


def test_rank0_only_exports_in_mpi_mode(solver_env, monkeypatch, nk):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)

    s.set_sampler(sampler_type="U1 Gauge Sampler")
    s.set_optimizer(optimizer_type="adam", use_sr=True)
    s.set_network(nk.models.RBM(alpha=1))
    s.initialize_vmc()

    monkeypatch.setattr(
        "neuralqx.solver.solver._mpi.is_global_master",
        lambda: False,
    )

    monkeypatch.setattr(
        "neuralqx.solver.solver._mpi.barrier",
        lambda: None,
    )

    s.export_state(marker="MPI")

    assert not any("MPI" in f for f in os.listdir(s.output_path))
