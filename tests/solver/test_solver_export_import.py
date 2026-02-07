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


def test_export_and_import_state_cycle(solver_env, nk):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)
    s.set_sampler(sampler_type="U1 Gauge Sampler")
    s.set_optimizer(optimizer_type="adam", use_sr=True)
    s.set_network(nk.models.RBM(alpha=2))
    s.initialize_vmc()
    s.export_state(marker="TEST")
    exported = [f for f in os.listdir(s.output_path) if "SerialisedState" in f]
    assert exported, "Exported state file should exist."

    fname = os.path.join(s.output_path, exported[0])
    s.import_state(fname, force_load_mpi=True)
    assert hasattr(s, "vmc_driver")
    assert hasattr(s.vmc_driver.state, "parameters")
