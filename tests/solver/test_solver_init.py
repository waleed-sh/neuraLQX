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


def test_solver_init_creates_output_dir(solver_env):
    Solver = solver_env.Solver
    s = Solver(solver_env.lqx, output_path=solver_env.output_path)

    assert os.path.isdir(s.output_path)

    assert os.path.basename(s.output_path) == s.hash

    assert os.path.basename(os.path.dirname(s.output_path)) == "solver_output"

    assert s.lqx is solver_env.lqx
    assert not s.is_solved
    assert s.log is not None


def test_solver_flags_and_default_values(solver_env):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)

    assert not s.is_initialised
    assert not s.is_solved

    assert not s._sampler_flag
    assert not s._opt_flag
    assert not s._network_flag
    assert not s._driver_flag
    assert not s._vstate_flag


def test_solver_uninitialised_access_raises(solver_env):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)

    with pytest.raises(AttributeError):
        _ = s.sampler

    with pytest.raises(AttributeError):
        _ = s.optimizer

    with pytest.raises(AttributeError):
        _ = s.network

    with pytest.raises(AttributeError):
        _ = s.vmc_driver
