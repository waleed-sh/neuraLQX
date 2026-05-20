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
import os
import sys

import numpy as np
import pytest


class _Series:
    def __init__(self, iters, mean, sigma):
        self.iters = np.asarray(iters)
        self.Mean = np.asarray(mean)
        self.Sigma = np.asarray(sigma)
        self.R_hat = np.ones_like(self.Mean)

    def __getitem__(self, item):
        mask = (self.iters >= item.start) & (self.iters < item.stop)
        return _Series(self.iters[mask], self.Mean[mask], self.Sigma[mask])


class _Log:
    def __init__(self):
        self.data = {
            "Constraint (state 0)": _Series([0, 1], [1.2, 1.0], [0.1, 0.08]),
            "Constraint/s1": _Series([0, 1], [1.4, 1.1], [0.12, 0.09]),
        }

    def serialize(self, path):
        with open(path + ".json", "w", encoding="utf-8") as handle:
            handle.write("{}")


def test_multi_state_public_api_is_exported(nqx):
    assert nqx.solver.MultiSolver is not None
    assert nqx.driver.MultiStateVMC is not None
    assert nqx.vqs.MultiMCState is not None


def test_promoted_multi_state_api_is_not_exported_from_experimental(nqx):
    for name in list(sys.modules):
        if name == "neuralqx.experimental" or name.startswith("neuralqx.experimental."):
            sys.modules.pop(name)

    with nqx.cfg.patch("EXPERIMENTAL", True):
        exp_solver = importlib.import_module("neuralqx.experimental.solver")
        exp_driver = importlib.import_module("neuralqx.experimental.driver")
        exp_vqs = importlib.import_module("neuralqx.experimental.vqs")
        exp_mc_state = importlib.import_module("neuralqx.experimental.vqs.mc.mc_state")

        assert not hasattr(exp_solver, "MultiSolver")
        assert not hasattr(exp_driver, "MultiStateVMC")
        assert not hasattr(exp_vqs, "MultiMCState")
        assert not hasattr(exp_mc_state, "MultiMCState")

        assert exp_solver.__all__ == ["STMultiSolver"]
        assert exp_driver.__all__ == ["SingleTrunkMultiHeadVMC"]
        assert exp_vqs.__all__ == ["mc", "STMultiMCState"]
        assert exp_mc_state.__all__ == ["STMultiMCState"]

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("neuralqx.experimental.solver.multi_state_solver")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("neuralqx.experimental.driver.mvmc")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("neuralqx.experimental.vqs.mc.mc_state.mtmh_state")


def test_multi_solver_initializes_public_state_and_driver(solver_env, nk, nqx):
    solver = nqx.solver.MultiSolver(solver_env.lqx, output_path=solver_env.output_path)

    solver.set_sampler(sampler_type="U1 Gauge Sampler", number_of_samples=16)
    solver.set_optimizer(optimizer_type="adam", learning_rate=0.01, use_sr=False)
    solver.set_network(
        [nk.models.RBM(alpha=1), nk.models.RBM(alpha=1)],
        lambda_ortho=0.0,
    )
    solver.initialize_vmc()

    assert isinstance(solver.variational_state, nqx.vqs.MultiMCState)
    assert isinstance(solver.vmc_driver, nqx.driver.MultiStateVMC)
    assert solver.variational_state.n_states == 2


def test_multi_solver_export_import_cycle(solver_env, nk, nqx):
    solver = nqx.solver.MultiSolver(solver_env.lqx, output_path=solver_env.output_path)

    solver.set_sampler(sampler_type="U1 Gauge Sampler", number_of_samples=16)
    solver.set_optimizer(optimizer_type="adam", learning_rate=0.01, use_sr=False)
    solver.set_network(
        [nk.models.RBM(alpha=1), nk.models.RBM(alpha=1)],
        lambda_ortho=0.0,
    )
    solver.initialize_vmc()

    checkpoint = solver.export_state(marker="TEST", silent=True)
    assert checkpoint is not None
    assert os.path.exists(checkpoint)

    restored = solver.import_state(checkpoint, force_load_mpi=True)
    assert isinstance(restored, nqx.vqs.MultiMCState)
    assert restored.n_states == 2


def test_multi_solver_plot_results_creates_images_from_state_series(solver_env, nqx):
    solver = nqx.solver.MultiSolver(solver_env.lqx, output_path=solver_env.output_path)
    solver._nk_log = _Log()
    solver._max_iters = 2

    solver.plot_results(silent_plot=True)

    files = os.listdir(solver.output_path)
    assert any(name.endswith(".png") for name in files)
    assert any(name.endswith(".json") for name in files)
