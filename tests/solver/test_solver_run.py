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


from netket.logging import RuntimeLog

from neuralqx.utils.io.runtime_loggers import DeferredRuntimeLog


def test_initialize_vmc_builds_vmc(solver_env, nk):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)
    s.set_sampler(sampler_type="U1 Gauge Sampler")
    s.set_optimizer(optimizer_type="adam", use_sr=False)
    s.set_network(nk.models.RBM(alpha=2))
    s.initialize_vmc()
    assert hasattr(s, "vmc_driver")
    assert hasattr(s.vmc_driver, "run")


def test_run_executes_vmc_loop_and_sets_solved_flag(solver_env, nk):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)
    s.set_sampler(sampler_type="U1 Gauge Sampler")
    s.set_optimizer(optimizer_type="adam", use_sr=False)
    s.set_network(nk.models.RBM(alpha=2))
    s.initialize_vmc()
    observables = {"v0": s.lqx.volume(0), "F": s.lqx.curvature_constraint()}
    s.run(5, observables=observables, live_monitoring=False)
    assert s._solved
    assert s.vmc_driver.step_count == 5
    assert isinstance(s.log, RuntimeLog)


def test_run_handles_keyboard_interrupt_gracefully(solver_env, monkeypatch, nk):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)

    s.set_sampler(sampler_type="U1 Gauge Sampler")
    s.set_optimizer(optimizer_type="adam", use_sr=False)
    s.set_network(nk.models.RBM(alpha=2))
    s.initialize_vmc()

    def raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(s.vmc_driver, "run", raise_interrupt)

    s.run(1)

    assert s.is_solved


def test_run_toggles_deferred_runtime_log_from_silent_print(solver_env, monkeypatch):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)

    # Avoid entering the expensive simulation/finalisation paths.
    monkeypatch.setattr(s, "_execute_run", lambda ctx: None)
    monkeypatch.setattr(s, "_finalise_run", lambda ctx: None)
    s._driver_flag = True

    s.run(1, silent_print=True, _ct=True)
    assert isinstance(s.log, DeferredRuntimeLog)
    assert s.log.defer_accumulation is True

    s.run(1, silent_print=False, _ct=True)
    assert s.log.defer_accumulation is False


def test_continue_simulation_recreates_deferred_runtime_log_if_missing(
    solver_env, monkeypatch
):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)

    captured = {}

    def fake_run(*args, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(s, "run", fake_run)
    s._driver_flag = True
    s._nk_log = None

    s.continue_simulation(n_iters=2, silent_print=True, state_path=None)

    assert isinstance(s.log, DeferredRuntimeLog)
    assert captured["kwargs"]["silent_print"] is True
