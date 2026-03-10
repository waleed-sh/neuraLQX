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


def test_plot_results_creates_images(solver_env, nk):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)

    s.set_sampler(sampler_type="U1 Gauge Sampler", number_of_samples=16)
    s.set_optimizer(optimizer_type="adam", learning_rate=0.01, use_sr=False)
    s.set_network(nk.models.RBM(alpha=1))
    s.initialize_vmc()

    s.run(
        1,
        silent_plot=True,
        silent_print=True,
        live_monitoring=False,
    )

    s.plot_results(silent_plot=True)

    images = [f for f in os.listdir(s.output_path) if f.endswith(".png")]

    assert images, "Expected at least one PNG file to be generated."


def test_plot_observables_handles_missing_keys(solver_env):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)

    with pytest.raises(ValueError):
        s.plot_observables("NonExistentObservable")
