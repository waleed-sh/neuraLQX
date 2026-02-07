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


def test_set_sampler_sets_flag_and_object(solver_env):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)

    s.set_sampler(
        sampler_type="U1 Gauge Sampler",
        number_of_samples=10,
    )

    assert s._sampler_flag is True

    sampler = s.sampler
    assert sampler is not None


def test_set_optimizer_sets_flag_and_values(solver_env):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)

    s.set_optimizer(
        optimizer_type="adam",
        learning_rate=0.001,
        diagonal_shift=0.2,
        use_sr=True,
    )

    assert s._opt_flag is True
    assert s.diagonal_shift == 0.2

    opt = s.optimizer
    assert opt is not None


def test_set_network_records_network_object(solver_env):
    s = solver_env.Solver(solver_env.lqx, output_path=solver_env.output_path)

    s.set_sampler(
        sampler_type="U1 Gauge Sampler",
        number_of_samples=10,
    )
    s.set_optimizer(
        optimizer_type="adam",
        learning_rate=0.001,
        use_sr=True,
    )

    import flax.linen as nn

    class DummyNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return x

    model = DummyNet()

    s.set_network(model)

    assert s._network_flag is True
    assert s.network is model
