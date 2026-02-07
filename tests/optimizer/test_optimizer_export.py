#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  License under the Apache License 2.0, Version 2.0 (the "License");
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


from __future__ import annotations


def test_export_info_with_constant_lr(optimizer_modules):
    tmod, _smod, _bmod, emod, _omod = optimizer_modules

    cfg = tmod.Adam(b1=0.9, b2=0.99, eps=1e-7)
    fields, values = emod.export_info(cfg, 0.001)

    assert fields[0] == "Optimizer"
    assert values[0] == "Adam"

    assert any(f.startswith("Optimizer param: ") for f in fields)
    assert "Learning rate" in fields
    assert values[fields.index("Learning rate")] == "0.001"


def test_export_info_with_scheduler(optimizer_modules):
    tmod, _smod, _bmod, emod, _omod = optimizer_modules

    cfg = tmod.SGD(momentum=0.2, nesterov=True)
    sched = tmod.ExponentialDecay(init_value=0.1, transition_steps=5, decay_rate=0.9)

    fields, values = emod.export_info(cfg, sched)

    assert fields[0] == "Optimizer"
    assert values[0] == "SGD"

    assert "Scheduler type" in fields
    assert values[fields.index("Scheduler type")] == "ExponentialDecay"

    assert any(f.startswith("Scheduler param: ") for f in fields)
