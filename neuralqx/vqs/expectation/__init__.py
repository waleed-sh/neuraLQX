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


"""Expectation, force, and gradient estimators."""

from .forces import can_use_covariance
from .forces import expect_and_forces
from .forces import force_to_grad
from .forces import forces_from_local_values
from .gradients import expect_and_grad
from .penalties import penalty_expect
from .penalties import penalty_expect_and_forces
from .penalties import penalty_expect_and_grad
from .penalties import register_penalty_expect
from .penalties import register_penalty_expect_and_forces
from .penalties import register_penalty_expect_and_grad
from .values import expect

__all__ = [
    "can_use_covariance",
    "expect",
    "expect_and_forces",
    "expect_and_grad",
    "force_to_grad",
    "forces_from_local_values",
    "penalty_expect",
    "penalty_expect_and_forces",
    "penalty_expect_and_grad",
    "register_penalty_expect",
    "register_penalty_expect_and_forces",
    "register_penalty_expect_and_grad",
]
