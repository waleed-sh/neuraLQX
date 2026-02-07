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

from __future__ import annotations

from plum import dispatch

import optax

from typing import Union

from .types import LearningRateSpec
from .types import ExponentialDecay
from .types import CosineDecay
from .types import LinearDecay
from .types import OptaxSchedule


@dispatch
def build_lr(spec: float) -> float:
    """Constant learning rate."""
    return spec


@dispatch
def build_lr(spec: ExponentialDecay) -> OptaxSchedule:
    return optax.exponential_decay(
        init_value=spec.init_value,
        transition_steps=spec.transition_steps,
        decay_rate=spec.decay_rate,
        staircase=spec.staircase,
        end_value=spec.end_value,
    )


@dispatch
def build_lr(spec: CosineDecay) -> OptaxSchedule:
    return optax.cosine_decay_schedule(
        init_value=spec.init_value,
        decay_steps=spec.decay_steps,
        alpha=spec.alpha,
    )


@dispatch
def build_lr(spec: LinearDecay) -> OptaxSchedule:
    return optax.linear_schedule(
        init_value=spec.init_value,
        end_value=spec.end_value,
        transition_steps=spec.transition_steps,
        transition_begin=spec.transition_begin,
    )


#
#
#   convenience: tolerant unified entry that returns float or schedule callable
@dispatch
def make_learning_rate(
    spec: Union[float, LearningRateSpec],
) -> Union[float, OptaxSchedule]:
    return build_lr(spec)
