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

from dataclasses import dataclass

from typing import Optional
from typing import Union
from typing import Dict
from typing import Any
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class OptaxSchedule(Protocol):
    """A Protocol so we can type the schedule callable that Optax returns."""

    def __call__(self, step: int) -> float: ...


#
#
#   learning rate specs (either float or schedule config)


@dataclass(frozen=True)
class ExponentialDecay:
    init_value: float
    transition_steps: int
    decay_rate: float
    staircase: bool = True
    end_value: Optional[float] = None


@dataclass(frozen=True)
class CosineDecay:
    init_value: float
    decay_steps: int
    alpha: float = 0.0


@dataclass(frozen=True)
class LinearDecay:
    init_value: float
    end_value: float
    transition_steps: int
    transition_begin: int = 0


LearningRateSpec = Union[ExponentialDecay, CosineDecay, LinearDecay]


#
#
#   optimiser configs


@dataclass(frozen=True)
class Adam:
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8


@dataclass(frozen=True)
class SGD:
    momentum: float = 0.0
    nesterov: bool = False


# alias Momentum to SGD semantics
Momentum = SGD


@dataclass(frozen=True)
class Adagrad:
    eps: float = 1e-7
    initial_accumulator_value: float = 0.1


@dataclass(frozen=True)
class RMSProp:
    decay: float = 0.9
    eps: float = 1e-7
    centered: bool = False


@dataclass(frozen=True)
class AdaBelief:
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-16
    eps_root: float = 1e-16


@dataclass(frozen=True)
class AMSGrad:
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    mu_dtype: Optional[Any] = None  # passthrough


@dataclass(frozen=True)
class Yogi:
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-3


#
#
#  a simple registry to map string names -> config constructors (for backwards compatibility)

OPTIMIZER_REGISTRY: Dict[str, Any] = {
    "adam": Adam,
    "sgd": SGD,
    "momentum": Momentum,
    "adagrad": Adagrad,
    "rmsprop": RMSProp,
    "adabelief": AdaBelief,
    "amsgrad": AMSGrad,
    "yogi": Yogi,
}

SCHEDULE_REGISTRY: Dict[str, Any] = {
    "exponential decay": ExponentialDecay,
    "exponential_decay": ExponentialDecay,
    "cosine decay": CosineDecay,
    "cosine_decay": CosineDecay,
    "linear decay": LinearDecay,
    "linear_decay": LinearDecay,
}
