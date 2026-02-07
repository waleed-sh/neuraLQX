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

from .types import Adam
from .types import SGD
from .types import Momentum
from .types import Adagrad
from .types import RMSProp
from .types import AdaBelief
from .types import AMSGrad
from .types import Yogi
from .types import LearningRateSpec
from .types import OptaxSchedule

from .schedule import make_learning_rate

LRLike = Union[float, OptaxSchedule]


#
#
#   Adam


@dispatch
def build_optimizer(cfg: Adam, lr: LRLike):
    return optax.adam(learning_rate=lr, b1=cfg.b1, b2=cfg.b2, eps=cfg.eps)


@dispatch
def build_optimizer(cfg: Adam, lr_spec: LearningRateSpec):
    return build_optimizer(cfg, make_learning_rate(lr_spec))


#
#
#   SGD / Momentum


@dispatch
def build_optimizer(cfg: SGD, lr: LRLike):
    return optax.sgd(learning_rate=lr, momentum=cfg.momentum, nesterov=cfg.nesterov)


@dispatch
def build_optimizer(cfg: Momentum, lr: LRLike):
    # identical to SGD, still gets its own dispatched overload for clarity
    return optax.sgd(learning_rate=lr, momentum=cfg.momentum, nesterov=cfg.nesterov)


@dispatch
def build_optimizer(cfg: SGD, lr_spec: LearningRateSpec):
    return build_optimizer(cfg, make_learning_rate(lr_spec))


@dispatch
def build_optimizer(cfg: Momentum, lr_spec: LearningRateSpec):
    return build_optimizer(cfg, make_learning_rate(lr_spec))


#
#
#   Adagrad


@dispatch
def build_optimizer(cfg: Adagrad, lr: LRLike):
    return optax.adagrad(
        learning_rate=lr,
        eps=cfg.eps,
        initial_accumulator_value=cfg.initial_accumulator_value,
    )


@dispatch
def build_optimizer(cfg: Adagrad, lr_spec: LearningRateSpec):
    return build_optimizer(cfg, make_learning_rate(lr_spec))


#
#
#   RMSProp


@dispatch
def build_optimizer(cfg: RMSProp, lr: LRLike):
    return optax.rmsprop(
        learning_rate=lr, decay=cfg.decay, eps=cfg.eps, centered=cfg.centered
    )


@dispatch
def build_optimizer(cfg: RMSProp, lr_spec: LearningRateSpec):
    return build_optimizer(cfg, make_learning_rate(lr_spec))


#
#
#   AdaBelief


@dispatch
def build_optimizer(cfg: AdaBelief, lr: LRLike):
    return optax.adabelief(
        learning_rate=lr, b1=cfg.b1, b2=cfg.b2, eps=cfg.eps, eps_root=cfg.eps_root
    )


@dispatch
def build_optimizer(cfg: AdaBelief, lr_spec: LearningRateSpec):
    return build_optimizer(cfg, make_learning_rate(lr_spec))


#
#
#   AMSGrad


@dispatch
def build_optimizer(cfg: AMSGrad, lr: LRLike):
    return optax.amsgrad(
        learning_rate=lr,
        b1=cfg.b1,
        b2=cfg.b2,
        eps=cfg.eps,
        eps_root=cfg.eps_root,
        mu_dtype=cfg.mu_dtype,
    )


@dispatch
def build_optimizer(cfg: AMSGrad, lr_spec: LearningRateSpec):
    return build_optimizer(cfg, make_learning_rate(lr_spec))


#
#
#   Yogi


@dispatch
def build_optimizer(cfg: Yogi, lr: LRLike):
    return optax.yogi(learning_rate=lr, b1=cfg.b1, b2=cfg.b2, eps=cfg.eps)


@dispatch
def build_optimizer(cfg: Yogi, lr_spec: LearningRateSpec):
    return build_optimizer(cfg, make_learning_rate(lr_spec))
