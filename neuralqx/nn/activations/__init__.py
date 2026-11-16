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


"""Activation functions usable from NNX, Linen and pure JAX models."""

from .complex import reim
from .complex import reim_gelu
from .complex import reim_logcosh
from .complex import reim_relu
from .complex import reim_selu
from .complex import reim_sigmoid
from .complex import reim_silu
from .complex import reim_softplus
from .complex import reim_tanh
from .functions import celu
from .functions import elu
from .functions import gelu
from .functions import glu
from .functions import hard_sigmoid
from .functions import hard_silu
from .functions import hard_swish
from .functions import hard_tanh
from .functions import identity
from .functions import leaky_relu
from .functions import log_sigmoid
from .functions import log_softmax
from .functions import logcosh
from .functions import mish
from .functions import relu
from .functions import relu6
from .functions import selu
from .functions import sigmoid
from .functions import silu
from .functions import soft_sign
from .functions import softmax
from .functions import softplus
from .functions import sparse_plus
from .functions import sparse_sigmoid
from .functions import squareplus
from .functions import swish
from .functions import tanh
from .registry import ActivationSpec
from .registry import activation
from .registry import activation_spec
from .registry import available_activations
from .registry import get_activation

__all__ = [
    "ActivationSpec",
    "activation",
    "activation_spec",
    "available_activations",
    "celu",
    "elu",
    "gelu",
    "get_activation",
    "glu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "identity",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "logcosh",
    "mish",
    "reim",
    "reim_gelu",
    "reim_logcosh",
    "reim_relu",
    "reim_selu",
    "reim_sigmoid",
    "reim_silu",
    "reim_softplus",
    "reim_tanh",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "soft_sign",
    "softmax",
    "softplus",
    "sparse_plus",
    "sparse_sigmoid",
    "squareplus",
    "swish",
    "tanh",
]
