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


from typing import Optional
from typing import Union
from typing import Callable
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

InitLike = Optional[Union[float, int, Callable[..., Any]]]


def _real_dtype_for_complex(dtype):
    if dtype == jnp.complex64:
        return jnp.float32
    if dtype == jnp.complex128:
        return jnp.float64
    return jnp.float32


def _as_initializer(spec: InitLike, default_std: float, dtype):
    """
    Resolve `spec` into a Flax initializer.

    if spec is a:
      - callable -> used directly
      - float/int -> normal(stddev=spec)
      - None -> normal(stddev=default_std)

    Complex dtypes get (r + i*j)/sqrt(2) with the same real stddev
    """

    if callable(spec):
        init = spec
    else:
        std = float(spec) if (spec is not None) else float(default_std)
        init = nn.initializers.normal(stddev=std)

    if jnp.issubdtype(dtype, jnp.complexfloating):
        real_init = init

        def complex_init(key, shape, _dtype=dtype):
            rkey, ikey = jax.random.split(key)
            rdtype = _real_dtype_for_complex(_dtype)
            r = real_init(rkey, shape, rdtype)
            i = real_init(ikey, shape, rdtype)
            return (r + 1j * i) / jnp.sqrt(2.0)

        return complex_init
    return init
