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

"""DType policy and JAX array initialisation utilities for neuraLQX."""

from neuralqx.utils.dtypes import names
from neuralqx.utils.dtypes.policy import DTypePolicy

from neuralqx.utils.dtypes.runtime import array_complex
from neuralqx.utils.dtypes.runtime import array_index
from neuralqx.utils.dtypes.runtime import array_real
from neuralqx.utils.dtypes.runtime import dtype_name_map
from neuralqx.utils.dtypes.runtime import full_complex
from neuralqx.utils.dtypes.runtime import full_real
from neuralqx.utils.dtypes.runtime import get_dtype_policy
from neuralqx.utils.dtypes.runtime import jax_complex_dtype
from neuralqx.utils.dtypes.runtime import jax_index_dtype
from neuralqx.utils.dtypes.runtime import jax_real_dtype
from neuralqx.utils.dtypes.runtime import ones_complex
from neuralqx.utils.dtypes.runtime import ones_real
from neuralqx.utils.dtypes.runtime import zeros_complex
from neuralqx.utils.dtypes.runtime import zeros_index
from neuralqx.utils.dtypes.runtime import zeros_real

__all__ = [
    "DTypePolicy",
    "array_complex",
    "array_index",
    "array_real",
    "dtype_name_map",
    "full_complex",
    "full_real",
    "get_dtype_policy",
    "jax_complex_dtype",
    "jax_index_dtype",
    "jax_real_dtype",
    "names",
    "ones_complex",
    "ones_real",
    "zeros_complex",
    "zeros_index",
    "zeros_real",
]
