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

import jax.numpy as jnp

from neuralqx.nn.projectors.utils import _project_log_psi


def test_project_log_psi_singleton_identity_preserves_log_branch():
    sigma = jnp.asarray(
        [
            [0.1, 0.2, 0.3],
            [1.1, 1.2, 1.3],
            [2.1, 2.2, 2.3],
        ],
        dtype=jnp.float32,
    )
    perms_full = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)

    def base_call(x):
        real_part = x[:, 0]
        # Force an imaginary phase outside the principal branch.
        imag_part = 4.0 * jnp.pi + x[:, 1]
        return real_part + 1j * imag_part

    projected = _project_log_psi(
        base_call=base_call,
        sigma=sigma,
        perms_full=perms_full,
        characters=None,
    )

    expected = base_call(sigma)
    assert jnp.allclose(projected, expected, atol=1e-7, rtol=0.0)
