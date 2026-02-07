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

"""
This file contains implementations of several Metropolis-Hastings type samplers (NetKet
implementation)
"""

import netket as nk
from netket.utils.struct import field
import jax


@nk.utils.struct.dataclass
class MetropolisKLocalRule(nk.sampler.rules.MetropolisRule):

    n_flips: int = field(pytree_node=False)
    """Number of local flips."""

    def transition(
        self,
        sampler,
        machine,
        params,
        sampler_state,
        key,
        σ,
    ):

        n_chains = σ.shape[0]
        hilb = sampler.hilbert

        # split RNG keys: one for indices, one for flips
        key_indx, key_flip = jax.random.split(key)

        # pick K random sites per chain
        indxs = jax.random.randint(
            key_indx,
            shape=(n_chains, self.n_flips),
            minval=0,
            maxval=hilb.size,
        )

        # split key_flip into K subkeys for independent flips
        flip_keys = jax.random.split(key_flip, self.n_flips)

        # apply K sequential single-site flips
        σp = σ
        for i in range(self.n_flips):
            σp, _ = nk.hilbert.random.flip_state(hilb, flip_keys[i], σp, indxs[:, i])

        # no log-prob correction term (None)
        return σp, None
