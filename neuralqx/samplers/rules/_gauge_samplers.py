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
This file contains logic for a U(1)^N gauge sampler which samples only U(1)^N gauge invariant states
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp

import flax.linen as nn
from netket.sampler import ParallelTemperingSampler

from netket.sampler.rules import MetropolisRule
from netket.utils import struct
from netket.utils.types import PyTree
from netket.utils.types import PRNGKeyT


@struct.dataclass
class RandomU1GaugeSampler(MetropolisRule):
    """
    This class implements a U(1) gauge sampler which samples states from the Hilbert space based
    on a specified gauge fixing. The proposed states are essentially random gauge invariant states
    """

    def init_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        key: PRNGKeyT,
    ) -> Optional[Any]:

        return sampler.hilbert.random_state(key=key, size=1)

    def random_state(  # pylint: disable=R0913
        self,
        sampler,
        machine,
        params,
        sampler_state,
        key,
    ):
        """
        A function which generates random states for the sampler which obey the given gauge
        criterion
        """
        if isinstance(sampler, ParallelTemperingSampler):
            f = sampler.n_batches
        else:
            f = sampler.n_chains_per_rank

        return sampler.hilbert.random_state(
            key=key,
            size=f,
        )

    def transition(  # pylint: disable=R0913
        self,
        sampler,
        machine,
        params,
        sampler_state,
        key,
        σ,  # pylint: disable=C2401
    ):
        """
        The transition kernel for the sampler which specifies the proposed configurations
        """

        # Load the Hilbert space of the sampler
        hilb = sampler.hilbert

        # split keys
        key, subkey = jax.random.split(key, 2)

        σₚ = hilb.random_state(  # pylint: disable=C2401
            key=subkey,
            size=σ.shape[0],
        )

        # log_q_scalar = -jnp.log(hilb.local_size_per_site**(hilb.gauge_dims * len(hilb.n_free)))
        #
        # if σ.ndim == 1:
        #     log_q = log_q_scalar
        # else:
        #     log_q = jnp.full((σ.shape[0],), log_q_scalar)

        return σₚ, None


@struct.dataclass
class U1GaugeSampler(MetropolisRule):
    """
    This class implements a U(1) gauge sampler which samples states from the Hilbert space based
    on a specified gauge fixing.

    The proposed states are ones which lie in the neighborhood of the input states. This is done by
    flipping one of the free edges' quantum numbers and reimposing the gauge fixing on the slave
    edges.
    """

    def init_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        key: PRNGKeyT,
    ) -> Optional[Any]:

        return sampler.hilbert.random_state(key=key, size=1)

    def random_state(  # pylint: disable=R0913
        self,
        sampler,
        machine,
        params,
        sampler_state,
        key,
    ):
        """
        A function which generates random states for the sampler which obey the given gauge
        criterion
        """
        if isinstance(sampler, ParallelTemperingSampler):
            f = sampler.n_batches
        else:
            f = sampler.n_chains_per_rank

        return sampler.hilbert.random_state(
            key=key,
            size=f,
        )

    def transition(  # pylint: disable=R0913
        self,
        sampler,
        machine,
        params,
        sampler_state,
        key,
        σ,  # pylint: disable=C2401
    ):
        """
        The transition kernel for the sampler which specifies the proposed configurations
        """

        # Load the Hilbert space of the sampler
        hilb = sampler.hilbert

        # split keys
        key, subkey = jax.random.split(key, 2)

        # change the basis label of one independent edge at a time and reimpose the fixing
        σₚ = hilb.flip_gi_state_sgd(
            σ,
            subkey,
            adjacency=False,
            number_of_edges=1,  # pylint: disable=C2401
        )

        # # compute log q(σ -> σₚ)
        # # number of independent ("free") edges across all gauge dimensions
        # n_free = hilb.gauge_dims * len(hilb.n_free)
        #
        # # number of allowed basis values for a free edge
        # num_vals = hilb.local_size_per_site
        #
        # # log probability for picking one free edge uniformly and one value uniformly
        # log_q_scalar = -jnp.log(n_free) - jnp.log(num_vals)
        #
        # # broadcast over batch if needed
        # if σ.ndim == 1:
        #     log_q = log_q_scalar
        # else:
        #     B = σ.shape[0]
        #     log_q = jnp.full((B,), log_q_scalar, dtype = jnp.result_type(0.0))

        return σₚ, None


@struct.dataclass
class U1InvariantPlaquetteSampler(MetropolisRule):
    """
    This class implements a U(1) gauge sampler which samples states from the Hilbert space based
    on a specified gauge fixing.

    The proposed states are ones which lie in the neighborhood of the input states. This is done by
    flipping one of the free edges' quantum numbers and reimposing the gauge fixing on the slave
    edges.
    """

    def init_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        key: PRNGKeyT,
    ) -> Optional[Any]:

        return sampler.hilbert.random_state(key=key, size=1)

    def random_state(  # pylint: disable=R0913
        self,
        sampler,
        machine,
        params,
        sampler_state,
        key,
    ):
        """
        A function which generates random states for the sampler which obey the given gauge
        criterion
        """
        if isinstance(sampler, ParallelTemperingSampler):
            f = sampler.n_batches
        else:
            f = sampler.n_chains_per_rank

        return sampler.hilbert.random_state(
            key=key,
            size=f,
        )

    def transition(  # pylint: disable=R0913
        self,
        sampler,
        machine,
        params,
        sampler_state,
        key,
        σ,  # pylint: disable=C2401
    ):
        """
        The transition kernel for the sampler which specifies the proposed configurations
        """

        # Load the Hilbert space of the sampler
        hilb = sampler.hilbert

        # split keys
        key, subkey = jax.random.split(key, 2)

        # change the basis label of one independent edge at a time and reimpose the fixing
        σₚ = hilb.plaquette_flip(
            σ,
            subkey,  # pylint: disable=C2401
        )

        return σₚ, None


@struct.dataclass
class U1GaugeSamplerNonzero(MetropolisRule):
    """
    This class implements a U(1) gauge sampler which samples states from the Hilbert space based
    on a specified gauge fixing.

    The proposed states are ones which lie in the neighborhood of the input states. This is done by
    flipping one of the free edges' quantum numbers and reimposing the gauge fixing on the slave
    edges.
    """

    def init_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        key: PRNGKeyT,
    ) -> Optional[Any]:

        def is_all_zero(x):
            # x has shape (B, dimH) or (dimH,)
            return jnp.all(x == 0, axis=-1)

        hilb = sampler.hilbert

        def sample(key):
            key, k = jax.random.split(key)
            return key, hilb.random_state(key=k, size=1)

        def cond_fun(carry):
            key, x = carry
            return is_all_zero(x)[0]  # shape (1,) → pick scalar

        def body_fun(carry):
            key, _ = carry
            return sample(key)

        key, x0 = sample(key)
        key, x = jax.lax.while_loop(cond_fun, body_fun, (key, x0))

        return x  # shape (1, dimH)

    def random_state(self, sampler, machine, params, sampler_state, key):

        def is_all_zero(x):
            # x has shape (B, dimH) or (dimH,)
            return jnp.all(x == 0, axis=-1)

        hilb = sampler.hilbert

        if isinstance(sampler, ParallelTemperingSampler):
            f = sampler.n_batches
        else:
            f = sampler.n_chains_per_rank

        def propose(key):
            key, k = jax.random.split(key)
            return key, hilb.random_state(key=k, size=f)

        def cond_fun(carry):
            key, x = carry
            # if ANY chain is all zero → reject batch
            return jnp.any(is_all_zero(x))

        def body_fun(carry):
            key, _ = carry
            return propose(key)

        key, x0 = propose(key)
        key, x = jax.lax.while_loop(cond_fun, body_fun, (key, x0))

        return x

    def transition(  # pylint: disable=R0913
        self,
        sampler,
        machine,
        params,
        sampler_state,
        key,
        σ,  # pylint: disable=C2401
    ):
        """
        The transition kernel for the sampler which specifies the proposed configurations
        """
        #
        # hilb = sampler.hilbert
        #
        # # split keys
        # key, subkey = jax.random.split(key)
        #
        # def propose_once(key):
        #     key, k = jax.random.split(key)
        #     σp = hilb.flip_gi_state_sgd(σ, k, adjacency = False, number_of_edges = 1)
        #     return key, σp
        #
        # def cond_fun(carry):
        #     key, σp = carry
        #     # Reject proposal if ANY chain produced the all-zero state
        #     return jnp.any(jnp.all(σp == 0, axis = 1))
        #
        # def body_fun(carry):
        #     key, _ = carry
        #     return propose_once(key)
        #
        # # First proposal
        # key, σp0 = propose_once(subkey)
        #
        # # Regenerate until no chain is all zeros
        # key, σp = jax.lax.while_loop(cond_fun, body_fun, (key, σp0))
        #
        # return σp, None

        hilb = sampler.hilbert
        key, subkey = jax.random.split(key)

        σp = hilb.flip_gi_state_sgd(σ, subkey, adjacency=False, number_of_edges=1)

        is_vac = jnp.all(σp == 0, axis=1)
        σp = jnp.where(is_vac[:, None], σ, σp)

        # symmetric "lazy" kernel means no log-prob correction...
        return σp, None
