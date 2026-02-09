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

from jax import random
from jax import vmap
from jax import lax
import jax.numpy as jnp

from netket.sampler.rules import MetropolisRule
from netket.utils import struct

from neuralqx.utils.experimental import experimental


@experimental
@struct.dataclass
class MuSampler(MetropolisRule):
    """
    A sampler which is specific to the spherical model. It operates under the guide that the volume
    on the internal vertices of the half ladder graph are non-zero. This is done by sampling states
    where the quantum numbers associated to the mu-edges is never allowed to be zero.
    """

    def transition(self, sampler, machine, parameters, state, key, σ):
        """
        For each chain b, do the following:
          1) pick an index i_b in {0, ..., L − 1} at random (L = no. of edges in the graph)
          2) flip σ[b, i_b] to something not equal to its old value, but also not equal to 0 if
           i_b is in the list of forbidden_indices
          3) then, for every f in forbidden_indices, if σp[b, f] == 0 (whether f == i_b or not),
           re‐draw σp[b, f] uniformly from != 0 levels
        """

        # keys
        key_flip, key_enforce = random.split(key, 2)

        # number of chains
        n_chains = σ.shape[0]

        # get the Hilbert space
        hilb = sampler.hilbert

        # the allowed quantum numbers as a JAX array
        # match the σ dtype otherwise a mismatch in the JAX loop
        local_states = jnp.array(hilb.local_states, dtype=σ.dtype)

        # the number of dofs in the space at any given site (since we are homogeneous)
        L = hilb.size

        # the mu indices, for nowthis has to be manually attached to the HomogeneousHilbert!
        forbidden_indices = hilb.graph.edges_mu_idx

        # choose one site per chain to flip
        indxs = random.randint(key_flip, (n_chains,), 0, L)

        # split sub‐keys for each chain
        keys = random.split(key_flip, n_chains)

        def flip_one(subkey, state, idx):
            old = state[idx]

            # build a mask: always ban old and if idx forbidden, also ban zero
            mask_old = local_states != old
            mask_nozero = local_states != 0
            is_forb = jnp.isin(idx, forbidden_indices)
            mask = jnp.where(is_forb, mask_old & mask_nozero, mask_old)

            # normalized probabilities
            p = mask.astype(float)
            p = p / jnp.sum(p)

            # sample the new local value
            new_val = random.choice(subkey, local_states, shape=(), replace=False, p=p)

            # apply the flip
            return state.at[idx].set(new_val), old

        # vectorize the single‐chain flip
        σp, old_vals = vmap(flip_one, in_axes=(0, 0, 0))(keys, σ, indxs)

        # now enforce "no zeros" at all forbidden_indices
        # if the new proposed configuration has no zeros on forbidden indices, this acts as an
        # identity, otherwise it will randomly flip any zeros on forbidden indices to non-zero vals
        def enforce_forbidden_per_chain(args):
            st, subkey = args
            # one sub‐key per forbidden index
            keys2 = random.split(subkey, len(forbidden_indices) + 1)[1:]

            def body_fn(i, state):
                f = forbidden_indices[i]
                # if site f is zero, resample a nonzero

                def resample(x):
                    mask_nz = local_states != 0
                    p_nz = mask_nz.astype(float)
                    p_nz = p_nz / jnp.sum(p_nz)
                    return random.choice(
                        keys2[i], local_states, shape=(), replace=False, p=p_nz
                    )

                new_val = lax.cond(
                    state[f] == 0, resample, lambda _: state[f], operand=None
                )

                return state.at[f].set(new_val)

            return lax.fori_loop(0, len(forbidden_indices), body_fn, st)

        # enforce across the batch
        keys_enf = random.split(key_enforce, n_chains)
        σp = vmap(enforce_forbidden_per_chain)((σp, keys_enf))

        # no log corr here, just flips
        # dev: we need a logprob correction here
        return σp, None
