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


# Copyright 2022 The NetKet Authors - All rights reserved.
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
NOTE: part(s) of, or the entire content, of this file is obtained from NetKet's source code
      the original copyright mentioned above applies.
"""

# pylint: skip-file

"""
This file includes the netket implementation of a weighted sampler
"""

from typing import Any
from typing import Optional
from typing import Tuple

from netket.utils.types import Array
from netket.utils.types import PyTree
from netket.utils.types import PRNGKeyT
from netket.sampler import MetropolisRule

import jax
import flax.linen as nn


class WeightedSamplerRule(MetropolisRule):
    """
    A Metropolis sampling rule that can be used to combine different rules acting
    on different subspaces of the same tensor-hilbert space.
    """

    probabilities: Array
    rules: Tuple[MetropolisRule, ...]

    def __init__(
        self,
        probabilities: Array,
        rules: Tuple[MetropolisRule, ...],
    ):
        self.probabilities = probabilities
        self.rules = rules
        super().__init__()

    def __post_init__(self):
        if not isinstance(self.probabilities, jax.Array):
            object.__setattr__(
                self, "probabilities", jax.numpy.array(self.probabilities)
            )

        if not isinstance(self.rules, (tuple, list)):
            raise TypeError(
                "The second argument (rules) must be a tuple of `MetropolisRule` "
                f"rules, but you have passed {type(self.rules)}."
            )

        if len(self.probabilities) != len(self.rules):
            raise ValueError(
                "Length mismatch between the probabilities and the rules: probabilities "
                f"has length {len(self.probabilities)} , rules has length {len(self.rules)}."
            )

    def init_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        key: PRNGKeyT,
    ) -> Optional[Any]:
        N = len(self.probabilities)
        keys = jax.random.split(key, N)
        return tuple(
            self.rules[i].init_state(sampler, machine, params, keys[i])
            for i in range(N)
        )

    def reset(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "sampler.SamplerState",  # noqa: F821
    ) -> Optional[Any]:
        rule_states = []
        for i in range(len(self.probabilities)):
            # construct temporary sampler and rule state with correct sub-hilbert and
            # sampler-state objects.
            _state = sampler_state.replace(rule_state=sampler_state.rule_state[i])
            rule_states.append(self.rules[i].reset(sampler, machine, params, _state))
        return tuple(rule_states)

    def transition(self, sampler, machine, params, sampler_state, key, σ):
        N = len(self.probabilities)
        keys = jax.random.split(key, N + 1)

        σps = []
        log_prob_corrs = []
        for i in range(N):
            # construct temporary rule state with correct sampler-state objects
            _state = sampler_state.replace(rule_state=sampler_state.rule_state[i])

            σps_i, log_prob_corr_i = self.rules[i].transition(
                sampler, machine, params, _state, keys[i], σ
            )

            σps.append(σps_i)
            log_prob_corrs.append(log_prob_corr_i)

        indices = jax.random.choice(
            keys[-1],
            N,
            shape=(sampler.n_chains_per_rank,),
            p=self.probabilities,
        )

        batch_select = jax.vmap(lambda σ, i: σ[i], in_axes=(1, 0), out_axes=0)
        σp = batch_select(jax.numpy.stack(σps), indices)

        # if not all log_prob_corr are 0, convert the Nones to 0s
        if any(x is not None for x in log_prob_corrs):
            log_prob_corrs = jax.numpy.stack(
                [x if x is not None else 0 for x in log_prob_corrs]
            )
            log_prob_corr = batch_select(log_prob_corrs, indices)
        else:
            log_prob_corr = None

        return σp, log_prob_corr

    def __repr__(self):
        return f"WeightedRule(probabilities={self.probabilities}, rules={self.rules})"
