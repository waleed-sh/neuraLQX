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

import pytest

from neuralqx.solver import deserialize_MultiMCState
from neuralqx.solver import serialize_MultiMCState


def test_multi_mcstate_public_container_roundtrip(mcstate, nqx, jnp):
    second = nqx.vqs.MCState(
        mcstate.sampler,
        mcstate.model,
        n_samples=mcstate.n_samples,
        n_discard_per_chain=mcstate.n_discard_per_chain,
        seed=124,
        sampler_seed=457,
        chunk_size=mcstate.chunk_size,
    )
    second.sample(n_samples=mcstate.n_samples, n_discard_per_chain=0)

    state = nqx.vqs.MultiMCState([mcstate, second])
    payload = serialize_MultiMCState(state)
    restored = deserialize_MultiMCState(state, payload)

    assert payload["kind"] == "MultiMCState"
    assert payload["schema_version"] == 1
    assert restored is not state
    assert restored.n_states == 2
    assert isinstance(restored, nqx.vqs.MultiMCState)
    assert jnp.allclose(
        restored.to_array(state=0),
        mcstate.to_array(),
        rtol=1e-7,
        atol=1e-8,
    )

    bad_kind = dict(payload, kind="MCState")
    with pytest.raises(ValueError, match="kind"):
        deserialize_MultiMCState(state, bad_kind)

    bad_schema = dict(payload, schema_version=999)
    with pytest.raises(ValueError, match="schema_version"):
        deserialize_MultiMCState(state, bad_schema)


def test_multi_mcstate_rejects_mismatched_setter_lengths(mcstate, nqx):
    state = nqx.vqs.MultiMCState([mcstate, mcstate])

    with pytest.raises(ValueError, match="n_samples"):
        state.n_samples = [32]

    with pytest.raises(ValueError, match="parameters"):
        state.parameters = [mcstate.parameters]


def test_single_multi_mcstate_accepts_raw_state_pytrees(mcstate, nqx, jnp):
    state = nqx.vqs.MultiMCState([mcstate])

    state.parameters = mcstate.parameters
    state.variables = mcstate.variables

    assert jnp.allclose(
        state.to_array(state=0),
        mcstate.to_array(),
        rtol=1e-7,
        atol=1e-8,
    )
