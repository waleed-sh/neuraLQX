#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


# Copyright 2021 The NetKet Authors - All rights reserved.
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

# pylint: skip-file
# fmt: off

"""
NOTE: part(s) of, or the entire content, of this file is obtained from NetKet's source code
      the original copyright mentioned above applies.
"""

from typing import Any, Union, Sequence
import warnings

from flax.core.scope import CollectionFilter, DenyList  # noqa: F401
from netket.operator import AbstractOperator

from netket.operator._abstract_observable import AbstractObservable
from netket.vqs import MCState

from neuralqx.vqs import expect_and_grad
from .state import MCState as NQXMCState
from .expect_grad import expect_and_grad_nonhermitian

def ignore_chunk_warning(vstate, operator, chunk_size, name=""):
    return f"""
            Ignoring chunk_size={chunk_size} for {name} method with signature
            ({type(vstate)}, {type(operator)}) because no implementation supporting
            chunking for this signature exists.
            """

#
#
#   Unspecified batch sizes

@expect_and_grad.dispatch
def expect_and_grad_chunking_unspecified(  # noqa: F811
    vstate: Union[MCState, NQXMCState],
    operator: Union[AbstractOperator, AbstractObservable],
    **kwargs,
):
    return expect_and_grad(vstate, operator, None, **kwargs)

@expect_and_grad.dispatch
def expect_and_grad_chunking_unspecified(  # noqa: F811
    vstate: Union[MCState, NQXMCState],
    operator: Sequence[Union[AbstractOperator, AbstractObservable]],
    **kwargs,
):
    return expect_and_grad(vstate, operator, None, **kwargs)

#
#
#   No implementations

@expect_and_grad.dispatch(precedence=-10)
def expect_and_grad_fallback(  # noqa: F811
    vstate: Union[MCState, NQXMCState],
    operator: Union[AbstractOperator, AbstractObservable],
    chunk_size: int | tuple,
    *args,
    **kwargs,
):
    warnings.warn(
        ignore_chunk_warning(vstate, operator, chunk_size, name="expect_and_grad")
    )
    return expect_and_grad(vstate, operator, None, *args, **kwargs)

@expect_and_grad.dispatch(precedence=-10)
def expect_and_grad_fallback(  # noqa: F811
    vstate: Union[MCState, NQXMCState],
    operator: Sequence[Union[AbstractOperator, AbstractObservable]],
    chunk_size: int | tuple,
    *args,
    **kwargs,
):
    warnings.warn(
        ignore_chunk_warning(vstate, operator, chunk_size, name="expect_and_grad")
    )
    return expect_and_grad(vstate, operator, None, *args, **kwargs)

# non-hermitian path runs unchunked
@expect_and_grad_nonhermitian.dispatch(precedence=-10)
def expect_and_grad_nonhermitian_chunk_fallback(
    vstate: Union[MCState, NQXMCState],
    Ô,
    chunk_size: Any,
    **kwargs,
):
    warnings.warn(
        ignore_chunk_warning(vstate, Ô, chunk_size, name="expect_and_grad_nonhermitian")
    )
    return expect_and_grad_nonhermitian(vstate, Ô, None, **kwargs)
