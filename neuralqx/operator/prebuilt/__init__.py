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


"""Physical model operators and U(1) charge-label projectors.

The public kernels return matrix rows: an input configuration is the bra
index and each connected configuration is the ket index. QR shifts stop at
the spin cutoff. WCL and spherical shifts use cyclic charge arithmetic.
All geometric expressions use the dimensionless normalizations described
on the individual classes.

`ChargeColoringOperator` selects a charge at one flat component, while
`ColoringOperator` selects a complete edge charge vector. Both are diagonal
projectors whose expectations give probabilities in the variational state.

The non-SU(2) implementations were compared with upstream neuraLQX commit
`f62ed1ba9a56be9887c7bb547489f40c7c2d95bd`. The mathematical regression tests
also check eigenvalues and conjugate-transpose identities independently of
upstream, including the corrected planar Thiemann adjoint.
"""

from . import bojowald_swiderski
from . import coloring
from . import qr
from . import su2
from . import wcl
from .coloring import ChargeColoringOperator
from .coloring import ColoringOperator
from .coloring import n_point_function

__all__ = [
    "ChargeColoringOperator",
    "ColoringOperator",
    "bojowald_swiderski",
    "coloring",
    "n_point_function",
    "qr",
    "su2",
    "wcl",
]
