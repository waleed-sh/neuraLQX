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


"""Configuration for changing the sampling measure without changing the state."""

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImportanceSampling:
    r"""Sample from :math:`|\psi|^{\alpha}` and retain Born expectations.

    The exponent is adapted between batches using stochastic directional
    derivatives of the gradient estimator variance. No parameter Jacobian is
    stored. Normalization, gradients and the quantum geometric tensor all use
    importance weights. Self-normalization has finite-batch bias that vanishes
    as the number of effectively independent samples grows.

    Set ``adaptive=False`` for a fixed exponent. An exponent below two broadens
    the distribution but does not add support at exact wavefunction zeros.
    Operator blurring supplies that additional support when needed.

    Adaptation targets the independent-sample variance of the self-normalized
    gradient using random tangent probes. Updates are clipped and affect only
    the next batch. They do not estimate Markov autocorrelation times or
    guarantee lower variance at every step. Adaptation is skipped for sums
    containing penalties. Their weighted expectations and gradients
    remain available. Use adaptive=False for a fixed sampling law.
    """

    exponent: float = 1.5
    """Initial exponent of the unnormalized sampling density."""

    adaptive: bool = True
    """Whether gradient evaluations tune the exponent for later batches."""

    min_exponent: float = 0.5
    """Lower bound on the sampling exponent."""

    max_exponent: float = 2.0
    """Upper bound on the sampling exponent."""

    learning_rate: float = 0.05
    """Step size for reducing the logarithm of gradient sampling variance."""

    max_change: float = 0.02
    """Largest exponent change from one adaptation update."""

    update_every: int = 10
    """Number of sampled training batches between adaptation updates."""

    probes: int = 2
    """Independent random tangent directions used to estimate gradient variance."""

    min_ess_fraction: float = 0.2
    """Below this weight ESS fraction, adaptation moves toward Born sampling."""

    def __post_init__(self):
        """Reject invalid bounds and adaptation controls before compiling kernels."""
        numbers = (
            self.exponent,
            self.min_exponent,
            self.max_exponent,
            self.learning_rate,
            self.max_change,
            self.min_ess_fraction,
        )
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("Importance-sampling controls must be finite.")
        if not 0 < self.min_exponent <= self.exponent <= self.max_exponent <= 2:
            raise ValueError(
                "Require 0 < min_exponent <= exponent <= max_exponent <= 2."
            )
        if (
            self.learning_rate <= 0
            or self.max_change <= 0
            or not 0 < self.min_ess_fraction <= 1
        ):
            raise ValueError(
                "Use positive adaptation steps and an ESS fraction in (0, 1]."
            )
        for name in ("update_every", "probes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.adaptive) is not bool:
            raise TypeError("adaptive must be a boolean.")


@dataclass(frozen=True, eq=False)
class OperatorBlur:
    r"""Broaden a sample batch through the connections of an operator.

    For ``Squared(A)``, outgoing moves follow columns of ``A`` so configurations
    contributing to :math:`\|A\psi\|^2` can be sampled even at nodes of
    :math:`\psi`. Linear operators use their columns. Sums select one term
    uniformly. Each term draws uniformly from its distinct nonzero off-diagonal
    destinations, with probability ``probability``. Rows without destinations
    stay put. The exact mixture density is evaluated for every retained sample.

    The operation preserves the base chain endpoints and needs no acceptance
    step. It applies to any discrete operator exposing matrix-row connectivity
    and its mathematical adjoint. Connectivity is coalesced before probabilities
    are assigned, including cancellation of duplicate matrix elements.

    Support is extended for the specified operators. An unrelated operator
    can require additional connections. Models must have well-defined
    derivatives on retained configurations, including any fixed nodal masks.
    Taking a logarithm at a parameter-dependent amplitude zero can still
    produce undefined model derivatives.

    Generic kernels execute in fixed-size device blocks. Operators exposing
    compact preparation reuse their sparse buffers and existing caches on
    each device. Reverse degrees require additional connectivity work. No
    dense quadratic operator or full Hilbert-space enumeration is needed.
    """

    operator: Any
    """Discrete operator, squared operator, or sum whose connections define moves."""

    probability: float = 0.5
    """Probability of an off-diagonal move after selecting a term."""

    chunk_size: int = 64
    """Largest source batch used for connectivity and reverse-degree evaluation."""

    def __post_init__(self):
        """Validate a lazy blur specification without evaluating connections."""
        if not math.isfinite(self.probability) or not 0 < self.probability < 1:
            raise ValueError(
                "Operator blur probability must lie strictly between zero and one."
            )
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("Operator blur chunk_size must be a positive integer.")
