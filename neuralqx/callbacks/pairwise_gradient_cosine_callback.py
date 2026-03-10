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
EXPERIMENTAL Callback for monitoring gradient alignment between two operators during VMC.

This module defines a NetKet driver callback that computes the cosine of the angle
between the parameter gradients of two operators A and B at every optimisation step.
It is useful for diagnosing frustration between objectives, for example between a
constraint and an added penalty term.

The callback can both log values into the driver log data and optionally print a
readable summary on the global MPI master process.
"""

import jax.numpy as jnp
from jax._src.flatten_util import ravel_pytree

from netket.operator import AbstractOperator
from netket.utils import struct

from neuralqx.utils import distributed as _dist
from neuralqx.utils.io.printing import NQXPrinter
from neuralqx.utils.experimental import experimental


@experimental
class GradientGradientCosineCallback(struct.Pytree, mutable=True):
    """
    Compute the gradient-gradient cosine between two operators during optimisation.

    At each driver step, the callback computes gradients of the expectations of A and B
    with respect to the variational parameters, flattens them, and evaluates

    cos(phi) = Re( <gA, gB> ) / (||gA|| * ||gB|| + eps)

    Interpretation guide
    - cos(phi) close to 1 suggests little or no frustration and gradients align
    - cos(phi) close to -1 suggests strong frustration and gradients oppose
    - cos(phi) close to 0 suggests near orthogonality

    On the global MPI master process, the callback stores results in `log_data` under
    the keys "GradientGradientCosine" and "expectB". When `display` is True, it also
    prints the cosine, both gradient norms, and the expectation value of B.

    :param A: First operator whose expectation gradient is used.
    :param B: Second operator whose expectation gradient is used.
    :param display: If True, print a summary each step. If False, only log values.
    :return: None.
    """

    A: AbstractOperator = struct.field(pytree_node=False)
    """The first operator to consider in the gradient-gradient cosine"""

    B: AbstractOperator = struct.field(pytree_node=False)
    """The second operator to consider in the gradient-gradient cosine"""

    eps: float = struct.field(pytree_node=False)
    """A small correction factor for the cosine computation"""

    _p: NQXPrinter = struct.field(pytree_node=False)
    """A neuraLQX printer"""

    display: bool = struct.field(pytree_node=False)
    """A flag to determine whether printing should be done, or only logging"""

    def __init__(
        self,
        A: AbstractOperator,
        B: AbstractOperator,
        display: bool = True,
    ):
        """
        Create the callback.

        :param A: First operator whose expectation gradient is used.
        :param B: Second operator whose expectation gradient is used.
        :param display: If True, print a summary each step. If False, only log values.
        :return: None.
        """

        self.A = A
        self.B = B

        self._p = NQXPrinter()

        self.eps = 1.0e-16

        self.display = display

    def __call__(self, step, log_data, driver):
        """
        Compute and report the gradient gradient cosine for the current optimisation step.

        This method
        - computes expectation and gradients for A and B via the driver state
        - flattens both gradient pytrees into vectors
        - computes the cosine of the angle between the two gradient vectors with a small
          stabilising `eps` in the denominator
        - logs results on the global MPI master process into `log_data`
        - optionally prints a summary when `display` is True

        :param step: Current optimisation step index.
        :param log_data: Mutable mapping used by the driver for logging. Updated on the
            global MPI master process with "GradientGradientCosine" and "expectB".
        :param driver: NetKet driver providing the current `state`, and access to the loss
            context for expectation and gradient evaluation.
        :return: Always True to indicate the optimisation should continue.
        """

        # compute expectations and gradients
        stats_A, grad_A = driver.state.expect_and_grad(self.A)
        stats_B, grad_B = driver.state.expect_and_grad(self.B)

        # flatten
        gA, _ = ravel_pytree(grad_A)
        gB, _ = ravel_pytree(grad_B)

        # compute the gradient-gradient angle
        cos_phi = jnp.vdot(gA, gB).real / (
            jnp.linalg.norm(gA) * jnp.linalg.norm(gB) + self.eps
        )

        # compute norms
        norm_A = jnp.linalg.norm(gA)
        norm_B = jnp.linalg.norm(gB)

        # expectation value of B
        expect_B = driver.state.expect(self.B)

        _dist.barrier()

        if _dist.is_global_master():
            log_data["GradientGradientCosine"] = float(cos_phi)
            log_data["expectB"] = expect_B

        if self.display:
            self._p.print(
                f"Gradient-Gradient Cosine at step {step} = {cos_phi}"
                f"\n"
                f"‖∇A‖ = {norm_A}, ‖∇B‖ = {norm_B}"
                f"\n"
                f"<B> = {expect_B}"
            )

        _dist.barrier()

        return True
