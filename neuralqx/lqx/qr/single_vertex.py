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


"""Single-vertex QR model facade."""

from __future__ import annotations

from neuralqx.lqx.base import LQXModel
from neuralqx.operator.prebuilt import qr


class LqxQR(LQXModel):
    r"""Operator factory for the three-edge single-vertex quantum-reduced model.

    The facade provides bounded spin shifts, flux powers, and the symmetric
    Euclidean and Lorentzian constraints. The full constraint is
    $H=H_E+H_L$. The component defaults use unit lapse, flux power $1/4$,
    and Immirzi parameter one. Use `get_euclidean_constraint` and
    `get_lorentzian_constraint` for parameterized construction, then assign
    their results to the component properties to change the model total.
    The inherited quadratic property represents $HH^\dagger$.

    Constraint construction requires one gauge copy, three base edges, and
    non-negative spins. For example, use
    `U1Hilbert(graph, cutoff=2, positive_qn=True)`. Edge indices 0, 1, and 2
    label the reduced spatial directions. The cutoff is noncyclic for the QR
    shift operators.

    Holonomies use the bounded single-edge QR shifts and require an explicit
    site. Area, volume, and mass defaults are not supplied by this facade.

    Args:
        hilbert (AbstractU1Hilbert): U(1) Hilbert space carrying the reduced spins.
        graph (AbstractGraph | None): Optional graph, which must be the object attached to `hilbert`.
    """

    def __repr__(self) -> str:
        """Return the model type and its defining Hilbert-space and graph parameters."""
        return f"LqxQR(hilbert={self.hilbert!r}, graph={self.graph!r})"

    def flux(self, site: int, *, power: float = 1.0, inverse: bool = False):
        """Construct a QR flux power on one flat site.

        Args:
            site: Flat spin index.
            power: Exponent of the spin eigenvalue.
            inverse: Use inverse powers with the zero eigenvalue mapped to zero.

        Returns:
            result (qr.QRFluxOperator): A diagonal `QRFluxOperator`.
        """
        return qr.QRFluxOperator(self.hilbert, site=site, power=power, inverse=inverse)

    def creation(self, site: int, *, n: int = 1):
        """Construct a bounded positive QR row shift.

        Args:
            site: Flat spin index to shift.
            n: Positive number of local lattice steps.

        Returns:
            result (qr.QRCreationOperator): A `QRCreationOperator` with zero amplitude beyond the cutoff.
        """
        return qr.QRCreationOperator(self.hilbert, site=site, n=n)

    def annihilation(self, site: int, *, n: int = 1):
        """Construct the bounded negative QR row shift.

        Args:
            site: Flat spin index to shift.
            n: Positive number of local lattice steps.

        Returns:
            result (qr.QRAnnihilationOperator): The adjoint of the corresponding `QRCreationOperator`.
        """
        return qr.QRAnnihilationOperator(self.hilbert, site=site, n=n)

    def get_euclidean_constraint(
        self,
        *,
        lapse: float = 1.0,
        power: float = 0.25,
        immirzi: float = 1.0,
    ):
        """Construct the symmetric QR Euclidean constraint.

        Args:
            lapse: Real multiplier of the Euclidean term.
            power: Flux-sandwich exponent, normally one quarter.
            immirzi: Nonzero real Barbero-Immirzi parameter.

        Returns:
            result (qr.QREuclideanConstraintOperator): A Hermitian `QREuclideanConstraintOperator`.
        """
        return qr.QREuclideanConstraintOperator(
            self.hilbert,
            lapse=lapse,
            power=power,
            immirzi=immirzi,
        )

    def get_holonomy(
        self, site: int | None = None, *, n: int = 1, adjoint: bool = False
    ):
        """Construct a bounded QR holonomy shift at one reduced direction.

        Args:
            site: Flat spin index. An explicit site is required.
            n: Positive number of lattice steps.
            adjoint: Select the negative row shift instead of the positive one.

        Returns:
            result (ComputationalOperator): A QR creation or annihilation operator.
                The action vanishes beyond the cutoff instead of wrapping.
        """
        if site is None:
            self._not_implemented("holonomy", "Select a site with get_holonomy(site).")
        return self.annihilation(site, n=n) if adjoint else self.creation(site, n=n)

    def get_lorentzian_constraint(self, *, immirzi: float = 1.0):
        """Construct the QR Lorentzian constraint in its unit-lapse normalization.

        Args:
            immirzi: Nonzero real Barbero-Immirzi parameter.

        Returns:
            result (qr.QRLorentzianConstraintOperator): A Hermitian, negative-semidefinite `QRLorentzianConstraintOperator`.
        """
        return qr.QRLorentzianConstraintOperator(self.hilbert, immirzi=immirzi)

    def total_constraint(
        self,
        *,
        lapse: float = 1.0,
        power: float = 0.25,
        immirzi: float = 1.0,
    ):
        """Construct the sum of the QR Euclidean and Lorentzian terms.

        Args:
            lapse: Multiplier of the Euclidean contribution. The Lorentzian
                contribution retains its unit-lapse normalization.
            power: Flux exponent in the Euclidean sandwich factors.
            immirzi: Nonzero real parameter shared by both contributions.

        Returns:
            result (SumOperator): A lazy sum representing the total constraint. This does not replace
                the cached default returned by `constraint`.
        """
        return self.get_euclidean_constraint(
            lapse=lapse,
            power=power,
            immirzi=immirzi,
        ) + self.get_lorentzian_constraint(immirzi=immirzi)


__all__ = ["LqxQR"]
