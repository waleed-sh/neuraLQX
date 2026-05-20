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

# pylint: skip-file
# fmt: off

r"""Experimental bi-covariance gradient for non-Hermitian operators.

For a generic (not necessarily Hermitian) operator ``A``, define

.. math::

    E_A(\theta) = \frac{\langle \psi_\theta | A | \psi_\theta \rangle}
    {\langle \psi_\theta | \psi_\theta \rangle}.

For a real coordinate :math:`\eta`, the exact gradient can be written as

.. math::

    \partial_\eta E_A =
    \operatorname{Cov}_{p_\theta}(\overline{O_\eta}, L_A)
    +
    \operatorname{Cov}_{p_\theta}(O_\eta, \overline{L_{A^\dagger}}),

where

.. math::

    O_\eta(x) &= \partial_\eta \log \psi_\theta(x), \\
    L_A(x) &= \frac{(A\psi_\theta)(x)}{\psi_\theta(x)}, \\
    L_{A^\dagger}(x) &= \frac{(A^\dagger\psi_\theta)(x)}{\psi_\theta(x)}.

In force notation:

.. math::

    F_A = \operatorname{Cov}(\overline{O_j}, L_A), \quad
    F_{A^\dagger} = \operatorname{Cov}(\overline{O_j}, L_{A^\dagger}).

For real parameters, :math:`g_j = F_{A,j} + \overline{F_{A^\dagger,j}}`.


This implementation reuses :func:`neuralqx.vqs.expect_and_forces` for both
``A`` and ``A†``. For sequences we pass the full list at once, which preserves
existing fused/unfused sequence behavior from the public forces API.

For complex holomorphic models, this module supports complex parameters through
the same force/Wirtinger branch used by :func:`neuralqx.vqs.expect_and_forces`.
For non-holomorphic models, it emits a warning and defers to the existing
generic non-Hermitian VJP path by returning ``None`` to the dispatcher.

When all parameter leaves are complex and holomorphicity is satisfied, the
implementation uses only the forward force term ``F_A`` and skips the
adjoint-force evaluation as an optimization.


If an operator does not expose ``adjoint``, this module emits a warning and
returns ``None`` so callers can fall back to the generic non-Hermitian VJP
path.
"""

import warnings
from functools import partial
from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp

from flax.core.scope import CollectionFilter

from netket.operator import AbstractOperator, Squared
from netket.stats import Stats
from netket.utils import is_probably_holomorphic
from netket.utils.types import PyTree
from netket.vqs.mc.mc_state.state import MCState

from neuralqx import cfg
from neuralqx.profile import section as prof_section
from neuralqx.vqs.mc.mc_state.state import MCState as NQXMCState
from neuralqx.vqs import expect_and_forces


def _tree_has_complex_leaves(tree: PyTree) -> bool:
    """Return ``True`` if any parameter leaf has a complex dtype."""
    return any(jnp.iscomplexobj(leaf) for leaf in jax.tree_util.tree_leaves(tree))


def _tree_all_complex_leaves(tree: PyTree) -> bool:
    """Return ``True`` if every parameter leaf has a complex dtype."""
    leaves = jax.tree_util.tree_leaves(tree)
    return len(leaves) > 0 and all(jnp.iscomplexobj(leaf) for leaf in leaves)


def _dtype_is_complex(dtype) -> bool:
    """Return ``True`` when ``dtype`` is complex-typed."""
    if dtype is None:
        return False
    try:
        return bool(jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating))
    except TypeError:
        return False


def _operator_dtype(operator: AbstractOperator):
    """Return operator dtype metadata used for objective-type routing."""
    if isinstance(operator, Squared):
        return operator.parent.dtype

    return operator.dtype


def _requires_complex_objective_fallback(operator: AbstractOperator) -> bool:
    """Return ``True`` when this operator should use generic complex-objective VJP.

    ``Squared`` objectives are real-valued by construction even if the parent
    has a complex dtype, so they never trigger this fallback.
    """
    if isinstance(operator, Squared):
        return False

    return _dtype_is_complex(_operator_dtype(operator))


def _sequence_requires_complex_objective_fallback(
    operators: Sequence[AbstractOperator],
) -> bool:
    """Return ``True`` when any operator in a sequence requires fallback."""
    return any(_requires_complex_objective_fallback(op) for op in operators)


def _is_probably_holomorphic_vstate(vstate: Union[MCState, NQXMCState]) -> bool:
    """Return NetKet holomorphicity diagnostic for a variational state."""
    required = ("_apply_fun", "parameters", "samples", "model_state")
    if vstate is None or not all(hasattr(vstate, attr) for attr in required):
        raise ValueError(
            "Unable to determine whether the state is holomorphic or not."
        )

    return bool(
        is_probably_holomorphic(
            vstate._apply_fun,
            vstate.parameters,
            vstate.samples,
            vstate.model_state,
        )
    )

@jax.jit
def _forces_to_holomorphic_complex_grad(F_A: PyTree, parameters: PyTree) -> PyTree:
    """Return the holomorphic complex-parameter gradient from ``F_A`` only."""
    return jax.tree_util.tree_map(lambda fa, p: fa.astype(p.dtype), F_A, parameters)


@partial(jax.jit, static_argnames=("project_real_for_real_params",))
def _forces_to_biadjoint_grad(
    F_A: PyTree,
    F_Adagger: PyTree,
    parameters: PyTree,
    *,
    project_real_for_real_params: bool,
) -> PyTree:
    """Assemble the bi-covariance gradient from the two force vectors.

    Args:
        F_A: Forces from the forward operator ``A``.
        F_Adagger: Forces from the adjoint operator ``A†``.
        parameters: Parameter pytree used to preserve leaf dtypes and complex/real
            semantics.
        project_real_for_real_params: Whether real-parameter leaves should be
            projected onto the real axis.

    Returns:
        The assembled gradient pytree.
    """

    def _combine(fa, fadj, p):
        correction = fa + jnp.conj(fadj)
        if jnp.iscomplexobj(p):
            return fa.astype(p.dtype)
        if project_real_for_real_params:
            return correction.real.astype(p.dtype)
        return correction.astype(jnp.result_type(correction.dtype, p.dtype))

    return jax.tree_util.tree_map(_combine, F_A, F_Adagger, parameters)


def _resolve_adjoint_operator(operator: AbstractOperator) -> AbstractOperator:
    """Resolve the adjoint operator used in the bi-covariance correction.

    Args:
        operator: Operator appearing in the objective.

    Returns:
        The adjoint operator to be used for the second covariance term.

    Raises:
        AttributeError: If the operator does not expose the required adjoint path.
        NotImplementedError: If the operator family does not implement adjoint.
    """
    if isinstance(operator, Squared):
        # `Squared(A)` corresponds to A†A in NetKet and is Hermitian by construction.
        # Its adjoint path is therefore itself.
        return operator
    return operator.adjoint


def expect_and_grad_biadjoint(
    vstate: Union[MCState, NQXMCState],
    Ô: AbstractOperator,
    chunk_size: Optional[int],
    *,
    mutable: CollectionFilter = False,
) -> Optional[tuple[Stats, PyTree]]:
    """Compute expectation and exact bi-covariance gradient for one operator.

    Args:
        vstate: A Monte Carlo state.
        Ô: Operator whose objective is being differentiated.
        chunk_size: Forwarded to :func:`neuralqx.vqs.expect_and_forces`.
        mutable: Flax mutable collection filter.

    Returns:
        A ``(stats, grad)`` tuple when the operator exposes an adjoint, otherwise
        ``None`` to signal fallback to the generic non-Hermitian VJP path. This
        function also returns ``None`` when ``cfg.EXPERIMENTAL_GRAD`` is
        disabled, for non-holomorphic models, and for complex-typed operator
        objectives (except ``Squared`` which is real-valued by construction).

    Notes:
        For holomorphic models with all-complex parameter leaves, this function
        uses only ``F_A`` (Wirtinger branch) and skips the adjoint-force pass.
    """
    if not cfg.get("EXPERIMENTAL_GRAD"):
        warnings.warn(
            "Experimental bi-covariance gradient is disabled because "
            "`cfg.EXPERIMENTAL_GRAD` is False. Falling back to the generic "
            "non-Hermitian gradient.",
            stacklevel=3,
        )
        return None

    with prof_section(
        "expect_and_grad.biadjoint.resolve_adjoint",
        cat="vqs.grad",
        args={"operator_type": type(Ô).__name__},
    ):
        try:
            Ô_adj = _resolve_adjoint_operator(Ô)
        except (AttributeError, NotImplementedError):
            warnings.warn(
                f"Operator {type(Ô).__name__} does not implement `.adjoint`. "
                "Falling back to the generic non-Hermitian gradient (VJP through "
                "local estimator). Implement `adjoint` on this operator to use the "
                "exact bi-covariance path.",
                stacklevel=3,
            )
            return None

    with prof_section(
        "expect_and_grad.biadjoint.objective_type",
        cat="vqs.grad",
        args={"operator_type": type(Ô).__name__},
    ):
        requires_complex_fallback = _requires_complex_objective_fallback(Ô)
    if requires_complex_fallback:
        warnings.warn(
            "Experimental bi-covariance gradient requires a real-valued "
            "objective. Detected a complex operator dtype, falling back to the "
            "generic non-Hermitian gradient (VJP through local estimator).",
            stacklevel=3,
        )
        return None

    with prof_section(
        "expect_and_grad.biadjoint.holomorphicity",
        cat="vqs.grad",
        args={"operator_type": type(Ô).__name__},
    ):
        requires_holo_check = _tree_has_complex_leaves(vstate.parameters)
        is_holomorphic = (
            _is_probably_holomorphic_vstate(vstate) if requires_holo_check else True
        )
    if requires_holo_check and not is_holomorphic:
        warnings.warn(
            "Experimental bi-covariance gradient requires holomorphicity for "
            "complex-parameter models. Detected a non-holomorphic model via "
            "`netket.utils.is_probably_holomorphic`, falling back to the "
            "generic non-Hermitian gradient (VJP through local estimator).",
            stacklevel=3,
        )
        return None

    with prof_section(
        "expect_and_grad.biadjoint.forces",
        cat="vqs.grad",
        args={"operator_type": type(Ô).__name__},
    ) as sec:
        Ō_stats, F_A = expect_and_forces(vstate, Ô, chunk_size, mutable=mutable)
        sec.sync((Ō_stats, F_A))

    # Holomorphic complex-parameter branch: ∂/∂z̄ uses only F_A.
    if _tree_all_complex_leaves(vstate.parameters):
        with prof_section(
            "expect_and_grad.biadjoint.assemble_holomorphic_complex",
            cat="vqs.grad",
            args={"operator_type": type(Ô).__name__},
        ) as sec:
            grad = _forces_to_holomorphic_complex_grad(F_A, vstate.parameters)
            sec.sync(grad)
        return Ō_stats, grad

    with prof_section(
        "expect_and_grad.biadjoint.forces_adjoint",
        cat="vqs.grad",
        args={"operator_type": type(Ô).__name__},
    ) as sec:
        _, F_Adagger = expect_and_forces(vstate, Ô_adj, chunk_size, mutable=False)
        sec.sync(F_Adagger)

    with prof_section(
        "expect_and_grad.biadjoint.assemble",
        cat="vqs.grad",
        args={"operator_type": type(Ô).__name__},
    ) as sec:
        grad = _forces_to_biadjoint_grad(
            F_A,
            F_Adagger,
            vstate.parameters,
            project_real_for_real_params=True,
        )
        sec.sync(grad)

    return Ō_stats, grad


def expect_and_grad_biadjoint_sequence(
    vstate: Union[MCState, NQXMCState],
    Ô_list: Sequence[AbstractOperator],
    chunk_size: Optional[int],
    *,
    mutable: CollectionFilter = False,
) -> Optional[tuple[Stats, PyTree]]:
    """Compute expectation and exact bi-covariance gradient for a sequence.

    Args:
        vstate: A Monte Carlo state.
        Ô_list: Sequence of operators whose objective contributions are summed.
        chunk_size: Forwarded to :func:`neuralqx.vqs.expect_and_forces`.
        mutable: Flax mutable collection filter.

    Returns:
        A ``(stats, grad)`` tuple when every operator exposes an adjoint, otherwise
        ``None`` to signal fallback to the generic non-Hermitian VJP path. This
        function also returns ``None`` when ``cfg.EXPERIMENTAL_GRAD`` is
        disabled, for non-holomorphic models, and for sequences that contain
        complex-typed operator terms (except ``Squared`` terms).

    Notes:
        For holomorphic models with all-complex parameter leaves, this function
        uses only forward sequence forces ``F_A`` (Wirtinger branch) and skips
        the adjoint-force sequence pass.
    """
    if not cfg.get("EXPERIMENTAL_GRAD"):
        warnings.warn(
            "Experimental bi-covariance sequence gradient is disabled because "
            "`cfg.EXPERIMENTAL_GRAD` is False. Falling back to the generic "
            "non-Hermitian sequence gradient.",
            stacklevel=3,
        )
        return None

    operators = list(Ô_list)

    with prof_section(
        "expect_and_grad.sequence.biadjoint.resolve_adjoint",
        cat="vqs.grad",
        args={"n_operators": int(len(operators))},
    ):
        adjoints: list[AbstractOperator] = []
        for ô in operators:
            try:
                adjoints.append(_resolve_adjoint_operator(ô))
            except (AttributeError, NotImplementedError):
                warnings.warn(
                    f"Operator {type(ô).__name__} does not implement `adjoint`. "
                    "Falling back to the generic non-Hermitian gradient for the entire "
                    "sequence. Implement `adjoint` on every operator to use the exact "
                    "bi-covariance path.",
                    stacklevel=3,
                )
                return None

    with prof_section(
        "expect_and_grad.sequence.biadjoint.objective_type",
        cat="vqs.grad",
        args={"n_operators": int(len(operators))},
    ):
        requires_complex_fallback = _sequence_requires_complex_objective_fallback(
            operators
        )
    if requires_complex_fallback:
        warnings.warn(
            "Experimental bi-covariance sequence gradient requires a real-valued "
            "objective. Detected complex operator dtype in the sequence, "
            "falling back to the generic non-Hermitian sequence gradient.",
            stacklevel=3,
        )
        return None

    with prof_section(
        "expect_and_grad.sequence.biadjoint.holomorphicity",
        cat="vqs.grad",
        args={"n_operators": int(len(operators))},
    ):
        requires_holo_check = _tree_has_complex_leaves(vstate.parameters)
        is_holomorphic = (
            _is_probably_holomorphic_vstate(vstate) if requires_holo_check else True
        )
    if requires_holo_check and not is_holomorphic:
        warnings.warn(
            "Experimental bi-covariance gradient requires holomorphicity for "
            "complex-parameter models. Detected a non-holomorphic model via "
            "`netket.utils.is_probably_holomorphic`, falling back to the "
            "generic non-Hermitian sequence gradient.",
            stacklevel=3,
        )
        return None

    with prof_section(
        "expect_and_grad.sequence.biadjoint.forces",
        cat="vqs.grad",
        args={"n_operators": int(len(operators))},
    ) as sec:
        stats, F_A = expect_and_forces(vstate, operators, chunk_size, mutable=mutable)
        sec.sync((stats, F_A))

    # Holomorphic complex-parameter branch: ∂/∂z̄ uses only F_A.
    if _tree_all_complex_leaves(vstate.parameters):
        with prof_section(
            "expect_and_grad.sequence.biadjoint.assemble_holomorphic_complex",
            cat="vqs.grad",
            args={"n_operators": int(len(operators))},
        ) as sec:
            grad = _forces_to_holomorphic_complex_grad(F_A, vstate.parameters)
            sec.sync(grad)
        return stats, grad

    with prof_section(
        "expect_and_grad.sequence.biadjoint.forces_adjoint",
        cat="vqs.grad",
        args={"n_operators": int(len(operators))},
    ) as sec:
        _, F_Adagger = expect_and_forces(vstate, adjoints, chunk_size, mutable=False)
        sec.sync(F_Adagger)

    with prof_section(
        "expect_and_grad.sequence.biadjoint.assemble",
        cat="vqs.grad",
        args={"n_operators": int(len(operators))},
    ) as sec:
        grad = _forces_to_biadjoint_grad(
            F_A,
            F_Adagger,
            vstate.parameters,
            project_real_for_real_params=True,
        )
        sec.sync(grad)

    return stats, grad
