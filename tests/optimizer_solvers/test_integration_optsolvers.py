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


from __future__ import annotations

import inspect
import numpy as np
import pytest


def _filter_kwargs(fn, **kwargs):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _extract_x(out):

    if isinstance(out, tuple) and len(out) > 0:
        return out[0]
    return out


def _call_solver_generic(slv, A, b):

    common = dict(
        tol=1e-12,
        atol=0.0,
        maxiter=500,
        rtol=1e-12,
        diag_shift=0.0,
    )

    try:
        kw = _filter_kwargs(slv, **common)
        return slv(A, b, **kw)
    except TypeError:
        pass

    try:
        A_mv = lambda x: A @ x
        kw = _filter_kwargs(slv, **common)
        return slv(A_mv, b, **kw)
    except TypeError:
        pass

    kw = _filter_kwargs(slv, **common)
    obj = slv(A, **kw)

    if callable(obj):
        return obj(b)
    if hasattr(obj, "solve"):
        return obj.solve(b)
    raise TypeError(f"Do not know how to call solver object of type {type(obj)}")


@pytest.mark.parametrize(
    "solver_key",
    [
        "conjugate gradient",
        "cholesky",
        "singular value decomposition",
        "default solver",
        "pseudo inverse",
        "pseudo inverse smooth",
        "lu",
    ],
)
def test_all_built_solvers_can_solve_small_system(solvers_modules, solver_key):

    jnp = pytest.importorskip("jax.numpy")

    pytest.importorskip("netket")

    tmod, bmod, _emod, smod = solvers_modules

    Cfg = tmod.SOLVERS_REGISTRY[solver_key]
    cfg = Cfg()
    slv = bmod.build_solver(cfg)
    assert slv is not None

    A = jnp.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=jnp.float32,
    )
    b = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

    out = _call_solver_generic(slv, A, b)
    x = _extract_x(out)

    x_np = np.asarray(x)
    assert x_np.shape == (4,)
    assert np.isfinite(x_np).all()

    r = np.asarray(A @ x - b)
    assert (
        np.linalg.norm(r) < 1e-4
    ), f"{solver_key}: residual too large {np.linalg.norm(r)}"


def test_solvers_class_builds_same_solver_as_build_solver(solvers_modules):

    pytest.importorskip("netket")
    pytest.importorskip("jax")

    tmod, bmod, _emod, smod = solvers_modules
    key = "cholesky"

    s = smod.Solvers(key)
    slv1 = s.solver

    cfg = tmod.SOLVERS_REGISTRY[key]()
    slv2 = bmod.build_solver(cfg)

    assert slv1 is slv2
