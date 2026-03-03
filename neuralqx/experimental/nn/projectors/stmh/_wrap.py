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

from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional

import jax.numpy as jnp
from flax import linen as nn

# TODO: move to types
Array = Any


class SingleTrunkMultiHeadLogPsi(nn.Module):
    r"""
    Generic single-trunk multi-head wrapper for Flax models.

    The wrapped ``trunk`` is expected to return **features per configuration**. The preferred
    convention NetKet's, which is a tensor whose leading axis is the batch axis, for example ``(batch, h)``.
    The wrapper flattens any trailing feature dimensions and applies lightweight linear heads
    to produce ``K`` log-amplitudes, one per target state/head.

    By default, the wrapper outputs complex log-amplitudes

    .. math::
        \log\psi_k(x) = a_k(f(x)) + i\, b_k(f(x)),

    where ``a_k`` and ``b_k`` are independent affine maps implemented as two Dense layers.


    Notes
    -----
    - For NetKet/MCState compatibility, use :class:`STMHHeadView` to expose a single head with
      output shape ``(batch,)``.
    - If your existing model currently returns a scalar ``(batch,)`` log-amplitude, you should
      ideally factor it into ``trunk`` + head. This wrapper *can* lift a scalar trunk output to a
      one-dimensional feature, but that yields only a 1D shared feature space.
    """

    trunk: nn.Module
    """Any Flax module that maps a batch of basis configurations to features."""

    n_heads: int
    """Number of output heads ``K``."""

    latent_dim: Optional[int] = None
    """Optional feature dimension after a projection layer. If provided and the trunk output feature  width differs, 
    the wrapper inserts a learned projection ``trunk_proj``.
    """

    trunk_output: Literal["auto", "features", "dict", "tuple"] = "auto"
    """
    How to interpret the trunk return value.
        - ``"auto"``: if the trunk returns a dict with ``"features"`` use that, if it returns
          a tuple/list use the first element, otherwise use the value directly.
        - ``"features"``: use the trunk return value directly.
        - ``"dict"``: expect a dict and extract ``features_key``.
        - ``"tuple"``: expect a tuple/list and take item ``tuple_index``.
    """

    complex_logpsi: bool = True
    """If ``True`` (default), output complex log-amplitudes using separate real/imag linear heads. If ``False``, output 
    real values (e.g. amplitude-only ansatz).
    """

    flatten_features: bool = True
    """Flatten all trailing feature axes to a vector per sample."""

    features_key: str = "features"
    """Key used when ``trunk_output='dict'``."""

    tuple_index: int = 0
    """Index used when ``trunk_output='tuple'``."""

    kernel_init: Callable[..., Any] = nn.initializers.glorot_uniform()
    """Initialisers for head Dense layers (and optional projection layer)."""

    bias_init: Callable[..., Any] = nn.initializers.zeros
    """Initialisers for biases for the head layers."""

    dtype: Any = None
    """Flax Dense dtypes."""

    param_dtype: Any = jnp.float32
    """Parameter dtypes for Flax."""

    def _extract_features(self, trunk_out: Any) -> Array:
        mode = self.trunk_output

        if mode == "auto":
            if isinstance(trunk_out, dict):
                if self.features_key in trunk_out:
                    feats = trunk_out[self.features_key]
                else:
                    raise KeyError(
                        f"trunk_output='auto' saw dict output but key '{self.features_key}' was not found."
                    )
            elif isinstance(trunk_out, (tuple, list)):
                feats = trunk_out[self.tuple_index]
            else:
                feats = trunk_out
        elif mode == "features":
            feats = trunk_out
        elif mode == "dict":
            if not isinstance(trunk_out, dict):
                raise TypeError(
                    "trunk_output='dict' requires the trunk to return a dict."
                )
            feats = trunk_out[self.features_key]
        elif mode == "tuple":
            if not isinstance(trunk_out, (tuple, list)):
                raise TypeError(
                    "trunk_output='tuple' requires the trunk to return a tuple/list."
                )
            feats = trunk_out[self.tuple_index]
        else:  # pragma: no cover
            raise ValueError(f"Unsupported trunk_output={mode!r}")

        feats = jnp.asarray(feats)

        # Lift scalar/trivial outputs to feature vectors. The expected convention is batch-first.
        if feats.ndim == 0:
            feats = feats.reshape(1, 1)
        elif feats.ndim == 1:
            # Interpret as batch of scalar features: (batch,) -> (batch, 1)
            feats = feats[:, None]
        elif self.flatten_features and feats.ndim > 2:
            feats = feats.reshape((feats.shape[0], -1))

        return feats

    @nn.compact
    def __call__(
        self, x: Array, *, head: Optional[int] = None, return_features: bool = False
    ):
        trunk_out = self.trunk(x)
        feats = self._extract_features(trunk_out)

        if self.latent_dim is not None and feats.shape[-1] != self.latent_dim:
            feats = nn.Dense(
                self.latent_dim,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="trunk_proj",
            )(feats)

        if self.complex_logpsi:
            re = nn.Dense(
                self.n_heads,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="head_re",
            )(feats)
            im = nn.Dense(
                self.n_heads,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="head_im",
            )(feats)
            out = re + 1j * im
        else:
            out = nn.Dense(
                self.n_heads,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="head_re",
            )(feats)

        if head is not None:
            out = out[..., int(head)]

        if return_features:
            return out, feats
        return out


class STMHHeadView(nn.Module):
    """
    Exposes a single head from :class:`SingleTrunkMultiHeadLogPsi` as a standard scalar-output model.

    This is the compatibility wrapper you should hand to ``MCState`` so that each head behaves like a
    normal NQS model returning ``(batch,)`` log-amplitudes.
    """

    base: nn.Module
    head: int

    @nn.compact
    def __call__(self, x: Array):
        return self.base(x, head=int(self.head))


class STMHAllHeadsView(nn.Module):
    """Thin alias wrapper returning all heads explicitly (shape ``(batch, K)``)."""

    base: nn.Module

    @nn.compact
    def __call__(self, x: Array):
        return self.base(x, head=None)


def wrap_trunk_as_stmh(
    trunk: nn.Module,
    n_heads: int,
    **kwargs,
) -> SingleTrunkMultiHeadLogPsi:
    """
    Convenience constructor for a single-state multi-head Ansatz. You should provide a trunk-compatible Flax
    based model at construction time and specify the number of desired heads (i.e. number of orthogonal states).

    Examples
    --------
    >>> stmh = wrap_trunk_as_stmh(MyTrunk(...), n_heads=2)  # doctest: +SKIP
    >>> head0_model = STMHHeadView(stmh, 0)  # doctest: +SKIP
    >>> head1_model = STMHHeadView(stmh, 1)  # doctest: +SKIP
    """

    return SingleTrunkMultiHeadLogPsi(trunk=trunk, n_heads=int(n_heads), **kwargs)


def make_stmh_head_models(
    base_stmh_model: nn.Module, n_heads: int
) -> list[STMHHeadView]:
    """Create ``n_heads`` scalar-output Flax models selecting each head of ``base_stmh_model``."""

    return [STMHHeadView(base=base_stmh_model, head=k) for k in range(int(n_heads))]
