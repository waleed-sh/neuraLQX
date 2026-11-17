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


"""Concrete sampled variational state."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

import jax
from neuralqx.hilbert import DiscreteHilbertSpace
from neuralqx.jax.random import as_prng_key
from neuralqx.jax.tree import tree_size
from neuralqx.sampler import MetropolisSampler
from neuralqx.sampler import MetropolisSamplerState
from neuralqx.sampler._samples import adjusted_sample_count
from neuralqx.utils.frameworks import AbstractModelFramework
from neuralqx.utils.frameworks import FrameworkApply
from neuralqx.utils.frameworks import FrameworkMerge
from neuralqx.utils.frameworks import as_framework
from neuralqx.utils.frameworks import static_training_kwargs
from neuralqx.utils.typing import Key

from .base import AbstractVariationalState
from .evaluation import amplitudes_from_log_values
from .evaluation import evaluate_log_values
from .evaluation import flatten_state_batch
from .evaluation import restore_log_values


class VariationalState(AbstractVariationalState):
    r"""A model wavefunction with persistent Monte Carlo chains and cached samples.

    The model represents the log amplitude
    :math:`f_\theta(\sigma)=\log\psi_\theta(\sigma)`. It receives a batch of
    flat Hilbert configurations and returns one real or complex scalar per
    configuration. Complex outputs encode both the amplitude and phase of
    the wavefunction. The sampler targets

    .. math::

        p_\theta(\sigma) =
        \frac{\exp\!\left(p\,\operatorname{Re}f_\theta(\sigma)\right)}
        {\sum_{\sigma'}\exp\!\left(p\,\operatorname{Re}f_\theta(\sigma')\right)},

    where :attr:`machine_pow` supplies :math:`p`. Its default value of two
    gives the Born distribution used for quantum expectation values.

    Passing ``hilbert`` without ``sampler`` constructs a ``MetropolisSampler``
    with a proposal suited to the space. ``U1GaugeInvariantHilbert`` uses
    ``U1GaugeRule`` to update free charges and reconstruct dependent edges.
    ``SU2GaugeInvariantHilbert``, including its ``SU2Hilbert`` alias, uses
    ``SU2MetropolisRule`` with spin and intertwiner moves and the corresponding
    Hastings corrections. Other spaces use ``LocalRule``. To choose chain
    counts, sweep sizes, or proposal settings, pass a configured sampler such
    as ``MetropolisU1Gauge`` or ``MetropolisSU2`` explicitly.

    Construction initializes the model and chain state but does not collect
    samples. The first access to :attr:`samples` draws a batch. Subsequent
    expectation and gradient calls reuse that batch until :meth:`sample`,
    :meth:`reset`, or a model-variable assignment invalidates it. Updating
    :attr:`parameters` refreshes chain probabilities on the next sampling
    operation and normally continues from the existing chain configurations.
    Assign parameters through the property rather than modifying their
    containing dictionary in place, so this invalidation takes effect.

    Args:
        sampler: Configured Metropolis sampler. When supplied, its Hilbert
            space and proposal settings are used unchanged.
        model: Flax Linen module, initialized Flax NNX module, framework
            adapter, or callable log-amplitude model. For a callable, the
            argument convention is controlled by ``expects_variables``.
        hilbert: Hilbert space used to construct the default sampler. Required
            when ``sampler`` is omitted. If both are supplied, their Hilbert
            spaces must agree.
        variables: Complete framework variables. These take precedence over
            variables extracted from ``model`` and over ``parameters`` and
            ``model_state``.
        parameters: Trainable parameter pytree to use when neither explicit
            variables nor variables from an initialized model are available.
        model_state: Non-trainable collections to merge with ``parameters``.
        apply_fn: Callable alternative to ``model``. It returns log amplitudes
            from ``(parameters, states)`` or ``(variables, states)``.
        init_fn: Initializer for a callable model, called as
            ``init_fn(key, (1, hilbert.size), sampler.dtype)`` when variables
            must be created. It may return parameters or a variable dictionary
            containing a ``"params"`` collection.
        expects_variables: Whether a callable model or ``apply_fn`` receives
            the complete variable dictionary instead of the parameter pytree.
        seed: Integer seed or JAX key for model initialization. Also seeds the
            sampler when ``sampler_seed`` is omitted. ``None`` uses seed zero.
        sampler_seed: Independent integer seed or JAX key for the chains.
        n_samples: Requested total number of retained configurations across
            chains. The sampler uses the nearest positive multiple of its
            chain count, choosing the larger multiple on a tie.
        n_discard_per_chain: Number of stored chain points discarded before
            each fresh batch, including batches after parameter updates.
            Each discarded point advances the chain by ``sampler.sweep_size``
            attempted proposals. The default is zero. SU(2) chains start in
            the vacuum, so choose this setting for their thermalization.
        chunk_size: Maximum batch size for chunked model evaluation in
            :meth:`log_value`, estimators, and the QGT. ``None`` evaluates
            those batches without model chunking. Sampler proposals use the
            sampler's own batching.
        mutable: Reserved for mutable model collections. Only ``False`` and
            ``None`` are supported. Assign :attr:`model_state` explicitly to
            update non-trainable collections.
        training_kwargs: Extra keyword arguments passed to model evaluations,
            including evaluations performed by the sampler. These settings
            are copied at construction and must support static JAX use.

    Examples:
        Use a Hilbert-aware default sampler with a log-amplitude model::

            state = VariationalState(hilbert=hilbert, model=model, seed=7)
            stats = state.expect(operator)
            stats, gradient = state.expect_and_grad(operator)

        Customize SU(2) sampling while keeping the same state API::

            sampler = MetropolisSU2(hilbert, n_chains=32, sweep_size=20)
            state = VariationalState(sampler, model, n_samples=1024, n_discard_per_chain=100)
    """

    def __init__(
        self,
        sampler: MetropolisSampler | None = None,
        model: Any | None = None,
        *,
        hilbert: DiscreteHilbertSpace | None = None,
        variables: Any | None = None,
        parameters: Any | None = None,
        model_state: Any | None = None,
        apply_fn: Any | None = None,
        init_fn: Any | None = None,
        expects_variables: bool = False,
        seed: Key | None = None,
        sampler_seed: Key | None = None,
        n_samples: int = 1000,
        n_discard_per_chain: int = 0,
        chunk_size: int | None = None,
        mutable: Any = False,
        training_kwargs: dict[str, Any] | None = None,
        _defer_sampler_initialization: bool = False,
    ) -> None:
        """Initialize model variables and chain state using the settings above."""
        framework, framework_variables = as_framework(
            model,
            apply_fn=apply_fn,
            init_fn=init_fn,
            expects_variables=expects_variables,
        )
        if sampler is None:
            if hilbert is None:
                raise ValueError("Pass either sampler or hilbert.")
            sampler = MetropolisSampler(hilbert)
        elif hilbert is not None and sampler.hilbert != hilbert:
            raise ValueError("sampler.hilbert and hilbert disagree.")

        self._sampler = sampler
        self._hilbert = sampler.hilbert
        self._framework = framework
        model_description = model if model is not None else apply_fn
        if isinstance(model_description, AbstractModelFramework):
            model_description = getattr(
                framework, "model", getattr(framework, "apply_fn", framework)
            )
        self._model_name = getattr(
            model_description, "__name__", type(model_description).__name__
        )
        self._chunk_size = chunk_size
        self._training_kwargs = dict(training_kwargs or {})
        self._apply_variables_kernel = FrameworkApply(
            framework,
            static_training_kwargs(self._training_kwargs),
        )
        self._merge_variables_kernel = FrameworkMerge(framework)
        if mutable is not False and mutable is not None:
            raise NotImplementedError(
                "Mutable model collections are not supported by VariationalState yet. "
                "Pass mutable=False or update model_state explicitly."
            )
        self._mutable = False
        self._n_samples = int(n_samples)
        self._n_discard_per_chain = int(n_discard_per_chain)
        if self._n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if self._n_discard_per_chain < 0:
            raise ValueError("n_discard_per_chain must be non-negative.")

        if variables is None and framework_variables is not None:
            variables = framework_variables
        if variables is None and parameters is not None:
            variables = framework.merge_variables(parameters, model_state)
        if variables is None:

            def initialize_variables():
                return framework.init(
                    as_prng_key(seed),
                    (1, self.hilbert.size),
                    dtype=self.sampler.dtype,
                )

            variables = (
                jax.eval_shape(initialize_variables)
                if _defer_sampler_initialization
                else initialize_variables()
            )
        self._parameters_structure = None
        self._variables = framework.place_variables(variables)
        self._parameters, self._model_state = framework.split_variables(self._variables)
        self._parameters_structure = jax.tree_util.tree_structure(self._parameters)

        self._sampler_key = as_prng_key(seed if sampler_seed is None else sampler_seed)
        self._sampler_state: MetropolisSamplerState | None = None
        self._sampler_state_stale = True
        self._samples: jax.Array | None = None
        self._initialize_importance()
        if not _defer_sampler_initialization:
            self.reset()

    def save(
        self,
        path="vstate.nqx",
        *,
        include_samples: bool = False,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ):
        """Save the current state to a single versioned ``.nqx`` file.

        Parameters, non-trainable collections, RNG implementations and key data,
        chain endpoints, acceptance counters and sampling settings are saved.
        Process-specific sampler root keys are retained as well as the keys
        that advance the chains on individual sample partitions.
        Saving never samples or refreshes stale chain probabilities. Cached
        sample batches are omitted unless ``include_samples=True``. Driver and
        optimizer state are not owned by this object and are not included.

        All initialized JAX processes must call this method with the same path
        on a shared filesystem. Each process writes its local array shards.
        The call blocks until a complete snapshot has been published. Existing
        files require ``overwrite=True``. Replacement is atomic, so a failed
        write leaves the previous checkpoint intact. Temporary staging beside
        the file needs disk space for both the shards and the new archive.

        ``metadata`` accepts additional JSON-compatible annotations. The file
        also records software versions, topology, exact array dtypes and model,
        Hilbert and sampler reconstruction recipes where available. A missing
        ``.nqx`` suffix is added automatically. Returns the resolved path.
        """
        from neuralqx.utils.io.serialization.variational import save_variational_state

        return save_variational_state(
            self,
            path,
            include_samples=include_samples,
            metadata=metadata,
            overwrite=overwrite,
        )

    def load(self, path="vstate.nqx", *, sampler_mode="continue", sampler_seed=None):
        """Restore a checkpoint into this compatible state and return ``self``.

        The model and Hilbert space must describe the saved wavefunction. All
        arrays are validated before this object is changed and are restored on
        its current device placements. Exact shapes and dtypes are required.
        Enable JAX 64-bit support before restoring 64-bit device arrays.

        ``sampler_mode="continue"`` restores chain endpoints, RNGs and counters.
        The chain count, proposal settings and number of sample partitions must
        match. On the same execution configuration, the next explicit sample
        call continues the saved random stream. Different hardware or software
        may change floating-point decisions. Without a saved sample batch, an
        expectation call draws new samples instead of reusing the old batch.

        ``sampler_mode="restart"`` permits a different sampler or sample-device
        count and starts fresh chains. Supply ``sampler_seed`` explicitly in
        this mode. Learned variables and state-level sampling settings are
        restored in both modes. This is a collective call in multi-process JAX.

        Loading temporarily unpacks the file beside its source, then removes
        that staging directory. The parent directory must be writable and
        shared by all processes. Earlier directory checkpoints remain readable.
        """
        from neuralqx.utils.io.serialization.variational import load_variational_state

        return load_variational_state(
            self, path, sampler_mode=sampler_mode, sampler_seed=sampler_seed
        )

    @classmethod
    def from_file(
        cls,
        path,
        *,
        model=None,
        sampler=None,
        sampler_mode="continue",
        sampler_seed=None,
        **model_options,
    ):
        """Construct a variational state from a ``.nqx`` checkpoint.

        Built-in NNX models, their Linen facades, standard Hilbert spaces and
        Metropolis samplers have data-only reconstruction recipes. Graph families
        are reconstructed as canonical ``Graph`` objects with the saved edge
        order, coordinates and embedding map. Chain initialization is skipped
        when continuing saved chains. Built-in model recipes create abstract
        variables, avoiding allocation of throwaway weights before restoration.

        For custom components, supply ``model`` or ``sampler`` explicitly, or
        register a codec through ``utils.io.serialization.register_codec``.
        ``model_options`` supplies constructor arguments such as ``init_fn`` or
        an initial ``variables`` template for a callable model. Checkpoints do
        not contain executable Python, so custom code must already be installed.

        Sampling continuation and distributed participation follow :meth:`load`.
        To restore into an existing model with chosen parameter shardings, call
        that state's :meth:`load` instead.
        """
        from neuralqx.utils.io.serialization.variational import (
            variational_state_from_file,
        )

        return variational_state_from_file(
            cls,
            path,
            model=model,
            sampler=sampler,
            sampler_mode=sampler_mode,
            sampler_seed=sampler_seed,
            **model_options,
        )

    def __repr__(self) -> str:
        """Summarize the model, Hilbert space, and current sampling configuration.

        Parameter counts include trainable entries only. Complex entries count
        once, with the corresponding real-coordinate count shown separately.
        The requested sample count and its chain-count adjustment describe the
        next batch. Cached samples are reported by shape, including manually
        supplied batches. Displaying the state does not draw samples, refresh
        chain probabilities, or evaluate the model.
        """
        leaves = [
            jnp.asarray(leaf) for leaf in jax.tree_util.tree_leaves(self.parameters)
        ]
        parameter_count = sum(leaf.size for leaf in leaves)
        real_coordinates = sum(
            leaf.size * (2 if jnp.iscomplexobj(leaf) else 1) for leaf in leaves
        )
        parameter_details = sorted({str(leaf.dtype) for leaf in leaves})
        if real_coordinates != parameter_count:
            parameter_details.append(f"{real_coordinates:,} real coordinates")
        parameter_summary = f"{parameter_count:,}"
        if parameter_details:
            parameter_summary += f" ({', '.join(parameter_details)})"

        hilbert_details = [f"size={self.hilbert.size}"]
        for name in ("gauge_dimensions", "cutoff", "two_j_max"):
            value = getattr(self.hilbert, name, None)
            if value is not None:
                hilbert_details.append(f"{name}={value}")

        adjusted_samples, chain_length = adjusted_sample_count(
            self.n_samples, self.sampler.n_chains
        )
        sample_details = f"{chain_length:,} per chain"
        if adjusted_samples != self.n_samples:
            sample_details = f"adjusted to {adjusted_samples:,}, {sample_details}"
        cached_samples = "None"
        if self._samples is not None:
            count = self._samples.size // self.hilbert.size
            cached_samples = f"{count:,} (shape={self._samples.shape})"

        policy = ""
        if self.importance_sampling_enabled:
            adaptive = getattr(self.sampler, "importance_sampling", None)
            policy = (
                f"  sampling_exponent={self.sampling_exponent}, "
                f"adaptive={adaptive is not None and adaptive.adaptive}, "
                f"operator_blur={getattr(self.sampler, 'operator_blur', None) is not None},\n"
            )

        return (
            f"{type(self).__name__}(\n"
            f"  hilbert={type(self.hilbert).__name__}({', '.join(hilbert_details)}),\n"
            f"  model={self._model_name} ({self.framework.name}),\n"
            f"  n_parameters={parameter_summary},\n"
            f"  sampler={type(self.sampler).__name__}(rule={type(self.sampler.rule).__name__}, "
            f"n_chains={self.sampler.n_chains}, sweep_size={self.sampler.sweep_size}),\n"
            f"  n_samples={self.n_samples:,} ({sample_details}),\n"
            f"  n_discard_per_chain={self.n_discard_per_chain},\n"
            f"  machine_pow={self.machine_pow},\n"
            f"{policy}"
            f"  chunk_size={self.chunk_size},\n"
            f"  cached_samples={cached_samples}\n"
            f")"
        )

    @property
    def hilbert(self) -> DiscreteHilbertSpace:
        """Hilbert space defining the basis and flat configuration layout."""
        return self._hilbert

    @property
    def sampler(self) -> MetropolisSampler:
        """Sampler used for initialization, transitions, and chain settings."""
        return self._sampler

    @property
    def sampler_state(self) -> MetropolisSamplerState:
        """Current chain configurations, log probabilities, RNG, and counters.

        Access refreshes stale log probabilities after model-variable updates.
        It does not collect a sample batch. Chain configurations are retained
        unless the sampler has ``reset_chains=True``.
        """
        if self._sampler_state is None or self._sampler_state_stale:
            self.reset()
        assert self._sampler_state is not None
        return self._sampler_state

    @property
    def framework(self) -> AbstractModelFramework:
        """Adapter that initializes, evaluates, and splits model variables."""
        return self._framework

    @property
    def apply_variables_kernel(self) -> FrameworkApply:
        """Model evaluator with the state's training keyword arguments bound.

        The callable accepts ``(variables, states)`` and returns log amplitudes.
        Its identity remains stable across parameter updates for reuse by
        compiled estimators and sampler kernels.
        """
        return self._apply_variables_kernel

    @property
    def merge_variables_kernel(self) -> FrameworkMerge:
        """Stable callable merging ``(parameters, model_state)`` into variables.

        Estimators use this adapter when differentiating with respect to the
        trainable parameters while keeping non-trainable collections fixed.
        """
        return self._merge_variables_kernel

    @property
    def machine_pow(self) -> Any:
        r"""Sampler exponent :math:`p` in :math:`p_\theta\propto|\psi_\theta|^p`.

        This value comes from the sampler and defaults to two for Born sampling.
        """
        return self.sampler.machine_pow

    @property
    def chunk_size(self) -> int | None:
        """Default model-evaluation batch limit, or ``None`` for no chunking.

        Evaluation and estimator methods can override this limit per call.
        It does not set the number of samples or the sampler's sweep size.
        """
        return self._chunk_size

    @property
    def n_samples(self) -> int:
        """Requested total sample count across chains, before divisibility adjustment.

        The actual batch size is ``samples.shape[0] * samples.shape[1]`` for
        sampler-generated batches. Assigning a positive count clears the
        cached samples without restarting the chains.
        """
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value: int) -> None:
        """Set a positive sample count and invalidate the cached sample batch."""
        value = int(value)
        if value <= 0:
            raise ValueError("n_samples must be positive.")
        self._n_samples = value
        self._samples = None
        self._sample_measure = None

    @property
    def n_discard_per_chain(self) -> int:
        """Stored chain points discarded before every newly generated batch.

        A point follows ``sampler.sweep_size`` proposals. Changing this setting
        affects the next :meth:`sample` call and leaves cached samples intact.
        """
        return self._n_discard_per_chain

    @n_discard_per_chain.setter
    def n_discard_per_chain(self, value: int) -> None:
        """Set a nonnegative discard count for future sampling calls."""
        value = int(value)
        if value < 0:
            raise ValueError("n_discard_per_chain must be non-negative.")
        self._n_discard_per_chain = value

    @property
    def parameters(self) -> Any:
        """Trainable model parameters as a JAX pytree.

        Assign a replacement with the original tree structure to update the
        model, invalidate cached samples, and mark chain probabilities stale.
        Non-trainable collections are preserved.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value: Any) -> None:
        """Replace parameters, requiring the original pytree structure."""
        structure = jax.tree_util.tree_structure(value)
        if (
            self._parameters_structure is not None
            and structure != self._parameters_structure
        ):
            raise ValueError(
                "New parameters do not match the original pytree structure."
            )
        variables = self.framework.place_variables(
            self.framework.merge_variables(value, self.model_state)
        )
        value, model_state = self.framework.split_variables(variables)
        self._variables = variables
        self._model_state = model_state
        self._parameters = value
        self._parameters_structure = structure
        self._invalidate_sampler_state()

    @property
    def model_state(self) -> Any:
        """Non-trainable model collections, such as normalization statistics.

        Assigning a replacement preserves the trainable parameters and
        invalidates samples and chain probabilities. Model evaluations do not
        mutate these collections automatically.
        """
        return self._model_state

    @model_state.setter
    def model_state(self, value: Any) -> None:
        """Replace non-trainable collections and invalidate sampling caches."""
        variables = self.framework.place_variables(
            self.framework.merge_variables(self.parameters, value)
        )
        self._parameters, self._model_state = self.framework.split_variables(variables)
        self._variables = variables
        self._invalidate_sampler_state()

    @property
    def variables(self) -> Any:
        """Complete framework variables containing parameters and model state.

        Assignment replaces both parts together and invalidates sampling
        caches. The parameter subtree must retain its original structure.
        """
        return self._variables

    @variables.setter
    def variables(self, value: Any) -> None:
        """Replace all variables while preserving the parameter tree structure."""
        value = self.framework.place_variables(value)
        parameters, model_state = self.framework.split_variables(value)
        structure = jax.tree_util.tree_structure(parameters)
        if (
            self._parameters_structure is not None
            and structure != self._parameters_structure
        ):
            raise ValueError(
                "New variables contain a different parameter pytree structure."
            )
        self._variables = value
        self._parameters = parameters
        self._model_state = model_state
        self._parameters_structure = structure
        self._invalidate_sampler_state()

    @property
    def samples(self) -> jax.Array:
        """Cached configurations, drawing a batch on first access if needed.

        Generated batches have shape ``(n_chains, chain_length, hilbert.size)``.
        Reading this property again reuses the same configurations. Call
        :meth:`sample` to advance the chains and replace the batch.

        Assigning an array supplies configurations directly to estimators.
        Only the trailing Hilbert dimension is checked. The assignment does
        not validate gauge constraints, alter chain state, or change
        :attr:`n_samples`.
        """
        if self._samples is None:
            self.sample()
        assert self._samples is not None
        return self._samples

    @samples.setter
    def samples(self, value: Any) -> None:
        """Cache supplied configurations without advancing or resetting chains."""
        if self.importance_sampling_enabled:
            raise ValueError(
                "Use set_samples(..., log_density=...) for importance samples."
            )
        arr = jnp.asarray(value)
        if arr.ndim == 0 or arr.shape[-1] != self.hilbert.size:
            raise ValueError("Samples have the wrong trailing Hilbert dimension.")
        self._samples = arr
        self._sample_measure = None

    @property
    def n_parameters(self) -> int:
        """Number of scalar entries in the trainable parameter pytree.

        A complex entry counts as one parameter here. A non-holomorphic QGT
        may represent that entry using two real coordinates.
        """
        return tree_size(self.parameters)

    @property
    def training_kwargs(self) -> dict[str, Any]:
        """Copy of the keyword arguments bound to every model evaluation.

        Editing the returned dictionary does not change the state's settings.
        """
        return dict(self._training_kwargs)

    def apply_variables(self, variables: Any, states: Any) -> Any:
        """Evaluate log amplitudes with supplied variables and training settings.

        Args:
            variables: Complete variables in the framework's representation.
            states: Model input batch, normally ``(batch_size, hilbert.size)``.

        Returns:
            The model's log-amplitude output. This method calls the model
            directly without reshaping or chunking the input. It does not
            replace the state's variables or invalidate samples.
        """
        return self.apply_variables_kernel(variables, states)

    def apply(self, states: Any, *, mutable: Any | None = None) -> Any:
        """Call the model with the state's current variables.

        Args:
            states: Model input batch, normally ``(batch_size, hilbert.size)``.
            mutable: Only ``False`` and ``None`` are supported.

        Returns:
            Log amplitudes in the model's output layout. Use :meth:`log_value`
            to handle a single configuration, arbitrary leading batch axes,
            or chunked evaluation.

        Raises:
            NotImplementedError: Mutable model collections were requested.
        """
        if mutable is not False and mutable is not None:
            raise NotImplementedError(
                "Mutable model collections are not supported yet."
            )
        return self.apply_variables(self.variables, states)

    def log_value(
        self,
        states: Any,
        *,
        chunk_size: int | None = None,
        mutable: Any | None = None,
    ) -> jax.Array:
        r"""Evaluate :math:`\log\psi_\theta` on one or more basis configurations.

        Args:
            states: Configurations with shape ``(..., hilbert.size)``. A single
                configuration may have shape ``(hilbert.size,)``.
            chunk_size: Model batch limit for this call. ``None`` uses the
                state's configured limit.
            mutable: Only ``False`` and ``None`` are supported.

        Returns:
            Log amplitudes with the input's leading batch shape. For a scalar
            log-amplitude model, one configuration produces a scalar.

        Raises:
            ValueError: The trailing dimension does not match the Hilbert size.
            NotImplementedError: Mutable model collections were requested.
        """
        if mutable is not False and mutable is not None:
            raise NotImplementedError(
                "Mutable model collections are not supported yet."
            )
        flat, state_shape = flatten_state_batch(states, self.hilbert.size)
        values = evaluate_log_values(
            self.apply_variables_kernel,
            self.variables,
            flat,
            chunk_size=self.chunk_size if chunk_size is None else chunk_size,
        )
        return restore_log_values(values, state_shape)

    def to_array(
        self,
        normalize: bool = True,
        *,
        max_states: int = 1_000_000,
        chunk_size: int | None = None,
    ) -> jax.Array:
        r"""Enumerate the basis and evaluate the full wavefunction.

        Args:
            normalize: Divide by the Euclidean norm so that
                :math:`\sum_\sigma|\psi(\sigma)|^2=1`. Normalization uses a
                common real shift of the log amplitudes to limit overflow.
                With ``False``, return ``exp(log_value)`` without rescaling.
            max_states: Maximum basis size allowed by Hilbert enumeration.
            chunk_size: Model batch limit for this call. ``None`` uses the
                state's configured limit.

        Returns:
            Amplitudes in ``hilbert.all_states()`` order. For a scalar model,
            the result is a vector of length ``hilbert.dimension``. This is
            an exact basis evaluation and does not use or replace MC samples.

        Raises:
            RuntimeError: The Hilbert space does not support basis indexing.
        """
        if not self.hilbert.is_indexable:
            raise RuntimeError("to_array requires an indexable Hilbert space.")
        states = self.hilbert.all_states(max_states=max_states)
        log_values = evaluate_log_values(
            self.apply_variables_kernel,
            self.variables,
            states,
            chunk_size=self.chunk_size if chunk_size is None else chunk_size,
        )
        return amplitudes_from_log_values(log_values, normalize=bool(normalize))

    def reset(self) -> None:
        """Clear samples and refresh the sampler against the current variables.

        The first call initializes chain configurations. Later calls retain
        those configurations unless ``sampler.reset_chains`` is true. Log
        probabilities and acceptance counters are refreshed in either case.
        This method does not collect samples or rewind the random stream.
        """
        self._samples = None
        self._sample_measure = None
        if hasattr(self, "_variables"):
            sampling_apply, sampling_variables = self._sampling_inputs()
            self._sampler_state = self.sampler.reset(
                sampling_apply,
                sampling_variables,
                state=self._sampler_state,
                key=self._sampler_key,
            )
            self._sampler_state_stale = False

    def _invalidate_sampler_state(self) -> None:
        """Clear samples and defer recomputing chain probabilities until needed."""
        self._samples = None
        self._sample_measure = None
        self._importance_weight_cache = None
        self._sampler_state_stale = True

    def sample(self) -> tuple[jax.Array, MetropolisSamplerState]:
        """Advance the chains and replace the cached sample batch.

        Refresh stale chain probabilities, discard :attr:`n_discard_per_chain`
        points from each chain, then collect :attr:`n_samples` configurations
        with the sampler's chain-count adjustment. Discards occur on every
        call, even when an earlier batch is still cached.

        Returns:
            ``(samples, sampler_state)``. Samples have shape
            ``(n_chains, chain_length, hilbert.size)`` and are also available
            through :attr:`samples`. The returned sampler state contains the
            final chain positions and updated RNG and acceptance counters.
        """
        state = self.sampler_state
        sampling_apply, sampling_variables = self._sampling_inputs()
        if self.n_discard_per_chain:
            _discarded, state = self.sampler.sample(
                sampling_apply,
                sampling_variables,
                state=state,
                chain_length=self.n_discard_per_chain,
            )
        samples, state = self.sampler.sample(
            sampling_apply,
            sampling_variables,
            state=state,
            n_samples=self.n_samples,
        )
        if self.importance_sampling_enabled:
            samples = self._record_sampling_measure(samples)
        self._samples = samples
        self._sampler_state = state
        return samples, state

    @property
    def importance_sampling_enabled(self) -> bool:
        """Whether samples carry a density correction for Born expectations."""
        return (
            getattr(self.sampler, "importance_sampling", None) is not None
            or getattr(self.sampler, "operator_blur", None) is not None
        )

    @property
    def sampling_exponent(self):
        """Current exponent for the next base-chain batch."""
        return self._importance_exponent

    @property
    def sample_measure(self):
        """Frozen proposal density belonging to the cached sample batch."""
        _ = self.samples
        return self._sample_measure

    @property
    def sample_weights(self):
        """Normalized Born weights for the current batch, or None for ordinary sampling."""
        if not self.importance_sampling_enabled:
            return None
        from neuralqx.sampler.importance.measure import normalized_weights

        samples = self.samples
        cached = self._importance_weight_cache
        if cached is not None and cached[0] is samples and cached[1] is self.variables:
            return cached[2]
        weights = normalized_weights(
            self.log_value(samples), self._sample_measure.log_density
        )
        self._cache_importance_weights(samples, weights)
        return weights

    @property
    def sampling_ess(self):
        """Weight effective sample size of the cached batch, before chain correlations.

        Reading this property draws samples if needed. Values close to the
        batch size mean that density weights are fairly even. This diagnostic
        does not assess chain mixing or prove that rare residuals were visited.
        """
        weights = self.sample_weights
        return (
            self.samples.size // self.hilbert.size
            if weights is None
            else 1 / jnp.sum(weights**2)
        )

    def _cache_importance_weights(self, samples, weights):
        if isinstance(weights, jax.core.Tracer):
            return
        self._importance_weight_cache = (
            samples,
            self.variables,
            jax.lax.stop_gradient(weights),
        )

    def sampling_measure_for(self, samples):
        """Return the matching density or reject an unlabelled external sample batch."""
        if not self.importance_sampling_enabled:
            return None
        if samples is not self._samples or self._sample_measure is None:
            raise ValueError(
                "Importance samples need their sampling density. Use set_samples(..., log_density=...)."
            )
        return self._sample_measure

    def set_samples(self, samples, *, log_density):
        """Supply configurations and their unnormalized sampling log density.

        The density must describe the process that produced these samples.
        An arbitrary common additive constant is allowed. This method does not
        advance the chains, and externally supplied batches do not adapt the
        sampling exponent.
        """
        from neuralqx.sampler.importance import SamplingMeasure

        if not self.importance_sampling_enabled:
            raise ValueError(
                "set_samples with a density requires importance_sampling or operator_blur."
            )
        values = jnp.asarray(samples)
        density = jnp.asarray(log_density)
        if jnp.iscomplexobj(density):
            raise ValueError("A sampling log density must be real.")
        if (
            values.ndim < 2
            or values.shape[-1] != self.hilbert.size
            or density.shape != values.shape[:-1]
        ):
            raise ValueError("Samples and log_density have incompatible shapes.")
        if not bool(jnp.all(jnp.isfinite(density))):
            raise ValueError(
                "The sampling density must be finite on every supplied sample."
            )
        self._samples = values
        self._importance_weight_cache = None
        self._sample_measure = SamplingMeasure(
            density, jnp.zeros_like(density), self._importance_exponent, -1
        )

    def _initialize_importance(self):
        from neuralqx.sampler.importance.measure import TemperedApply

        config = getattr(self.sampler, "importance_sampling", None)
        self._importance_exponent = (
            self.machine_pow if config is None else jnp.asarray(config.exponent)
        )
        self._sample_measure = None
        self._importance_weight_cache = None
        self._importance_batch = 0
        self._importance_last_adapted = -1
        self._importance_key = (
            jax.random.fold_in(self._sampler_key, 194732)
            if self.importance_sampling_enabled
            else self._sampler_key
        )
        if self.importance_sampling_enabled and jax.process_count() > 1:
            from jax.experimental import multihost_utils

            raw = multihost_utils.broadcast_one_to_all(
                jax.random.key_data(self._importance_key)
            )
            self._importance_key = jax.random.wrap_key_data(
                raw, impl=str(jax.random.key_impl(self._importance_key))
            )
        self._sampling_apply = TemperedApply(self.apply_variables_kernel)
        self._blur_operators = None
        if getattr(self.sampler, "operator_blur", None) is not None:
            from neuralqx.sampler.importance.blur import blur_operators

            self._blur_operators = blur_operators(
                getattr(self.sampler, "operator_blur", None), self.hilbert
            )

    def _sampling_inputs(self):
        if getattr(self.sampler, "importance_sampling", None) is None:
            return self.apply_variables_kernel, self.variables
        return self._sampling_apply, (self.variables, self._importance_exponent)

    def _record_sampling_measure(self, samples):
        from neuralqx.sampler.importance import SamplingMeasure

        if getattr(self.sampler, "operator_blur", None) is not None:
            from neuralqx.sampler.importance.blur import blur_samples

            self._importance_key, key = jax.random.split(self._importance_key)
            samples, density, score = blur_samples(
                getattr(self.sampler, "operator_blur", None),
                self._blur_operators,
                self.apply_variables_kernel,
                self.variables,
                samples,
                self._importance_exponent,
                key,
                self.chunk_size,
            )
        else:
            values = self.log_value(samples).real
            density = self._importance_exponent * values
            score = values - jnp.mean(values)
        self._importance_batch += 1
        self._sample_measure = SamplingMeasure(
            jax.lax.stop_gradient(density),
            jax.lax.stop_gradient(score),
            self._importance_exponent,
            self._importance_batch,
        )
        return samples


__all__ = ["VariationalState"]
