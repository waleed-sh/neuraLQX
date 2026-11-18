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


"""Capture and restore sampled-state semantics around the general .nqx storage API."""

import warnings

import numpy as np

import jax
from neuralqx.jax.random import as_prng_key
from neuralqx.jax.sharding import sample_batch_sharding
from neuralqx.jax.sharding import sample_partition_count
from neuralqx.jax.sharding.arrays import place_global_array
from neuralqx.utils.frameworks import FrameworkApply
from neuralqx.utils.frameworks import static_training_kwargs

from .collective import agree
from .collective import collective_call
from .errors import CheckpointError
from .errors import ReconstructionError
from .metadata import read_manifest
from .registry import component_description
from .registry import decode_config
from .registry import encode_config
from .storage import load_pytree
from .storage import load_subtree
from .storage import save_pytree
from .trees import replicated_sharding


def _model(state):
    framework = state.framework
    if framework.name == "flax.nnx":
        from flax import nnx

        return nnx.merge(framework.graphdef, state.parameters, state.model_state)
    return getattr(framework, "model", getattr(framework, "apply_fn", framework))


def _sampler_payload(state):
    if state is None:
        return None
    return {
        name: getattr(state, name)
        for name in (
            "sigma",
            "log_prob",
            "rng",
            "rule_state",
            "n_steps",
            "n_accepted_per_chain",
        )
    }


def _payload(state, include_samples):
    payload = {
        "variables": state.variables,
        "sampler_key": state._sampler_key,
        "estimator_key": state._estimator_key,
        "sampler_state": _sampler_payload(state._sampler_state),
        "samples": state._samples if include_samples else None,
    }
    if state.importance_sampling_enabled:
        measure = state._sample_measure if include_samples else None
        payload["importance"] = {
            "exponent": state._importance_exponent,
            "key": state._importance_key,
            "measure": (
                None
                if measure is None
                else {
                    "log_density": measure.log_density,
                    "exponent_score": measure.exponent_score,
                    "exponent": measure.exponent,
                }
            ),
        }
    return payload


def _sampler_keys(key):
    if jax.process_count() == 1:
        return key
    from jax.experimental import multihost_utils

    def local_data():
        raw = jax.random.key_data(key)
        if not raw.is_fully_addressable:
            if not raw.is_fully_replicated:
                raise CheckpointError(
                    "The sampler root key must be local or replicated."
                )
            raw = raw.addressable_shards[0].data
        return np.asarray(raw)

    raw = collective_call(local_data, "process sampler keys")
    implementation = str(jax.random.key_impl(key))
    agree(
        {"shape": list(raw.shape), "impl": implementation}, "sampler key representation"
    )
    gathered = np.asarray(multihost_utils.process_allgather(raw, tiled=False))
    array = jax.make_array_from_callback(
        gathered.shape, replicated_sharding(), lambda index: gathered[index]
    )
    return jax.random.wrap_key_data(array, impl=implementation)


def _local_sampler_key(keys, description):
    if not description["process_sampler_keys"]:
        return keys
    local = keys.addressable_shards[0].data
    saved_processes = local.shape[0]
    if saved_processes == jax.process_count():
        return local[jax.process_index()]
    raw = np.asarray(jax.random.key_data(local))
    if not np.all(raw == raw[0]):
        raise CheckpointError(
            "Process-specific sampler root keys cannot be mapped to a different process count. Use sampler_mode='restart' with sampler_seed."
        )
    return local[0]


def _settings(state):
    return {
        "n_samples": state.n_samples,
        "n_discard_per_chain": state.n_discard_per_chain,
        "chunk_size": None if state.chunk_size is None else int(state.chunk_size),
        "training_kwargs": state.training_kwargs,
    }


def _importance_policy(state):
    """Keep sampling controls checkable even when a custom operator needs a template."""
    blur = getattr(state.sampler, "operator_blur", None)
    return {
        "importance_sampling": encode_config(
            getattr(state.sampler, "importance_sampling", None)
        ),
        "operator_blur": (
            None
            if blur is None
            else {
                "probability": blur.probability,
                "chunk_size": blur.chunk_size,
                "operator": component_description(blur.operator),
            }
        ),
    }


def save_variational_state(
    state, path="vstate.nqx", *, include_samples=False, metadata=None, overwrite=False
):
    """Save variables, current chain endpoints, RNGs and sampler bookkeeping.

    Reading internal state here deliberately avoids ``sampler_state`` and
    ``samples`` properties, which could otherwise evaluate the model or advance
    random streams. Cached samples are optional and never generated by saving.
    Optimizer and driver state are outside a variational state's ownership.
    """

    def capture():
        description = {
            "kind": "variational_state",
            "state_class": f"{type(state).__module__}.{type(state).__qualname__}",
            "framework": state.framework.name,
            "model": component_description(_model(state)),
            "model_name": state._model_name,
            "sampler": component_description(state.sampler),
            "hilbert": component_description(state.hilbert),
            "hilbert_size": state.hilbert.size,
            "settings": encode_config(_settings(state)),
            "sample_partitions": sample_partition_count(),
            "process_sampler_keys": jax.process_count() > 1,
            "sampler_initialized": state._sampler_state is not None,
            "sampler_state_stale": bool(state._sampler_state_stale),
            "cached_samples": bool(include_samples and state._samples is not None),
            "importance": (
                None
                if not state.importance_sampling_enabled
                else {
                    "policy": _importance_policy(state),
                    "batch": state._importance_batch,
                    "last_adapted": state._importance_last_adapted,
                    "measure_batch": (
                        state._sample_measure.batch_id
                        if include_samples and state._sample_measure is not None
                        else None
                    ),
                }
            ),
            "annotations": dict(metadata or {}),
        }
        return _payload(state, include_samples), description

    payload, description = collective_call(capture, "variational state capture")
    payload["sampler_key"] = _sampler_keys(payload["sampler_key"])
    return save_pytree(path, payload, metadata=description, overwrite=overwrite)


def _state_description(manifest):
    description = manifest["metadata"]["user"]
    if description.get("kind") != "variational_state":
        raise CheckpointError(
            "This .nqx checkpoint does not contain a variational state."
        )
    return description


def _check_component(saved, current, label):
    actual = component_description(current)
    if saved["recipe"] is not None:
        if saved["recipe"] != actual["recipe"]:
            raise CheckpointError(
                f"The target {label} configuration differs from the checkpoint."
            )
    elif saved["class"] != actual["class"]:
        raise CheckpointError(f"The target {label} type differs from the checkpoint.")
    elif saved.get("definition") != actual.get("definition"):
        raise CheckpointError(
            f"The target {label} callable differs from the checkpoint."
        )


def _array_template(item, sharding):
    if item.get("key_impl"):
        key = jax.random.key(0, impl=item["key_impl"])
        key = jax.numpy.broadcast_to(key, tuple(item["shape"]))
        return place_global_array(key, sharding)
    dtype = np.dtype(item["dtype"])
    if jax.dtypes.canonicalize_dtype(dtype) != dtype:
        raise CheckpointError(
            f"Checkpoint uses {dtype}. Enable jax_enable_x64 before loading."
        )
    return jax.ShapeDtypeStruct(
        tuple(item["shape"]),
        dtype,
        sharding=sharding,
        weak_type=item.get("weak_type", False),
    )


def _restore_template(state, manifest, sampler_mode):
    description = _state_description(manifest)
    leaves = {item["path"]: item for item in manifest["metadata"]["tree"]["leaves"]}
    replica = replicated_sharding()
    payload = {
        "variables": state.variables,
        "sampler_key": _array_template(leaves["['sampler_key']"], replica),
        "sampler_state": None,
        "samples": None,
    }
    if "['estimator_key']" in leaves:
        payload["estimator_key"] = _array_template(leaves["['estimator_key']"], replica)
    if description["sampler_initialized"]:
        expected_shapes = {
            "sigma": [state.sampler.n_chains, state.hilbert.size],
            "log_prob": [state.sampler.n_chains],
            "n_accepted_per_chain": [state.sampler.n_chains],
            "n_steps": [],
            "rng": (
                []
                if description["sample_partitions"] == 1
                else [description["sample_partitions"]]
            ),
        }
        for name, shape in expected_shapes.items():
            if leaves[f"['sampler_state']['{name}']"]["shape"] != shape:
                raise CheckpointError(
                    f"Saved sampler {name} shape disagrees with its chain configuration."
                )
        if state._sampler_state is not None:
            rule_state = state._sampler_state.rule_state
        else:
            rule_state = state.sampler.rule.init_state(
                state.sampler,
                state.apply_variables_kernel,
                state.variables,
                jax.random.key(0),
            )
        sampler = {"rule_state": rule_state}
        for name in ("sigma", "log_prob", "rng", "n_steps", "n_accepted_per_chain"):
            item = leaves[f"['sampler_state']['{name}']"]
            placement = replica
            if sampler_mode == "continue" and name != "n_steps" and item["shape"]:
                placement = sample_batch_sharding(len(item["shape"])) or replica
            sampler[name] = _array_template(item, placement)
        payload["sampler_state"] = sampler
    if description["cached_samples"]:
        item = leaves["['samples']"]
        if not item["shape"] or item["shape"][-1] != state.hilbert.size:
            raise CheckpointError("Saved samples have the wrong Hilbert dimension.")
        placement = replica
        if item["shape"] and item["shape"][0] % sample_partition_count() == 0:
            placement = sample_batch_sharding(len(item["shape"])) or replica
        payload["samples"] = _array_template(item, placement)
    if description.get("importance") is not None:
        importance = {
            name: _array_template(leaves[f"['importance']['{name}']"], replica)
            for name in ("exponent", "key")
        }
        importance["measure"] = None
        if description["importance"]["measure_batch"] is not None:
            measure = {}
            for name in ("log_density", "exponent_score", "exponent"):
                item = leaves[f"['importance']['measure']['{name}']"]
                placement = replica
                if name != "exponent":
                    layout = payload["samples"].sharding
                    placement = jax.sharding.NamedSharding(
                        layout.mesh,
                        jax.sharding.PartitionSpec(*layout.spec[: len(item["shape"])]),
                    )
                measure[name] = _array_template(item, placement)
            importance["measure"] = measure
        payload["importance"] = importance
    return payload


def load_variational_state(
    state,
    path="vstate.nqx",
    *,
    sampler_mode="continue",
    sampler_seed=None,
    _model_from_recipe=False,
):
    """Restore into a compatible live state, committing only after validation.

    ``continue`` preserves sampler RNG implementations, endpoints and counters.
    It requires the same chain layout and number of sample partitions. Device
    types and process placement may change, but different arithmetic can still
    change future Metropolis decisions. ``restart`` restores the wavefunction
    and starts new chains using an explicitly supplied ``sampler_seed``.
    """
    manifest = collective_call(lambda: read_manifest(path), "variational manifest")
    agree(manifest, "variational checkpoint")
    description = _state_description(manifest)

    def validate():
        if sampler_mode not in ("continue", "restart"):
            raise ValueError("sampler_mode must be 'continue' or 'restart'.")
        if sampler_mode == "restart" and sampler_seed is None:
            raise ValueError(
                "sampler_mode='restart' requires an explicit sampler_seed."
            )
        if sampler_mode == "continue" and sampler_seed is not None:
            raise ValueError("sampler_seed is only used with sampler_mode='restart'.")
        if state.framework.name != description["framework"]:
            raise CheckpointError("The saved and target model frameworks differ.")
        if not _model_from_recipe:
            _check_component(description["model"], _model(state), "model")
        _check_component(description["hilbert"], state.hilbert, "Hilbert space")
        if state.hilbert.size != description["hilbert_size"]:
            raise CheckpointError("The saved and target Hilbert dimensions differ.")
        if sampler_mode == "continue":
            if (
                description.get("importance") is not None
            ) != state.importance_sampling_enabled:
                raise CheckpointError(
                    "Saved and target importance-sampling policies differ."
                )
            importance = description.get("importance")
            if importance is not None and importance.get("policy") is not None:
                if importance["policy"] != _importance_policy(state):
                    raise CheckpointError(
                        "The target importance-sampling configuration differs from the checkpoint."
                    )
            if sample_partition_count() != description["sample_partitions"]:
                raise CheckpointError(
                    "The number of sample partitions changed. Use sampler_mode='restart' with sampler_seed to start new random streams."
                )
            _check_component(description["sampler"], state.sampler, "sampler")
        settings = decode_config(description["settings"])
        if (
            not isinstance(settings, dict)
            or set(settings)
            != {"n_samples", "n_discard_per_chain", "chunk_size", "training_kwargs"}
            or type(settings["n_samples"]) is not int
            or settings["n_samples"] <= 0
            or type(settings["n_discard_per_chain"]) is not int
            or settings["n_discard_per_chain"] < 0
            or not isinstance(settings["training_kwargs"], dict)
        ):
            raise CheckpointError(
                "The saved variational sampling settings are invalid."
            )
        chunk = settings["chunk_size"]
        if chunk is not None and (type(chunk) is not int or chunk <= 0):
            raise CheckpointError("The saved chunk_size must be positive or None.")
        return settings

    settings = collective_call(validate, "variational compatibility")
    agree(
        {"mode": sampler_mode, "settings": description["settings"]},
        "restoration settings",
    )
    if sampler_mode == "continue":
        template = collective_call(
            lambda: _restore_template(state, manifest, sampler_mode),
            "variational template",
        )
        payload = load_pytree(path, template, manifest=manifest)
    else:
        payload = {
            "variables": load_subtree(
                path, "variables", state.variables, manifest=manifest
            ),
            "sampler_state": None,
            "sampler_key": as_prng_key(sampler_seed),
            "samples": None,
        }

    def prepare():
        from neuralqx.sampler import MetropolisSamplerState
        from neuralqx.sampler.importance import SamplingMeasure
        from neuralqx.sampler.importance.measure import TemperedApply

        variables = payload["variables"]
        parameters, model_state = state.framework.split_variables(variables)
        apply_kernel = FrameworkApply(
            state.framework, static_training_kwargs(settings["training_kwargs"])
        )
        sampler_key = (
            _local_sampler_key(payload["sampler_key"], description)
            if sampler_mode == "continue"
            else payload["sampler_key"]
        )
        sampler_state = (
            None
            if payload["sampler_state"] is None
            else MetropolisSamplerState(**payload["sampler_state"])
        )
        stale = description["sampler_state_stale"]
        samples = payload["samples"]
        runtime = payload.get("importance")
        importance_exponent = (
            state._importance_exponent if runtime is None else runtime["exponent"]
        )
        importance_key = state._importance_key if runtime is None else runtime["key"]
        estimator_key = payload.get("estimator_key", state._estimator_key)
        measure = None
        if runtime is not None and runtime["measure"] is not None:
            measure = SamplingMeasure(
                **runtime["measure"],
                batch_id=description["importance"]["measure_batch"],
            )
        sampling_apply = TemperedApply(apply_kernel)
        if sampler_mode == "restart":
            sampler_key = as_prng_key(sampler_seed)
            estimator_key = jax.random.fold_in(sampler_key, 741923)
            if jax.process_count() > 1:
                from jax.experimental import multihost_utils

                raw = multihost_utils.broadcast_one_to_all(
                    jax.random.key_data(estimator_key)
                )
                estimator_key = jax.random.wrap_key_data(
                    raw, impl=str(jax.random.key_impl(estimator_key))
                )
            importance_key = jax.random.fold_in(sampler_key, 194732)
            if state.importance_sampling_enabled and jax.process_count() > 1:
                from jax.experimental import multihost_utils

                raw = multihost_utils.broadcast_one_to_all(
                    jax.random.key_data(importance_key)
                )
                importance_key = jax.random.wrap_key_data(
                    raw, impl=str(jax.random.key_impl(importance_key))
                )
            config = getattr(state.sampler, "importance_sampling", None)
            importance_exponent = jax.numpy.asarray(
                2.0 if config is None else config.exponent
            )
            function = apply_kernel if config is None else sampling_apply
            inputs = variables if config is None else (variables, importance_exponent)
            sampler_state = state.sampler.reset(function, inputs, key=sampler_key)
            stale = False
            samples = None
        jax.block_until_ready((variables, sampler_state, sampler_key, samples))
        return {
            "_variables": variables,
            "_parameters": parameters,
            "_model_state": model_state,
            "_parameters_structure": jax.tree.structure(parameters),
            "_sampler_key": sampler_key,
            "_estimator_key": estimator_key,
            "_sampler_state": sampler_state,
            "_sampler_state_stale": stale,
            "_samples": samples,
            "_sample_measure": measure,
            "_importance_weight_cache": None,
            "_sampling_apply": sampling_apply,
            "_importance_exponent": importance_exponent,
            "_importance_key": importance_key,
            "_importance_batch": (
                0 if runtime is None else description["importance"]["batch"]
            ),
            "_importance_last_adapted": (
                -1 if runtime is None else description["importance"]["last_adapted"]
            ),
            "_n_samples": settings["n_samples"],
            "_n_discard_per_chain": settings["n_discard_per_chain"],
            "_chunk_size": settings["chunk_size"],
            "_training_kwargs": settings["training_kwargs"],
            "_apply_variables_kernel": apply_kernel,
            "_model_name": description["model_name"],
        }

    updates = collective_call(prepare, "restored state preparation")
    state.__dict__.update(updates)
    return state


def variational_state_from_file(
    cls,
    path,
    *,
    model=None,
    sampler=None,
    sampler_mode="continue",
    sampler_seed=None,
    **model_options,
):
    """Reconstruct supported components and restore without first warming up chains."""
    manifest = collective_call(lambda: read_manifest(path), "variational manifest")
    agree(manifest, "variational checkpoint")
    description = _state_description(manifest)
    model_from_recipe = model is None

    def construct():
        nonlocal model, sampler
        for name, value in (("model", model), ("sampler", sampler)):
            if value is None and description[name]["recipe"] is None:
                raise ReconstructionError(
                    f"Pass {name}=... when loading this checkpoint. "
                    + description[name].get("requires_template", "")
                )
        if model is None:
            model = decode_config(description["model"]["recipe"])
        if sampler is None:
            sampler = decode_config(description["sampler"]["recipe"])
        settings = decode_config(description["settings"])
        return cls(
            sampler,
            model,
            **settings,
            **model_options,
            _defer_sampler_initialization=True,
        )

    # A different software version is useful provenance, not an automatic
    # rejection. Shape, dtype, recipe and serialization-version checks decide
    # whether this particular state can be restored.
    from neuralqx._version import __version__

    saved_version = manifest["metadata"]["runtime"]["versions"]["neuralqx"]
    if saved_version != __version__ and jax.process_index() == 0:
        warnings.warn(
            f"Loading neuraLQX {saved_version} state with {__version__}. Compatibility checks remain enabled.",
            RuntimeWarning,
            stacklevel=2,
        )
    state = collective_call(construct, "variational reconstruction")
    return load_variational_state(
        state,
        path,
        sampler_mode=sampler_mode,
        sampler_seed=sampler_seed,
        _model_from_recipe=model_from_recipe,
    )
