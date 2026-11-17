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


"""Versioned checkpoint envelopes and reproducibility metadata."""

import hashlib
import importlib.metadata
import json
import platform
import re
import zipfile
from datetime import UTC
from datetime import datetime
from pathlib import Path

import jax

from .collective import canonical_json
from .errors import CheckpointError

FORMAT = "neuralqx.nqx"
"""Magic identifier for the neuraLQX checkpoint envelope."""

FORMAT_VERSION = 2
"""Archive format version. Version one directory checkpoints remain readable."""


def runtime_metadata():
    """Describe software and execution topology without copying device arrays."""
    from neuralqx._version import __version__

    versions = {"neuralqx": __version__, "python": platform.python_version()}
    for name in (
        "jax",
        "jaxlib",
        "flax",
        "orbax-checkpoint",
        "numpy",
        "networkx",
        "scipy",
        "optax",
    ):
        versions[name] = importlib.metadata.version(name)
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "versions": versions,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "jax_settings": {
            name: jax.config.values.get(name)
            for name in (
                "jax_default_prng_impl",
                "jax_threefry_partitionable",
                "jax_default_matmul_precision",
            )
        },
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "devices": [
            {"platform": d.platform, "kind": d.device_kind, "process": d.process_index}
            for d in jax.devices()
        ],
    }


def checkpoint_path(path):
    """Resolve a local .nqx checkpoint path, adding its suffix when omitted."""
    path = Path(path).expanduser().absolute()
    if path.suffix != ".nqx":
        path = path.with_name(path.name + ".nqx")
    return path


def envelope(snapshot, metadata):
    """Seal one committed snapshot's metadata with a deterministic checksum."""
    content = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "snapshot": snapshot,
        "metadata": metadata,
    }
    return {**content, "sha256": hashlib.sha256(canonical_json(content)).hexdigest()}


def read_manifest(path):
    """Read and validate the committed manifest without loading tensor data."""
    path = checkpoint_path(path)
    try:
        if path.is_dir():
            with (path / "manifest.json").open(encoding="utf-8") as stream:
                value = json.load(stream)
        else:
            with zipfile.ZipFile(path, "r") as archive:
                if archive.namelist().count("manifest.json") != 1:
                    raise CheckpointError(
                        "The .nqx archive must contain exactly one manifest."
                    )
                with archive.open("manifest.json") as stream:
                    value = json.load(stream)
    except FileNotFoundError as error:
        raise CheckpointError(f"No committed .nqx checkpoint at {path}.") from error
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Invalid .nqx checkpoint at {path}.") from error
    return validate_manifest(value)


def validate_manifest(value):
    """Check a decoded manifest independently of its physical storage container."""
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise CheckpointError("This is not a neuraLQX .nqx checkpoint.")
    if type(value.get("format_version")) is not int or value["format_version"] not in (
        1,
        FORMAT_VERSION,
    ):
        raise CheckpointError(
            f"Unsupported .nqx serialization version {value.get('format_version')!r}."
        )
    checksum = value.get("sha256")
    content = {key: item for key, item in value.items() if key != "sha256"}
    if checksum != hashlib.sha256(canonical_json(content)).hexdigest():
        raise CheckpointError("The .nqx manifest checksum does not match its contents.")
    if not re.fullmatch(r"[0-9a-f]{32}", str(value.get("snapshot", ""))):
        raise CheckpointError("The .nqx snapshot identifier is malformed.")
    if not isinstance(value.get("metadata"), dict):
        raise CheckpointError("The .nqx metadata must be a mapping.")
    return value


def inspect_checkpoint(path):
    """Return a checkpoint's versions, recipes, shapes and sampling settings.

    Inspection is read-only, loads no tensors, and is not collective. No saved
    Python classes or functions are imported or executed.
    """
    return read_manifest(path)
