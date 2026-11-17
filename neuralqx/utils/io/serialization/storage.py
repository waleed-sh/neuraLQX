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


"""Atomic .nqx snapshot publication with distributed Orbax tensor storage."""

import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path

import jax

from .archive import snapshot_directory
from .archive import write_archive
from .collective import agree
from .collective import canonical_json
from .collective import collective_call
from .collective import primary_call
from .errors import CheckpointError
from .metadata import checkpoint_path
from .metadata import envelope
from .metadata import read_manifest
from .metadata import runtime_metadata
from .trees import describe_tree
from .trees import prepare_arrays
from .trees import rebuild_tree
from .trees import restore_targets


def _checkpointer():
    import orbax.checkpoint as ocp

    # Blocking completion gives save() a simple lifetime contract. Orbax still
    # performs shard transfers and storage writes concurrently within the save.
    return ocp.Checkpointer(
        ocp.StandardCheckpointHandler(save_concurrent_gb=1, restore_concurrent_gb=1)
    )


def save_pytree(path, tree, *, metadata=None, overwrite=False):
    """Commit a single .nqx file containing a pytree and JSON metadata.

    All JAX processes must call this function with the same shared-filesystem
    path. Each process writes its addressable shards through Orbax. The call
    returns only after the snapshot is durable and published. Completed shards
    are streamed into a ZIP64 container without gathering the tensor tree on
    one process. No pickle or executable Python payload is stored.

    Temporary files live beside the destination and are removed on completion.
    Allow free disk space for the staged shards and the new archive, in addition
    to any existing checkpoint. ``overwrite=True`` atomically replaces that
    checkpoint only after the new file is complete. A writer lock rejects
    concurrent writers. Legacy directory checkpoints must be saved to a new
    filename when converting them to this format.
    """
    import orbax.checkpoint as ocp

    path = collective_call(lambda: checkpoint_path(path), "checkpoint path")
    agree({"path": str(path), "overwrite": bool(overwrite)}, "save options")
    schema, arrays = prepare_arrays(tree)
    details = collective_call(lambda: dict(metadata or {}), "checkpoint metadata")
    collective_call(lambda: canonical_json(details), "JSON metadata validation")
    agree(details, "checkpoint metadata")
    runtime = primary_call(runtime_metadata, "runtime metadata")
    details = {"runtime": runtime, "tree": schema, "user": details}
    details["subtrees"] = collective_call(
        lambda: (
            {
                key: describe_tree(value)
                for key, value in tree.items()
                if isinstance(key, str)
            }
            if isinstance(tree, Mapping)
            else {}
        ),
        "subtree descriptions",
    )
    agree(details["subtrees"], "subtree descriptions")

    def begin():
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_name(f".{path.name}.writer.lock")
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise CheckpointError(
                f"A writer lock exists at {lock}. If its writer was terminated, remove that lock before retrying."
            ) from error
        directory = None
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(str(os.getpid()))
            if os.path.lexists(path) and not overwrite:
                raise FileExistsError(path)
            if path.is_dir():
                raise IsADirectoryError(
                    f"{path} is a legacy directory checkpoint. Load it and save to a new filename."
                )
            directory = Path(
                tempfile.mkdtemp(prefix=f".{path.name}-write-", dir=path.parent)
            )
            (directory / "snapshots").mkdir()
            return {
                "directory": str(directory),
                "snapshot": uuid.uuid4().hex,
                "lock": str(lock),
            }
        except Exception:
            if directory is not None:
                shutil.rmtree(directory)
            lock.unlink(missing_ok=True)
            raise

    transaction = primary_call(begin, "checkpoint transaction")
    staging = Path(transaction["directory"])
    snapshot = transaction["snapshot"]
    destination = staging / "snapshots" / snapshot
    try:

        def save():
            with _checkpointer() as checkpointer:
                checkpointer.save(
                    destination,
                    args=ocp.args.StandardSave(arrays),
                    custom_metadata=details,
                )

        collective_call(save, "checkpoint shard writes")
        manifest = envelope(snapshot, details)

        def publish():
            temporary = staging / "checkpoint.nqx"
            write_archive(staging, temporary, manifest)
            if overwrite:
                os.replace(temporary, path)
            else:
                # A hard link publishes atomically without replacing a file
                # created since the transaction began, even by another program.
                os.link(temporary, path)
                temporary.unlink()
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        primary_call(publish, "checkpoint publication")
    finally:

        def finish():
            try:
                shutil.rmtree(staging)
            finally:
                Path(transaction["lock"]).unlink(missing_ok=True)

        primary_call(finish, "checkpoint writer cleanup")
    return path


def load_pytree(path, target, *, manifest=None):
    """Restore a .nqx pytree into the structure and placements of ``target``.

    This is collective across initialized JAX processes. The target's array
    shapes and dtypes must match exactly, while its shardings may differ from
    those used when saving. Restoration reads the shards required by the new
    layout without collecting the complete tree on a coordinator.

    The archive is unpacked to a temporary directory beside the file and
    removed after all processes finish reading it. Its parent directory must
    be writable and shared by every process. Legacy directory checkpoints
    remain readable without staging.
    """
    import orbax.checkpoint as ocp

    path = collective_call(lambda: checkpoint_path(path), "checkpoint path")
    manifest = collective_call(
        lambda: read_manifest(path) if manifest is None else manifest,
        "checkpoint manifest",
    )
    agree({"path": str(path), "manifest": manifest}, "load options")
    schema = manifest["metadata"]["tree"]
    targets = collective_call(
        lambda: restore_targets(target, schema), "restoration template"
    )
    with snapshot_directory(path, manifest) as directory:

        def restore():
            with _checkpointer() as checkpointer:
                arrays = checkpointer.restore(
                    directory, args=ocp.args.StandardRestore(targets, strict=True)
                )
            return jax.block_until_ready(rebuild_tree(target, schema, arrays))

        result = collective_call(restore, "checkpoint restoration")
    return result


def load_subtree(path, key, target, *, manifest=None):
    """Restore one string-keyed root subtree without allocating the other arrays.

    This collective operation is useful for loading model variables while
    discarding an old sampler or cached samples. Structure, dtype and shape
    validation are identical to :func:`load_pytree`. Archive staging still
    unpacks the complete container onto the shared filesystem.
    """
    import numpy as np
    import orbax.checkpoint as ocp

    import jax

    path = collective_call(lambda: checkpoint_path(path), "checkpoint path")
    manifest = collective_call(
        lambda: read_manifest(path) if manifest is None else manifest,
        "checkpoint manifest",
    )
    agree({"path": str(path), "manifest": manifest, "key": key}, "subtree load options")

    def prepare():
        try:
            schema = manifest["metadata"]["subtrees"][key]
        except KeyError as error:
            raise CheckpointError(f"No saved root subtree {key!r}.") from error
        targets = restore_targets(target, schema)
        prefix = jax.tree_util.keystr((jax.tree_util.DictKey(key),))
        indices = [
            index
            for index, item in enumerate(manifest["metadata"]["tree"]["leaves"])
            if item["path"].startswith(prefix)
        ]
        if len(indices) != len(schema["leaves"]):
            raise CheckpointError(
                "The subtree array index is inconsistent with the manifest."
            )
        renamed = {
            f"leaf_{global_index:08d}": targets[f"leaf_{index:08d}"]
            for index, global_index in enumerate(indices)
        }
        args = {
            name: ocp.ArrayRestoreArgs(
                restore_type=jax.Array if value.sharding is not None else np.ndarray,
                dtype=value.dtype,
                sharding=value.sharding,
                global_shape=value.shape,
                strict=True,
            )
            for name, value in renamed.items()
        }
        return schema, indices, renamed, args

    schema, indices, targets, args = collective_call(
        prepare, "subtree restoration template"
    )
    if not indices:
        return rebuild_tree(target, schema, {})
    with snapshot_directory(path, manifest) as directory:

        def restore():
            with ocp.Checkpointer(
                ocp.PyTreeCheckpointHandler(restore_concurrent_gb=1)
            ) as checkpointer:
                arrays = checkpointer.restore(
                    directory,
                    args=ocp.args.PyTreeRestore(
                        item=targets, restore_args=args, partial_restore=True
                    ),
                )
            arrays = {
                f"leaf_{index:08d}": arrays[f"leaf_{global_index:08d}"]
                for index, global_index in enumerate(indices)
            }
            return jax.block_until_ready(rebuild_tree(target, schema, arrays))

        return collective_call(restore, "checkpoint subtree restoration")
