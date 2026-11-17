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


"""Streaming archive transport around distributed checkpoint directories."""

import json
import os
import shutil
import stat
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from pathlib import PurePosixPath

from .collective import canonical_json
from .collective import primary_call
from .errors import CheckpointError
from .metadata import read_manifest
from .metadata import validate_manifest

_COPY_BUFFER = 8 * 1024 * 1024


def write_archive(source, destination, manifest):
    """Write one ZIP64 archive without buffering whole tensors or recompressing them.

    Orbax has already encoded the tensor chunks. The outer container stores
    those bytes directly, including checksums for each member. Closing and
    syncing the archive happens before its caller publishes the final path.
    """
    root = Path(source)
    with Path(destination).open("xb") as stream:
        with zipfile.ZipFile(
            stream, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            archive.writestr("manifest.json", canonical_json(manifest))
            for item in sorted((root / "snapshots" / manifest["snapshot"]).rglob("*")):
                if item.is_symlink():
                    raise CheckpointError(
                        "A checkpoint snapshot cannot contain symbolic links."
                    )
                if item.is_file():
                    info = zipfile.ZipInfo.from_file(
                        item, item.relative_to(root).as_posix()
                    )
                    with item.open("rb") as source, archive.open(info, "w") as target:
                        shutil.copyfileobj(source, target, length=_COPY_BUFFER)
        stream.flush()
        os.fsync(stream.fileno())


def _members(archive, manifest):
    prefix = f"snapshots/{manifest['snapshot']}/"
    seen = set()
    for member in archive.infolist():
        name = member.filename
        relative = PurePosixPath(name)
        mode = stat.S_IFMT(member.external_attr >> 16)
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or ":" in name
            or relative.as_posix() != name.rstrip("/")
            or name in seen
            or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
            or member.flag_bits & 1
            or member.compress_type != zipfile.ZIP_STORED
            or (name != "manifest.json" and not name.startswith(prefix))
        ):
            raise CheckpointError(f"Invalid .nqx archive member {name!r}.")
        seen.add(name)
        if name != "manifest.json":
            yield member


def _unpack(path, expected):
    directory = None
    try:
        # Hold one open file descriptor throughout extraction. An atomic
        # overwrite can replace the path without changing this snapshot.
        with zipfile.ZipFile(path, "r") as archive:
            with archive.open("manifest.json") as stream:
                actual = validate_manifest(json.load(stream))
            if actual != expected:
                raise CheckpointError(
                    "The checkpoint changed while preparing restoration. Retry the load."
                )
            members = list(_members(archive, actual))
            directory = Path(
                tempfile.mkdtemp(prefix=f".{path.name}-read-", dir=path.parent)
            )
            for member in members:
                target = directory.joinpath(*PurePosixPath(member.filename).parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=_COPY_BUFFER)
        return str(directory)
    except Exception as error:
        if directory is not None:
            shutil.rmtree(directory)
        if isinstance(error, (zipfile.BadZipFile, KeyError, json.JSONDecodeError)):
            raise CheckpointError(f"Invalid .nqx archive at {path}.") from error
        raise


@contextmanager
def snapshot_directory(path, manifest):
    """Stage one immutable snapshot for collective Orbax reads, then clean it up.

    Only the primary process unpacks the archive, using bounded host buffers
    and a temporary directory beside the source file. All processes therefore
    need the same writable shared parent directory. A legacy directory bundle
    is read in place. No tensor tree is gathered onto the primary process.
    """
    path = Path(path)

    def prepare():
        if path.is_dir():
            if read_manifest(path) != manifest:
                raise CheckpointError(
                    "The checkpoint changed while preparing restoration. Retry the load."
                )
            return {"directory": str(path), "temporary": False}
        return {"directory": _unpack(path, manifest), "temporary": True}

    staged = primary_call(prepare, "checkpoint archive staging")
    try:
        yield Path(staged["directory"]) / "snapshots" / manifest["snapshot"]
    finally:
        if staged["temporary"]:
            primary_call(
                lambda: shutil.rmtree(staged["directory"]), "checkpoint archive cleanup"
            )
