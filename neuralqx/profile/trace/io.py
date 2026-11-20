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


"""Atomic files and incremental Chrome Trace Event JSON input and output."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from collections.abc import Iterable
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import TextIO

from ..errors import ProfileError


@contextmanager
def atomic_file(path: Path, *, binary: bool = False):
    """Yield a temporary output stream and replace the destination only on success.

    Args:
        path: Destination file. Parent directories are created as needed.
        binary: Whether to expose a binary stream instead of UTF-8 text.

    Yields:
        Writable stream backed by a temporary file on the destination filesystem.

    An exception leaves the previous destination intact and removes the temporary
    file. Atomic replacement does not imply a durable filesystem fsync.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(
            fd, "wb" if binary else "w", encoding=None if binary else "utf-8"
        ) as stream:
            yield stream
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    """Atomically serialize a finite JSON control value.

    Args:
        path: Destination control file.
        value: JSON-serializable value. Nonfinite numbers are rejected.
    """
    with atomic_file(path) as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))


def read_json(path: Path) -> Any:
    """Read one UTF-8 JSON control file.

    Args:
        path: Completed control file to load.

    Returns:
        Decoded JSON value. Trace event streams use iter_events instead.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_trace(path: Path, events: Iterable[dict], metadata: dict) -> None:
    """Atomically stream a Chrome Trace Event artifact with optional gzip compression.

    Args:
        path: Destination path. A .gz suffix selects gzip.
        events: Iterable yielding one event dictionary at a time.
        metadata: Capture metadata stored under the neuralqx key.
    """
    with atomic_file(path, binary=True) as raw:
        if path.suffix == ".gz":
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                _write_events(compressed, events, metadata)
        else:
            _write_events(raw, events, metadata)


def _write_events(stream, events, metadata) -> None:
    stream.write(b'{"displayTimeUnit":"ms","traceEvents":[')
    separator = b""
    for event in events:
        stream.write(separator)
        stream.write(
            json.dumps(event, allow_nan=False, separators=(",", ":")).encode("utf-8")
        )
        separator = b","
    stream.write(b'],"neuralqx":')
    stream.write(
        json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode("utf-8")
    )
    stream.write(b"}")


class _JsonStream:
    """Incremental JSON token reader with a bounded largest individual value."""

    def __init__(self, stream: TextIO):
        self.stream = stream
        """Readable text stream supplying JSON chunks."""

        self.buffer = ""
        """Unconsumed text plus the most recently read chunk."""

        self.position = 0
        """Index of the next unread character in buffer."""

        self.eof = False
        """Whether the underlying text stream has reached EOF."""

        self.decoder = json.JSONDecoder()
        """JSON decoder used for one complete value at a time."""

    def fill(self) -> None:
        """Discard consumed text and read the next bounded chunk."""
        self.buffer = self.buffer[self.position :]
        self.position = 0
        chunk = self.stream.read(64 * 1024)
        self.eof = not chunk
        self.buffer += chunk
        if len(self.buffer) > 64 * 1024 * 1024:
            raise ProfileError("A trace JSON value exceeds the 64 MiB parsing limit.")

    def peek(self) -> str:
        """Return the next non-whitespace character, or an empty string at EOF."""
        while True:
            while (
                self.position < len(self.buffer)
                and self.buffer[self.position].isspace()
            ):
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.fill()

    def expect(self, token: str) -> None:
        """Consume one required punctuation token or raise a malformed-trace error."""
        if self.peek() != token:
            raise ProfileError(f"Malformed trace JSON, expected {token!r}.")
        self.position += 1

    def value(self):
        """Decode one complete JSON value, reading more chunks when necessary."""
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                # A number can end exactly at a read boundary and continue in
                # the next chunk. Read more before accepting that token.
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                self.position = end
                return value
            except json.JSONDecodeError as error:
                if self.eof:
                    raise ProfileError("Truncated or malformed trace JSON.") from error
                self.fill()


def iter_events(path: Path) -> Iterator[dict]:
    """Read native or normalized trace events without loading the complete trace.

    Args:
        path: JSON trace path, optionally gzip-compressed with a .gz suffix.

    Yields:
        Nonempty event dictionaries from the traceEvents array.

    Raises:
        ProfileError: The JSON is malformed, lacks traceEvents, or one parsed
            value exceeds the 64 MiB safety limit.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        reader = _JsonStream(stream)
        reader.expect("{")
        found = False
        while reader.peek() != "}":
            key = reader.value()
            if not isinstance(key, str):
                raise ProfileError("Trace JSON object keys must be strings.")
            reader.expect(":")
            if key == "traceEvents":
                if found:
                    raise ProfileError("Duplicate traceEvents array.")
                found = True
                reader.expect("[")
                while reader.peek() != "]":
                    event = reader.value()
                    if not isinstance(event, dict):
                        raise ProfileError("Trace events must be JSON objects.")
                    if event:
                        yield event
                    if reader.peek() != "]":
                        reader.expect(",")
                        if reader.peek() == "]":
                            raise ProfileError("Trailing comma in traceEvents.")
                reader.expect("]")
            else:
                reader.value()
            if reader.peek() != "}":
                reader.expect(",")
                if reader.peek() == "}":
                    raise ProfileError("Trailing comma in trace JSON.")
        reader.expect("}")
        if reader.peek() or not found:
            raise ProfileError("Expected one JSON object containing traceEvents.")
