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


import json
import reprlib
from itertools import islice

_MISSING = object()
_REPR = reprlib.Repr()
_REPR.maxstring = 120
_REPR.maxother = 120


def _preview(value):
    if value is _MISSING:
        return "<missing>"
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind in ("dtype", "activation"):
            return _REPR.repr(value["name"])
        if kind == "component":
            return f"{value['name']}({_preview(value['config'])})"
        if kind in ("mapping", "frozen_mapping"):
            entries = value["items"]
            body = ", ".join(f"{_preview(k)}: {_preview(v)}" for k, v in entries[:4])
            return "{" + body + (", ..." if len(entries) > 4 else "") + "}"
        if kind in ("tuple", "list"):
            entries = value["items"]
            body = ", ".join(_preview(v) for v in entries[:6])
            body += ", ..." if len(entries) > 6 else ""
            if kind == "tuple":
                return "(" + body + ("," if len(entries) == 1 else "") + ")"
            return "[" + body + "]"
        if kind == "array":
            return f"array(shape={value['shape']}, dtype={value['dtype']!r})"
    return _REPR.repr(value)


def _line(path, saved, target):
    # Keep a graph or nested operator recipe from flooding an exception.
    def short(value):
        text = _preview(value)
        return text if len(text) <= 180 else text[:177] + "..."

    return f"{path}: saved={short(saved)}, target={short(target)}"


def _field(path, key):
    if isinstance(key, str) and key.isidentifier():
        return f"{path}.{key}" if path else key
    return f"{path}[{_preview(key)}]"


def _mapping_differences(saved, target, path, *, ordered=False):
    # Encoded keys can themselves be tuples or components. Compare their
    # data-only representations instead of decoding arbitrary registered code.
    old = {json.dumps(k, sort_keys=True): (k, v) for k, v in saved}
    new = {json.dumps(k, sort_keys=True): (k, v) for k, v in target}
    for token in dict.fromkeys((*old, *new)):
        key = old[token][0] if token in old else new[token][0]
        a = old[token][1] if token in old else _MISSING
        b = new[token][1] if token in new else _MISSING
        yield from differences(a, b, _field(path, key))
    if ordered and old.keys() == new.keys() and list(old) != list(new):
        yield _line(f"{path}.key_order", [k for k, _ in saved], [k for k, _ in target])


def differences(saved, target, path):
    """Yield named fields and saved/target values from plain or encoded metadata."""
    if saved == target:
        return
    if isinstance(saved, dict) and isinstance(target, dict):
        kind = saved.get("kind")
        if kind == target.get("kind"):
            if kind in ("mapping", "frozen_mapping"):
                yield from _mapping_differences(
                    saved["items"], target["items"], path, ordered=True
                )
                return
            if kind == "component":
                yield from differences(saved["name"], target["name"], f"{path}.type")
                yield from differences(
                    saved["version"], target["version"], f"{path}.codec_version"
                )
                yield from differences(saved["config"], target["config"], path)
                return
            if kind in ("dtype", "activation"):
                yield from differences(saved["name"], target["name"], path)
                old = [(k, v) for k, v in saved.items() if k not in ("kind", "name")]
                new = [(k, v) for k, v in target.items() if k not in ("kind", "name")]
                yield from _mapping_differences(old, new, path)
                return
            if kind in ("tuple", "list"):
                a, b = saved["items"], target["items"]
                if (
                    len(a) <= 6
                    and len(b) <= 6
                    and all(not isinstance(v, (dict, list)) for v in (*a, *b))
                ):
                    yield _line(path, saved, target)
                else:
                    yield from differences(a, b, path)
                return
        yield from _mapping_differences(list(saved.items()), list(target.items()), path)
    elif isinstance(saved, (list, tuple)) and isinstance(target, type(saved)):
        if (
            len(saved) <= 6
            and len(target) <= 6
            and all(not isinstance(v, (dict, list, tuple)) for v in (*saved, *target))
        ):
            yield _line(path, saved, target)
            return
        if len(saved) != len(target):
            yield _line(f"{path}.length", len(saved), len(target))
        for index, (a, b) in enumerate(zip(saved, target, strict=False)):
            yield from differences(a, b, f"{path}[{index}]")
    else:
        yield _line(path, saved, target)


def mismatch_report(title, entries, *, hint=None):
    """Format at most eight differences, marking longer reports explicitly."""
    lines = list(islice(entries, 9))
    result = title + "\n" + "\n".join(f"  - {line}" for line in lines[:8])
    if len(lines) > 8:
        result += "\n  - Additional differences omitted."
    if hint:
        result += "\n" + hint
    return result


def mismatch_message(title, saved, target, *, path, hint=None):
    return mismatch_report(title, differences(saved, target, path), hint=hint)
