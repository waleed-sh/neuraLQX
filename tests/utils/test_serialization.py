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


import hashlib
from pathlib import Path

import msgpack
import numpy as np
import pytest

from neuralqx.utils import serialization as ser


class Plain:
    def __init__(self):
        self.a = 1
        self.b = "x"

        self._callable = lambda: 123


class WithCustomSerializable:
    def __init__(self):
        self.keep = 7
        self.drop = 999

    def __serializable_attributes__(self):
        return {"keep": self.keep}


class WithPostHook:
    __class_version__ = 3

    def __init__(self, x):
        self.x = x
        self.hook_called = False

    def _post_load_hook(self):
        self.hook_called = True
        self.ephemeral = self.x * 2


class SlotsOnly:
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a


@ser.auto_serializable
class AutoDecorated:
    def __init__(self):
        self.v = 5

        self.fn = lambda: 0


@ser.auto_serializable
class AutoDecoratedWithExisting:
    def __init__(self):
        self.v = 1

    def __serializable_attributes__(self):
        return {"v": 123}


def test_dynamic_import_class_success():
    cls = ser.SerializerEngine.dynamic_import_class("pathlib.Path")
    assert cls is Path


def test_dynamic_import_class_failure():
    with pytest.raises((ImportError, AttributeError)):
        ser.SerializerEngine.dynamic_import_class("no_such_module.Nope")


def test_compute_sha256_matches_hashlib():
    data = b"abc"
    expected = hashlib.sha256(data).hexdigest()
    assert ser.SerializerEngine.compute_sha256(data) == expected


def test_get_fully_qualified_name_custom_class():
    obj = Plain()
    name = ser.SerializerEngine.get_fully_qualified_name(obj)
    assert name.endswith(".Plain")
    assert "." in name


def test_maybe_post_load_hook_calls_only_when_present():
    obj = WithPostHook(10)
    assert obj.hook_called is False
    ser.SerializerEngine.maybe_post_load_hook(obj)
    assert obj.hook_called is True
    assert obj.ephemeral == 20

    ser.SerializerEngine.maybe_post_load_hook(Plain())


def test_get_serializable_state_prefers_custom_method():
    obj = WithCustomSerializable()
    state = ser.SerializerEngine.get_serializable_state(obj)
    assert state == {"keep": 7}
    assert "drop" not in state


def test_get_serializable_state_uses_dict_and_filters_callables():
    obj = Plain()
    state = ser.SerializerEngine.get_serializable_state(obj)
    assert state["a"] == 1
    assert state["b"] == "x"
    assert "_callable" not in state


def test_get_serializable_state_no_dict_returns_empty():
    obj = SlotsOnly(10)
    state = ser.SerializerEngine.get_serializable_state(obj)
    assert state == {}


@pytest.mark.parametrize("value", [None, True, False, 0, 123, -5, 1.25, "hi"])
def test_serialize_deserialize_primitives(value):
    snap = ser.SerializerEngine._serialize_any(value)
    out = ser.SerializerEngine._deserialize_any(snap)
    assert out == value


def test_serialize_deserialize_complex():
    z = 3.25 - 1.5j
    snap = ser.SerializerEngine._serialize_any(z)
    assert snap["__is_complex__"] is True
    out = ser.SerializerEngine._deserialize_any(snap)
    assert out == z


def test_serialize_deserialize_bytes():
    b = b"\x00\x01\x02hello"
    snap = ser.SerializerEngine._serialize_any(b)
    assert snap["__is_bytes__"] is True
    out = ser.SerializerEngine._deserialize_any(snap)
    assert out == b


def test_serialize_deserialize_numpy_array_roundtrip():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    snap = ser.SerializerEngine._serialize_any(arr)
    assert snap["__is_numpy_array__"] is True
    out = ser.SerializerEngine._deserialize_any(snap)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)
    assert out.dtype == np.int32
    assert np.array_equal(out, arr)


def test_serialize_deserialize_numpy_dtype():
    dt = np.dtype("float32")
    snap = ser.SerializerEngine._serialize_any(dt)
    assert snap["__is_numpy_dtype__"] is True
    out = ser.SerializerEngine._deserialize_any(snap)
    assert out == dt


def test_serialize_numpy_generic_becomes_python_scalar():
    x = np.float32(1.5)
    snap = ser.SerializerEngine._serialize_any(x)
    assert isinstance(snap, float)
    out = ser.SerializerEngine._deserialize_any(snap)
    assert isinstance(out, float)
    assert out == pytest.approx(1.5)


def test_serialize_deserialize_containers_nested():
    obj = {
        "a": [1, 2, (3, 4)],
        "b": {"x": "y"},
        "c": set([1, 2]),
    }
    snap = ser.SerializerEngine._serialize_any(obj)
    out = ser.SerializerEngine._deserialize_any(snap)
    assert out["a"] == [1, 2, (3, 4)]
    assert out["b"] == {"x": "y"}
    assert out["c"] == set([1, 2])


def test_deserialize_plain_dict_without_markers_recurses():
    snap = {"x": {"__is_bytes__": True, "data": b"ok"}}
    out = ser.SerializerEngine._deserialize_any(snap)
    assert out == {"x": b"ok"}


def test_public_serialize_container_path():
    data = {"k": [1, 2, 3]}
    snap = ser.serialize(data)
    assert isinstance(snap, dict)
    assert snap.get("__is_dict__") is True

    out = ser.deserialize(snap)
    assert out == data


def test_object_serialize_deserialize_roundtrip_calls_post_hook():
    obj = WithPostHook(21)
    snap = ser.serialize(obj)
    assert snap["__classname__"].endswith(".WithPostHook")
    assert snap["__class_version__"] == 3

    restored = ser.deserialize(snap)
    assert isinstance(restored, WithPostHook)
    assert restored.x == 21
    assert restored.hook_called is True
    assert restored.ephemeral == 42


def test_auto_serializable_adds_method_and_filters_callables():
    obj = AutoDecorated()
    snap = ser.serialize(obj)
    restored = ser.deserialize(snap)

    assert isinstance(restored, AutoDecorated)
    assert restored.v == 5
    assert not hasattr(restored, "fn")


def test_auto_serializable_does_not_override_existing():
    obj = AutoDecoratedWithExisting()
    snap = ser.serialize(obj)
    restored = ser.deserialize(snap)

    assert isinstance(restored, AutoDecoratedWithExisting)
    assert restored.v == 123


def test_save_load_roundtrip_object(tmp_path_writable: Path):
    p = tmp_path_writable / "obj.msgpack"
    obj = WithPostHook(11)
    ser.save_to_file(obj, str(p))

    restored = ser.load_from_file(str(p))
    assert isinstance(restored, WithPostHook)
    assert restored.x == 11
    assert restored.hook_called is True
    assert restored.ephemeral == 22


def test_save_load_roundtrip_dict_path(tmp_path_writable: Path):
    p = tmp_path_writable / "dict.msgpack"
    data = {"a": 1, "b": [2, 3]}
    ser.save_to_file(data, str(p))

    out = ser.load_from_file(str(p))
    assert out == data


def test_load_raw_returns_data_not_object(tmp_path_writable: Path):
    p = tmp_path_writable / "raw.msgpack"
    obj = WithPostHook(9)
    ser.save_to_file(obj, str(p))

    raw = ser.load_from_file(str(p), raw=True)
    assert isinstance(raw, dict)
    assert raw["__classname__"].endswith(".WithPostHook")
    assert "__attributes__" in raw


def test_load_from_file_checksum_mismatch_raises(tmp_path_writable: Path):
    p = tmp_path_writable / "tamper.msgpack"
    obj = Plain()
    ser.save_to_file(obj, str(p))

    final_structure = msgpack.unpackb(p.read_bytes(), raw=False)

    final_structure["checksum"] = "0" * 64

    p.write_bytes(msgpack.packb(final_structure, use_bin_type=True))

    with pytest.raises(ValueError, match="Checksum mismatch"):
        ser.load_from_file(str(p))


def test_load_from_file_malformed_missing_keys_raises(tmp_path_writable: Path):
    p = tmp_path_writable / "malformed.msgpack"

    bad = {"snapshot": {"x": 1}, "checksum": "nope"}
    p.write_bytes(msgpack.packb(bad, use_bin_type=True))

    with pytest.raises(ValueError, match="Malformed serialization file"):
        ser.load_from_file(str(p))


def test_load_from_file_invalid_msgpack_raises(tmp_path_writable: Path):
    p = tmp_path_writable / "invalid.msgpack"
    p.write_bytes(b"not msgpack")

    with pytest.raises(ValueError, match="Failed to unpack msgpack data"):
        ser.load_from_file(str(p))


def test_load_from_file_format_version_warns(tmp_path_writable: Path):
    p = tmp_path_writable / "warn.msgpack"
    obj = WithPostHook(5)
    snapshot = ser.SerializerEngine.serialize(obj)
    snapshot_bytes = msgpack.packb(snapshot, use_bin_type=True)
    checksum = ser.SerializerEngine.compute_sha256(snapshot_bytes)

    final_structure = {
        "snapshot": snapshot,
        "checksum": checksum,
        "format_version": 999,
    }
    p.write_bytes(msgpack.packb(final_structure, use_bin_type=True))

    with pytest.warns(UserWarning):
        out = ser.load_from_file(str(p))
    assert isinstance(out, WithPostHook)
    assert out.x == 5


def test_partial_serialize_missing_attr_becomes_none():
    obj = Plain()
    partial = ser.partial_serialize(obj, ["a", "missing_attr"])
    assert "a" in partial
    assert "missing_attr" in partial
    assert ser.SerializerEngine._deserialize_any(partial["a"]) == 1
    assert ser.SerializerEngine._deserialize_any(partial["missing_attr"]) is None


def test_partial_deserialize_updates_existing_object():
    obj = Plain()
    partial = {
        "a": ser.SerializerEngine._serialize_any(999),
        "new": ser.SerializerEngine._serialize_any("ok"),
    }
    ser.partial_deserialize(partial, obj)
    assert obj.a == 999
    assert obj.new == "ok"


def test_save_load_multi_partial_merges_in_place_and_returns_master(
    tmp_path_writable: Path,
):
    p = tmp_path_writable / "multi.msgpack"

    a = Plain()
    b = Plain()
    a.a = 1
    b.a = 2

    obj_map_to_save = {
        "A": (a, ["a", "b"]),
        "B": (b, ["a"]),
    }
    ser.save_multi_partial(obj_map_to_save, str(p))

    target_A = Plain()
    target_B = Plain()

    target_C = Plain()

    target_A.a = -1
    target_B.a = -1
    target_C.a = -1

    out = ser.load_multi_partial({"A": target_A, "B": target_B, "C": target_C}, str(p))

    assert out["A"]["a"] is not None
    assert target_A.a == 1

    assert target_A.b == "x"
    assert target_B.a == 2

    assert target_C.a == -1


def test_load_multi_partial_checksum_mismatch_raises(tmp_path_writable: Path):
    p = tmp_path_writable / "multi_tamper.msgpack"

    a = Plain()
    ser.save_multi_partial({"A": (a, ["a"])}, str(p))

    final_structure = msgpack.unpackb(p.read_bytes(), raw=False)
    final_structure["checksum"] = "0" * 64

    p.write_bytes(msgpack.packb(final_structure, use_bin_type=True))

    with pytest.raises(ValueError, match="Checksum mismatch"):
        ser.load_multi_partial({"A": Plain()}, str(p))


def test_load_multi_partial_malformed_missing_keys_raises(tmp_path_writable: Path):
    p = tmp_path_writable / "multi_malformed.msgpack"

    bad = {"multi_partial_data": {}, "checksum": "x"}
    p.write_bytes(msgpack.packb(bad, use_bin_type=True))

    with pytest.raises(ValueError, match="Malformed multi-partial file"):
        ser.load_multi_partial({"A": Plain()}, str(p))


def test_load_multi_partial_invalid_msgpack_raises(tmp_path_writable: Path):
    p = tmp_path_writable / "multi_invalid.msgpack"
    p.write_bytes(b"not msgpack")

    with pytest.raises(ValueError, match="Failed to unpack msgpack data"):
        ser.load_multi_partial({"A": Plain()}, str(p))


@pytest.mark.skipif(not getattr(ser, "_HAS_jax", False), reason="JAX not installed")
def test_jax_array_roundtrip_in_memory():
    import jax.numpy as jnp

    x = jnp.arange(6).reshape(2, 3)
    snap = ser.SerializerEngine._serialize_any(x)
    assert snap["__is_jax_array__"] is True

    out = ser.SerializerEngine._deserialize_any(snap)
    assert hasattr(out, "shape")
    assert tuple(out.shape) == (2, 3)


def test_set_roundtrip_in_memory_ok():
    s = {1, 2, 3}
    snap = ser.SerializerEngine._serialize_any(s)
    out = ser.SerializerEngine._deserialize_any(snap)
    assert out == s


@pytest.mark.xfail(
    reason="msgpack cannot encode Python 'set' objects,"
    " current implementation stores a set under '__is_set__'."
)
def test_set_save_to_file_should_work_future(tmp_path_writable: Path):
    p = tmp_path_writable / "set.msgpack"
    ser.save_to_file({"s": {1, 2}}, str(p))
    assert ser.load_from_file(str(p)) == {"s": {1, 2}}


def test_load_from_file_data_corruption_triggers_checksum_mismatch(
    tmp_path_writable: Path,
):
    p = tmp_path_writable / "corrupt_snapshot.msgpack"
    obj = Plain()
    ser.save_to_file(obj, str(p))

    final_structure = msgpack.unpackb(p.read_bytes(), raw=False)

    snapshot = final_structure["snapshot"]
    assert "__attributes__" in snapshot
    attrs = snapshot["__attributes__"]
    assert attrs.get("__is_dict__") is True

    items = list(attrs["items"])
    found = False
    for i, (k, v) in enumerate(items):
        if k == "a":
            items[i] = (k, ser.SerializerEngine._serialize_any(999))
            found = True
            break
    assert found, "Expected attribute 'a' to exist in serialized Plain() snapshot"

    attrs["items"] = items
    final_structure["snapshot"] = snapshot

    p.write_bytes(msgpack.packb(final_structure, use_bin_type=True))

    with pytest.raises(ValueError, match="Checksum mismatch"):
        ser.load_from_file(str(p))


def test_load_multi_partial_data_corruption_triggers_checksum_mismatch(
    tmp_path_writable: Path,
):
    p = tmp_path_writable / "corrupt_multi.msgpack"

    a = Plain()
    ser.save_multi_partial({"A": (a, ["a"])}, str(p))

    final_structure = msgpack.unpackb(p.read_bytes(), raw=False)

    master = final_structure["multi_partial_data"]
    assert "A" in master and "a" in master["A"]

    master["A"]["a"] = ser.SerializerEngine._serialize_any(12345)
    final_structure["multi_partial_data"] = master

    p.write_bytes(msgpack.packb(final_structure, use_bin_type=True))

    with pytest.raises(ValueError, match="Checksum mismatch"):
        ser.load_multi_partial({"A": Plain()}, str(p))
