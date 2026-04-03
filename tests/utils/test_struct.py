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


import inspect
import io
import json
from pathlib import Path
from typing import ClassVar
import zipfile

import numpy as np
import pytest


class _ImportableStructRegistryTarget:
    pass


class _AutoTree:
    def __init__(self, value):
        self.value = value

    def tree_flatten(self):
        return (self.value,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])


class _HalfAutoTree:
    def __init__(self, value):
        self.value = value

    def tree_flatten(self):
        return (self.value,), None

    tree_unflatten = None


def _assert_array_equal(a, b):
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_field_rejects_default_and_default_factory_together(nqx):
    s = nqx.experimental.utils.struct
    with pytest.raises(ValueError, match="both 'default' and 'default_factory'"):
        s.field(default=1, default_factory=lambda: 1)


def test_field_rejects_static_opaque_combo(nqx):
    s = nqx.experimental.utils.struct
    with pytest.raises(ValueError, match="both static and opaque"):
        s.field(static=True, pytree=False)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"init": True, "static": True}, "must set init=False"),
        ({"init": False, "static": True, "default": 1}, "cannot define a default"),
        ({"init": False}, "must be static or opaque"),
        ({"init": False, "static": True, "serialize": True}, "cannot be serialized"),
    ],
)
def test_field_derived_constraints(nqx, kwargs, message):
    s = nqx.experimental.utils.struct
    with pytest.raises(ValueError, match=message):
        s.field(derived=lambda self: 1, **kwargs)


def test_fieldspec_default_materialization(nqx):
    s = nqx.experimental.utils.struct

    spec_default = s.field(default=7)
    assert spec_default.has_default is True
    assert spec_default.make_default() == 7

    spec_factory = s.field(default_factory=list)
    assert spec_factory.has_default is True
    a = spec_factory.make_default()
    b = spec_factory.make_default()
    assert a == []
    assert b == []
    assert a is not b

    spec_required = s.field()
    assert spec_required.has_default is False
    with pytest.raises(TypeError, match="requires a value"):
        spec_required.make_default()


def test_struct_class_definition_rejects_required_after_default(nqx):
    s = nqx.experimental.utils.struct
    ValidationError = nqx.utils.errors.ValidationError

    with pytest.raises(ValidationError, match="required positional field follows"):

        class _Bad(s.Struct):
            a: int = 0
            b: int


def test_classvar_annotations_are_ignored(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        value: int
        marker: ClassVar[int] = 3

    assert tuple(s.fields(_Model).keys()) == ("value",)


def test_generated_init_signature_includes_kw_only_and_factory_marker(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        a: int
        b: int = s.field(default=1)
        c: list[int] = s.field(default_factory=list, kw_only=True)

    sig = str(inspect.signature(_Model))
    assert "*, c=<default>" in sig
    assert "b=1" in sig


def test_generated_init_rejects_too_many_positional_args(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        a: int

    with pytest.raises(TypeError, match="at most 1 positional argument"):
        _Model(1, 2)


def test_generated_init_rejects_multiple_values(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        a: int

    with pytest.raises(TypeError, match="multiple values"):
        _Model(1, a=2)


def test_generated_init_rejects_unknown_keyword(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        a: int

    with pytest.raises(TypeError, match="unexpected keyword"):
        _Model(a=1, b=2)


def test_generated_init_rejects_missing_required(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        a: int
        b: int

    with pytest.raises(TypeError, match="Missing required field 'b'"):
        _Model(1)


def test_non_init_non_derived_field_without_default_fails_at_instantiation(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        a: int
        b: int = s.field(init=False)

    with pytest.raises(TypeError, match="init=False and no default"):
        _Model(1)


def test_derived_field_rejects_explicit_input_in_custom_init_path(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        a: int
        doubled: int = s.field(static=True, init=False, derived=lambda self: self.a * 2)

        def __init__(self, **kwargs):
            self._init_fields(**kwargs)

    with pytest.raises(TypeError, match="derived and cannot be supplied explicitly"):
        _Model(a=1, doubled=4)


def test_converter_supports_value_only_and_self_value_signatures(nqx):
    s = nqx.experimental.utils.struct

    class _ValueOnly(s.Struct):
        x: int = s.field(converter=int)

    class _SelfValue(s.Struct):
        offset: int = s.field(static=True)
        x: int = s.field(converter=lambda self, value: value + self.offset)

    v = _ValueOnly("3")
    sv = _SelfValue(2, 5)
    assert v.x == 3
    assert sv.x == 7


def test_converter_invalid_arity_raises_type_error(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int = s.field(converter=lambda a, b, c: c)

    with pytest.raises(TypeError, match="1 or 2 positional argument"):
        _Model(1)


def test_validator_supports_sequence_and_false_result_raises(nqx):
    s = nqx.experimental.utils.struct
    ValidationError = nqx.utils.errors.ValidationError
    calls = []

    def _v1(value):
        calls.append(("v1", value))
        return True

    def _v2(self, value):
        calls.append(("v2", value))
        return False

    class _Model(s.Struct):
        x: int = s.field(validator=[_v1, _v2])

    with pytest.raises(ValidationError, match="Validation failed for field"):
        _Model(3)

    assert calls == [("v1", 3), ("v2", 3)]


def test_post_init_and_derived_recompute_order(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int
        y: int = s.field(static=True, init=False, derived=lambda self: self.x * 2)

        def __post_init__(self):
            self.x = self.x + 1

    m = _Model(2)
    assert m.x == 3
    assert m.y == 6


def test_derived_supports_zero_arg_callable(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int
        y: int = s.field(static=True, init=False, derived=lambda: 11)

    m = _Model(1)
    assert m.y == 11


def test_derived_invalid_arity_raises_type_error(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int
        y: int = s.field(static=True, init=False, derived=lambda a, b: 1)

    with pytest.raises(TypeError, match="0 or 1 positional argument"):
        _Model(1)


def test_struct_is_frozen_after_init(nqx):
    s = nqx.experimental.utils.struct
    FrozenStructError = nqx.utils.errors.FrozenStructError

    class _Model(s.Struct):
        x: int

    m = _Model(1)
    with pytest.raises(FrozenStructError):
        m.x = 2
    with pytest.raises(FrozenStructError):
        del m.x


def test_replace_rejects_unknown_and_derived_updates(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int
        y: int = s.field(static=True, init=False, derived=lambda self: self.x + 1)

    m = _Model(2)
    with pytest.raises(TypeError, match="Unknown field"):
        m.replace(z=1)
    with pytest.raises(TypeError, match="Derived field"):
        m.replace(y=10)


def test_replace_preserves_non_init_opaque_field_identity(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int
        token: object = s.field(pytree=False, init=False, default_factory=object)

    m = _Model(1)
    m2 = m.replace(x=2)
    assert m2.x == 2
    assert m2.token is m.token


def test_rederive_recomputes_after_nested_mutation(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        data: list[int]
        n: int = s.field(static=True, init=False, derived=lambda self: len(self.data))

    m = _Model([1])
    assert m.n == 1
    m.data.append(2)
    assert m.n == 1
    m.rederive()
    assert m.n == 2


def test_repr_and_compare_semantics(nqx):
    s = nqx.experimental.utils.struct

    class _ReprModel(s.Struct):
        x: np.ndarray
        hidden: int = s.field(default=1, repr=False)

    class _CompareModel(s.Struct):
        x: int
        token: object = s.field(pytree=False, default_factory=object, compare=False)

    rm = _ReprModel(np.array([1, 2]), 5)
    text = repr(rm)
    assert "x=" in text
    assert "hidden=" not in text

    assert _CompareModel(1) == _CompareModel(1)
    assert _CompareModel(1) != _CompareModel(2)


def test_iter_preserves_field_order(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        a: int
        b: int
        c: int = 3

    m = _Model(1, 2)
    assert list(iter(m)) == [1, 2, 3]


def test_tree_size_and_tree_map_preserve_static_and_opaque(nqx, jax, jnp):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: jax.Array
        tag: str = s.field(static=True, default="run")
        token: object = s.field(pytree=False, default_factory=object)

    m = _Model(jnp.array([1, 2]))
    assert m.tree_size() == 1

    mapped = jax.tree_util.tree_map(lambda v: v + 1, m)
    _assert_array_equal(mapped.x, jnp.array([2, 3]))
    assert mapped.tag == "run"
    assert mapped.token is m.token


def test_static_field_rejects_unhashable_values(nqx):
    s = nqx.experimental.utils.struct
    ValidationError = nqx.utils.errors.ValidationError

    class _Model(s.Struct):
        x: int
        meta: list[int] = s.field(static=True, default_factory=list)

    with pytest.raises(ValidationError, match="must be hashable"):
        _Model(1)


def test_static_field_rejects_nested_array_values(nqx):
    s = nqx.experimental.utils.struct
    ValidationError = nqx.utils.errors.ValidationError

    class _Model(s.Struct):
        x: int
        meta: dict[str, np.ndarray] = s.field(
            static=True,
            default_factory=lambda: {"arr": np.array([1, 2])},
        )

    with pytest.raises(ValidationError, match="cannot contain array-like values"):
        _Model(1)


def test_static_field_accepts_numpy_scalar(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int
        count: np.generic = s.field(static=True, default=np.int64(3))

    m = _Model(1)
    assert int(m.count) == 3


def test_to_dict_recursive_and_include_opaque_controls(nqx):
    s = nqx.experimental.utils.struct

    class _Child(s.Struct):
        y: int

    class _Parent(s.Struct):
        child: _Child
        token: object = s.field(pytree=False, default_factory=object)

    p = _Parent(_Child(7))
    recursive = p.to_dict(recursive=True, include_opaque=False)
    assert recursive == {"child": {"y": 7}}

    non_recursive = p.to_dict(recursive=False, include_opaque=True)
    assert isinstance(non_recursive["child"], _Child)
    assert "token" in non_recursive


def test_register_class_preserves_super_runtime_semantics(nqx):
    s = nqx.experimental.utils.struct

    class _Parent:
        def label(self):
            return "parent"

    @s.register_class
    class _Child(_Parent):
        x: int
        y: int

        def pair(self):
            return super().label(), self.x, self.y

    c = _Child(1, 2)
    assert c.pair() == ("parent", 1, 2)


def test_register_class_with_name_override(nqx):
    s = nqx.experimental.utils.struct

    @s.register_class(name="RenamedStruct")
    class _Raw:
        x: int

    assert _Raw.__name__ == "RenamedStruct"
    assert _Raw(1).x == 1


def test_register_class_on_struct_subclass_is_idempotent(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int

    out = s.register_class(_Model)
    assert out is _Model


def test_dataclass_alias_registers_plain_class_as_struct(nqx):
    s = nqx.experimental.utils.struct

    @s.dataclass
    class _Plain:
        x: int

    p = _Plain(1)
    assert isinstance(p, s.Struct)
    assert p.x == 1


def test_module_level_helpers_match_class_helpers(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int
        y: int = s.field(static=True, default=2)
        z: object = s.field(pytree=False, default_factory=object)
        w: int = s.field(static=True, init=False, derived=lambda self: self.x + self.y)

    assert s.fields(_Model) == _Model.fields()
    assert s.node_fields(_Model) == _Model.node_fields() == ("x",)
    assert s.static_fields(_Model) == _Model.static_fields() == ("y", "w")
    assert s.opaque_fields(_Model) == _Model.opaque_fields() == ("z",)
    assert s.derived_fields(_Model) == _Model.derived_fields() == ("w",)


def test_is_registered_pytree_type_reflects_registration(nqx):
    s = nqx.experimental.utils.struct

    class _Node:
        def __init__(self, data):
            self.data = data

    assert s.is_registered_pytree_type(_Node) is False

    s.register_pytree_type(
        _Node,
        flatten=lambda obj: ([obj.data], None),
        unflatten=lambda aux, children: _Node(children[0]),
    )

    assert s.is_registered_pytree_type(_Node) is True


def test_register_pytree_type_flatten_mode_roundtrip(nqx, jax, jnp):
    s = nqx.experimental.utils.struct

    class _Node:
        def __init__(self, data, tag):
            self.data = data
            self.tag = tag

    s.register_pytree_type(
        _Node,
        flatten=lambda obj: ([obj.data], obj.tag),
        unflatten=lambda aux, children: _Node(children[0], aux),
    )

    node = _Node(jnp.array([1.0, 2.0]), "meta")
    leaves = jax.tree_util.tree_leaves(node)
    assert len(leaves) == 1
    _assert_array_equal(leaves[0], node.data)

    restored = s.io.from_state_dict(s.io.to_state_dict(node))
    assert isinstance(restored, _Node)
    assert restored.tag == "meta"
    _assert_array_equal(restored.data, node.data)


def test_register_pytree_type_serializer_deserializer_roundtrip(nqx, jnp):
    s = nqx.experimental.utils.struct

    class _SerializableNode:
        def __init__(self, arr, label):
            self.arr = arr
            self.label = label

    s.register_pytree_type(
        _SerializableNode,
        flatten=lambda obj: ([obj.arr], obj.label),
        unflatten=lambda aux, children: _SerializableNode(children[0], aux),
        serializer=lambda obj: {"arr": np.asarray(obj.arr), "label": obj.label},
        deserializer=lambda payload: _SerializableNode(
            payload["arr"], payload["label"]
        ),
    )

    node = _SerializableNode(jnp.array([3, 4]), "s")
    state = s.io.to_state_dict(node)
    assert state["manifest"]["mode"] == "serializer"

    restored = s.io.from_state_dict(state)
    assert isinstance(restored, _SerializableNode)
    assert restored.label == "s"
    _assert_array_equal(restored.arr, node.arr)


def test_serializer_mode_without_deserializer_raises(nqx, jnp):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Node:
        def __init__(self, arr):
            self.arr = arr

    s.register_pytree_type(
        _Node,
        flatten=lambda obj: ([obj.arr], None),
        unflatten=lambda aux, children: _Node(children[0]),
        serializer=lambda obj: {"arr": np.asarray(obj.arr)},
    )

    state = s.io.to_state_dict(_Node(jnp.array([1])))
    with pytest.raises(SerializationError, match="requires a deserializer"):
        s.io.from_state_dict(state)


def test_register_attrs_type_constructor_is_used(nqx, jnp):
    s = nqx.experimental.utils.struct
    called = {"n": 0}

    class _AttrNode:
        def __init__(self, data, tag):
            self.data = data
            self.tag = tag

    def _ctor(values):
        called["n"] += 1
        obj = object.__new__(_AttrNode)
        obj.data = values["data"]
        obj.tag = values["tag"] + "-rebuilt"
        return obj

    s.register_attrs_type(
        _AttrNode,
        node_fields=("data",),
        static_fields=("tag",),
        constructor=_ctor,
    )

    node = _AttrNode(jnp.array([8]), "x")
    restored = s.io.from_state_dict(s.io.to_state_dict(node))
    assert called["n"] == 1
    assert isinstance(restored, _AttrNode)
    assert restored.tag == "x-rebuilt"
    _assert_array_equal(restored.data, node.data)


def test_register_attrs_type_fallback_for_incompatible_init(nqx, jnp):
    s = nqx.experimental.utils.struct

    class _HardInit:
        def __init__(self):
            raise RuntimeError("should not be called during fallback")

    s.register_attrs_type(
        _HardInit,
        node_fields=("data",),
        static_fields=("tag",),
    )

    node = object.__new__(_HardInit)
    node.data = jnp.array([5, 6])
    node.tag = "safe"
    restored = s.io.from_state_dict(s.io.to_state_dict(node))
    assert isinstance(restored, _HardInit)
    assert restored.tag == "safe"
    _assert_array_equal(restored.data, node.data)


def test_resolve_pytree_spec_autodiscovers_tree_methods(nqx, jnp):
    s = nqx.experimental.utils.struct

    ref = s.class_ref(_AutoTree)
    spec = s.resolve_pytree_spec(ref)
    assert spec.cls is _AutoTree

    obj = _AutoTree(jnp.array([9]))
    restored = s.io.from_state_dict(s.io.to_state_dict(obj))
    assert isinstance(restored, _AutoTree)
    _assert_array_equal(restored.value, obj.value)


def test_resolve_pytree_spec_unknown_reference_raises(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    with pytest.raises(SerializationError, match="No pytree registration found"):
        s.resolve_pytree_spec("builtins:dict")


def test_class_ref_and_resolve_class_roundtrip(nqx):
    s = nqx.experimental.utils.struct
    ref = s.class_ref(Path)
    assert ref.endswith(":Path")
    assert "pathlib" in ref
    assert s.resolve_class(ref) is Path


def test_resolve_class_rejects_invalid_format(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    with pytest.raises(SerializationError, match="Invalid class reference format"):
        s.resolve_class("not_a_valid_ref")


def test_resolve_class_import_failure(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    with pytest.raises(SerializationError, match="Could not import module"):
        s.resolve_class("no_such_module_xyz:Something")


def test_resolve_class_rejects_non_class_target(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    with pytest.raises(SerializationError, match="Resolved reference is not a class"):
        s.resolve_class("builtins:str.lower")


def test_to_python_handles_nested_native_types(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        arr: np.ndarray
        path: Path
        comp: complex
        items: tuple
        bag: set[int]
        frozen: frozenset[int]
        token: object = s.field(pytree=False, default_factory=object)

    m = _Model(
        np.array([1, 2]),
        Path("here"),
        1 + 2j,
        (1, [2, 3]),
        {1, 2},
        frozenset({3, 4}),
    )
    py = s.io.to_python(m, include_opaque=False)

    assert isinstance(py["arr"], np.ndarray)
    assert py["path"] == "here"
    assert py["comp"] == 1 + 2j
    assert py["items"] == (1, [2, 3])
    assert py["bag"] == {1, 2}
    assert py["frozen"] == frozenset({3, 4})
    assert "token" not in py


def test_to_python_for_registered_pytree_returns_structural_view(nqx, jnp):
    s = nqx.experimental.utils.struct

    class _Node:
        def __init__(self, arr, tag):
            self.arr = arr
            self.tag = tag

    s.register_pytree_type(
        _Node,
        flatten=lambda obj: ([obj.arr], obj.tag),
        unflatten=lambda aux, children: _Node(children[0], aux),
    )

    view = s.io.to_python(_Node(jnp.array([1]), "meta"))
    assert view["__kind__"] == "registered-pytree-view"
    assert view["type"] == s.class_ref(_Node)
    assert view["aux"] == "meta"


def test_struct_state_dict_roundtrip_with_supported_value_kinds(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: np.ndarray
        meta: dict[str, tuple[int, int]]
        path: Path
        comp: complex

    m = _Model(
        np.array([1, 2, 3]),
        {"a": (1, 2)},
        Path("abc"),
        2 - 3j,
    )
    restored = _Model.from_state_dict(m.to_state_dict())
    _assert_array_equal(restored.x, m.x)
    assert restored.meta == m.meta
    assert restored.path == m.path
    assert restored.comp == m.comp


def test_from_state_dict_expected_cls_mismatch_raises(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _A(s.Struct):
        x: int

    class _B(s.Struct):
        x: int

    state = _A(1).to_state_dict()
    with pytest.raises(SerializationError, match="Expected _B"):
        s.io.from_state_dict(state, expected_cls=_B)


def test_from_state_dict_missing_array_data_raises(nqx):
    SerializationError = nqx.utils.errors.SerializationError
    with pytest.raises(SerializationError, match="missing 'array_data'"):
        nqx.experimental.utils.struct.io.from_state_dict(
            {"version": 1, "manifest": None, "arrays": {}}
        )


def test_from_state_dict_array_data_must_be_mapping(nqx):
    SerializationError = nqx.utils.errors.SerializationError
    with pytest.raises(SerializationError, match="must be a mapping"):
        nqx.experimental.utils.struct.io.from_state_dict(
            {"version": 1, "manifest": None, "arrays": {}, "array_data": 1}
        )


def test_from_state_dict_rejects_unsupported_version(nqx):
    SerializationError = nqx.utils.errors.SerializationError
    with pytest.raises(SerializationError, match="Unsupported state dict version"):
        nqx.experimental.utils.struct.io.from_state_dict(
            {"version": 2, "manifest": None, "arrays": {}, "array_data": {}}
        )


def test_from_state_dict_rejects_malformed_arrays_section(nqx):
    SerializationError = nqx.utils.errors.SerializationError
    with pytest.raises(SerializationError, match="Malformed 'arrays' section"):
        nqx.experimental.utils.struct.io.from_state_dict(
            {"version": 1, "manifest": None, "arrays": 1, "array_data": {}}
        )


def test_from_state_dict_rejects_missing_array_payload(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Model(s.Struct):
        x: np.ndarray

    state = _Model(np.array([1, 2])).to_state_dict()
    name = next(iter(state["array_data"]))
    del state["array_data"][name]

    with pytest.raises(SerializationError, match="Missing array payload"):
        s.io.from_state_dict(state)


def test_from_state_dict_rejects_array_manifest_shape_mismatch(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Model(s.Struct):
        x: np.ndarray

    state = _Model(np.array([1, 2])).to_state_dict()
    name = next(iter(state["arrays"]))
    state["arrays"][name]["shape"] = [99]

    with pytest.raises(SerializationError, match="shape mismatch"):
        s.io.from_state_dict(state)


def test_from_state_dict_rejects_array_manifest_dtype_mismatch(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Model(s.Struct):
        x: np.ndarray

    state = _Model(np.array([1, 2], dtype=np.int32)).to_state_dict()
    name = next(iter(state["arrays"]))
    state["arrays"][name]["dtype"] = "float32"

    with pytest.raises(SerializationError, match="dtype mismatch"):
        s.io.from_state_dict(state)


def test_from_state_dict_rejects_decode_payload_shape_mismatch(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Model(s.Struct):
        x: np.ndarray

    state = _Model(np.array([1, 2])).to_state_dict()
    state["manifest"]["fields"]["x"]["shape"] = [123]

    with pytest.raises(SerializationError, match="shape mismatch"):
        s.io.from_state_dict(state)


def test_from_state_dict_rejects_decode_payload_dtype_mismatch(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Model(s.Struct):
        x: np.ndarray

    state = _Model(np.array([1, 2], dtype=np.int32)).to_state_dict()
    state["manifest"]["fields"]["x"]["dtype"] = "float64"

    with pytest.raises(SerializationError, match="dtype mismatch"):
        s.io.from_state_dict(state)


def test_from_state_dict_rejects_malformed_payload_type(nqx):
    SerializationError = nqx.utils.errors.SerializationError
    with pytest.raises(SerializationError, match="Malformed serialized payload"):
        nqx.experimental.utils.struct.io.from_state_dict(
            {"version": 1, "manifest": object(), "arrays": {}, "array_data": {}}
        )


def test_from_state_dict_rejects_unknown_payload_kind(nqx):
    SerializationError = nqx.utils.errors.SerializationError
    with pytest.raises(SerializationError, match="Unknown serialized payload kind"):
        nqx.experimental.utils.struct.io.from_state_dict(
            {
                "version": 1,
                "manifest": {"__kind__": "unknown-kind"},
                "arrays": {},
                "array_data": {},
            }
        )


def test_from_state_dict_rejects_struct_payload_for_non_struct_class(nqx):
    SerializationError = nqx.utils.errors.SerializationError
    with pytest.raises(SerializationError, match="not a Struct-compatible type"):
        nqx.experimental.utils.struct.io.from_state_dict(
            {
                "version": 1,
                "manifest": {
                    "__kind__": "struct",
                    "type": "builtins:tuple",
                    "fields": {},
                },
                "arrays": {},
                "array_data": {},
            }
        )


def test_from_state_dict_rejects_missing_required_serialized_field(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Model(s.Struct):
        x: int
        y: int

    state = _Model(1, 2).to_state_dict()
    del state["manifest"]["fields"]["y"]

    with pytest.raises(SerializationError, match="Missing serialized field 'y'"):
        s.io.from_state_dict(state)


def test_from_state_dict_rejects_unserialized_field_without_default(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Model(s.Struct):
        token: object = s.field(pytree=False)

    state = _Model(object()).to_state_dict()
    with pytest.raises(
        SerializationError, match="was not serialized and has no default"
    ):
        _Model.from_state_dict(state)


def test_to_state_dict_rejects_unsupported_runtime_type(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Model(s.Struct):
        x: object

    with pytest.raises(
        SerializationError, match="Cannot serialize value of type object"
    ):
        _Model(object()).to_state_dict()


def test_export_load_directory_roundtrip(nqx, tmp_path_writable):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: np.ndarray
        tag: str = s.field(static=True, default="run")

    m = _Model(np.array([1, 2]), "A")
    out = tmp_path_writable / "struct_bundle"
    exported = m.export(out)
    assert exported == out
    assert out.joinpath("manifest.json").exists()
    assert out.joinpath("arrays.npz").exists()

    loaded = _Model.load(out)
    _assert_array_equal(loaded.x, m.x)
    assert loaded.tag == "A"


def test_export_load_zip_roundtrip(nqx, tmp_path_writable):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: np.ndarray

    m = _Model(np.array([5, 6]))
    out = tmp_path_writable / "struct_bundle.zip"
    exported = m.export(out)
    assert exported == out
    assert out.exists()

    loaded = _Model.load(out)
    _assert_array_equal(loaded.x, m.x)


def test_export_existing_path_without_overwrite_raises(nqx, tmp_path_writable):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int

    target = tmp_path_writable / "exists"
    target.mkdir()
    with pytest.raises(FileExistsError):
        _Model(1).export(target, overwrite=False)


def test_load_missing_path_raises(nqx, tmp_path_writable):
    with pytest.raises(FileNotFoundError):
        nqx.experimental.utils.struct.io.load(tmp_path_writable / "does-not-exist")


def test_load_rejects_malformed_manifest_payload(nqx, tmp_path_writable):
    SerializationError = nqx.utils.errors.SerializationError

    bad = tmp_path_writable / "bad_bundle"
    bad.mkdir()
    bad.joinpath("manifest.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    np.savez_compressed(bad / "arrays.npz")

    with pytest.raises(SerializationError, match="Malformed manifest payload"):
        nqx.experimental.utils.struct.io.load(bad)


def test_field_kind_selection_and_metadata_copy(nqx):
    s = nqx.experimental.utils.struct
    metadata = {"role": "node"}

    node = s.field(metadata=metadata)
    static = s.field(static=True)
    opaque = s.field(pytree=False)

    metadata["role"] = "changed"
    assert node.kind is s.FieldKind.NODE
    assert static.kind is s.FieldKind.STATIC
    assert opaque.kind is s.FieldKind.OPAQUE
    assert node.metadata["role"] == "node"
    assert node.metadata is not metadata


def test_fieldspec_should_serialize_policy(nqx):
    s = nqx.experimental.utils.struct

    node = s.field()
    opaque = s.field(pytree=False)
    opaque_explicit = s.field(pytree=False, serialize=True)
    derived_override = s.FieldSpec(
        name="d",
        kind=s.FieldKind.STATIC,
        init=False,
        derived=lambda self: 1,
        serialize=True,
    )

    assert node.should_serialize is True
    assert opaque.should_serialize is False
    assert opaque_explicit.should_serialize is True
    assert derived_override.should_serialize is False


def test_dataclass_name_override(nqx):
    s = nqx.experimental.utils.struct

    @s.dataclass(name="RenamedConfig")
    class _Cfg:
        x: int

    assert _Cfg.__name__ == "RenamedConfig"
    assert _Cfg(1).x == 1


def test_match_args_excludes_kw_only_fields(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        a: int
        b: int = s.field(default=2, kw_only=True)
        c: int = 3

    assert _Model.__match_args__ == ("a", "c")


def test_struct_equality_returns_false_for_different_types(nqx):
    s = nqx.experimental.utils.struct

    class _A(s.Struct):
        x: int

    class _B(s.Struct):
        x: int

    assert (_A(1) == _B(1)) is False


def test_replace_reexecutes_post_init_and_derivation(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int
        y: int = s.field(static=True, init=False, derived=lambda self: self.x * 2)

        def __post_init__(self):
            self.x = self.x + 1

    m = _Model(1)
    assert (m.x, m.y) == (2, 4)

    m2 = m.replace(x=3)
    assert (m2.x, m2.y) == (4, 8)


def test_rederive_revalidates_static_constraints(nqx):
    s = nqx.experimental.utils.struct
    ValidationError = nqx.utils.errors.ValidationError

    class _Model(s.Struct):
        x: int
        meta: tuple[int, ...] = s.field(static=True, default=(1, 2))

    m = _Model(1)
    object.__setattr__(m, "meta", [1, 2])
    with pytest.raises(ValidationError, match="must be hashable"):
        m.rederive()


def test_replace_restores_non_init_fields_through_converter(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int
        y: int = s.field(init=False, default=1, converter=lambda value: int(value))

    m = _Model(1)
    m2 = m.replace(x=2, y="7")
    assert m2.x == 2
    assert m2.y == 7
    assert isinstance(m2.y, int)


def test_struct_unflatten_missing_required_field_raises(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Model(s.Struct):
        x: int

    with pytest.raises(SerializationError, match="Missing field 'x'"):
        _Model.__struct_unflatten__({})


def test_registry_resolve_class_uses_cached_registration(nqx):
    from neuralqx.experimental.utils.struct import registry as reg

    ref = reg.class_ref(_ImportableStructRegistryTarget)
    reg.register_class_ref(_ImportableStructRegistryTarget)
    assert reg.resolve_class(ref) is _ImportableStructRegistryTarget


def test_resolve_pytree_spec_caches_reference_when_only_class_spec_exists(nqx):
    s = nqx.experimental.utils.struct
    from neuralqx.experimental.utils.struct._internals import pytree_registry as preg

    class _Node:
        def __init__(self, data):
            self.data = data

    s.register_pytree_type(
        _Node,
        flatten=lambda obj: ([obj.data], None),
        unflatten=lambda aux, children: _Node(children[0]),
    )
    ref = s.class_ref(_Node)
    by_class = preg.PYTREE_SPECS_BY_CLASS[_Node]
    preg.PYTREE_SPECS_BY_REF.pop(ref, None)

    resolved = s.resolve_pytree_spec(ref)
    assert resolved is by_class
    assert preg.PYTREE_SPECS_BY_REF[ref] is by_class


def test_resolve_pytree_spec_requires_callable_unflatten_for_autodiscovery(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError
    ref = s.class_ref(_HalfAutoTree)

    with pytest.raises(SerializationError, match="No pytree registration found"):
        s.resolve_pytree_spec(ref)


def test_register_pytree_type_reregistration_updates_local_spec(nqx):
    s = nqx.experimental.utils.struct

    class _Node:
        def __init__(self, data):
            self.data = data

    s.register_pytree_type(
        _Node,
        flatten=lambda obj: ([obj.data], "v1"),
        unflatten=lambda aux, children: _Node(children[0]),
    )
    s.register_pytree_type(
        _Node,
        flatten=lambda obj: ([obj.data * 2], "v2"),
        unflatten=lambda aux, children: _Node(children[0] // 2),
    )

    spec = s.resolve_pytree_spec(s.class_ref(_Node))
    children, aux = spec.flatten(_Node(3))
    assert aux == "v2"
    assert children == [6]


def test_to_python_uses_registered_serializer_for_pytree(nqx, jnp):
    s = nqx.experimental.utils.struct

    class _Node:
        def __init__(self, arr, tag):
            self.arr = arr
            self.tag = tag

    s.register_pytree_type(
        _Node,
        flatten=lambda obj: ([obj.arr], obj.tag),
        unflatten=lambda aux, children: _Node(children[0], aux),
        serializer=lambda obj: {"arr": np.asarray(obj.arr), "tag": obj.tag},
        deserializer=lambda payload: _Node(payload["arr"], payload["tag"]),
    )

    view = s.io.to_python(_Node(jnp.array([2, 3]), "meta"))
    assert view["tag"] == "meta"
    assert isinstance(view["arr"], np.ndarray)
    _assert_array_equal(view["arr"], np.array([2, 3]))


def test_state_dict_roundtrip_for_plain_python_roots(nqx):
    s = nqx.experimental.utils.struct
    root = {
        "path": Path("run"),
        "value": np.int64(7),
        "comp": 1 + 2j,
        "items": [1, 2, 3],
        "frozen": frozenset({4, 5}),
    }

    restored = s.io.from_state_dict(s.io.to_state_dict(root))
    assert restored["path"] == Path("run")
    assert restored["value"] == 7
    assert restored["comp"] == 1 + 2j
    assert restored["items"] == [1, 2, 3]
    assert restored["frozen"] == frozenset({4, 5})


def test_from_state_dict_decodes_list_manifest_payload(nqx):
    payload = {
        "version": 1,
        "manifest": [1, {"__kind__": "complex", "real": 2.0, "imag": -1.5}],
        "arrays": {},
        "array_data": {},
    }
    out = nqx.experimental.utils.struct.io.from_state_dict(payload)
    assert out == [1, 2 - 1.5j]


def test_export_overwrite_true_replaces_existing_directory(nqx, tmp_path_writable):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int

    target = tmp_path_writable / "overwrite_dir"
    target.mkdir()
    stale = target / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    _Model(1).export(target, overwrite=True)
    assert target.joinpath("manifest.json").exists()
    assert target.joinpath("arrays.npz").exists()
    assert not stale.exists()


def test_export_overwrite_true_replaces_existing_zip_file(nqx, tmp_path_writable):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: np.ndarray

    target = tmp_path_writable / "overwrite.zip"
    target.write_text("not-a-zip", encoding="utf-8")
    _Model(np.array([1, 2])).export(target, overwrite=True)

    loaded = _Model.load(target)
    _assert_array_equal(loaded.x, np.array([1, 2]))


def test_io_load_with_expected_cls_guard(nqx, tmp_path_writable):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _A(s.Struct):
        x: int

    class _B(s.Struct):
        x: int

    path = tmp_path_writable / "expected_cls"
    _A(3).export(path)
    with pytest.raises(SerializationError, match="Expected _B"):
        s.io.load(path, expected_cls=_B)


def test_load_directory_missing_arrays_file_raises(nqx, tmp_path_writable):
    broken = tmp_path_writable / "missing_arrays"
    broken.mkdir()
    broken.joinpath("manifest.json").write_text(
        json.dumps({"manifest": None, "arrays": {}}), encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError):
        nqx.experimental.utils.struct.io.load(broken)


def test_load_zip_rejects_malformed_manifest_payload(nqx, tmp_path_writable):
    SerializationError = nqx.utils.errors.SerializationError
    bad_zip = tmp_path_writable / "bad_manifest.zip"

    arrays_bytes = io.BytesIO()
    np.savez_compressed(arrays_bytes)
    arrays_bytes.seek(0)

    with zipfile.ZipFile(bad_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps([1, 2, 3]))
        zf.writestr("arrays.npz", arrays_bytes.read())

    with pytest.raises(SerializationError, match="Malformed manifest payload"):
        nqx.experimental.utils.struct.io.load(bad_zip)


def test_equality_handles_nested_mapping_arrays(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        meta: dict[str, np.ndarray]

    a = _Model({"x": np.array([1, 2])})
    b = _Model({"x": np.array([1, 2])})
    c = _Model({"y": np.array([1, 2])})

    assert a == b
    assert a != c


def test_repr_truncates_large_payloads(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        payload: list[int]

    m = _Model(list(range(200)))
    text = repr(m)
    assert "payload=" in text
    assert "..." in text


def test_export_distributed_master_uses_safe_io_and_barriers(
    nqx, tmp_path_writable, monkeypatch
):
    s = nqx.experimental.utils.struct
    from neuralqx.experimental.utils.struct._internals import io

    class _Model(s.Struct):
        x: np.ndarray

    target = tmp_path_writable / "dist_master_export"
    calls: list[tuple] = []

    class _FakeDist:
        def safe_replicate_for_io(
            self, value, *, replicate_to_all_processes, block_until_ready
        ):
            calls.append(("safe", replicate_to_all_processes, block_until_ready))
            return value

        def barrier(self, name):
            calls.append(("barrier", name))

        def is_global_master(self):
            calls.append(("is_master",))
            return True

        def bcast(self, value, *, root):
            calls.append(("bcast", root))
            return value

    monkeypatch.setattr(io, "_distributed", lambda: _FakeDist())
    out = _Model(np.array([1, 2])).export(target)
    assert out == target
    assert target.joinpath("manifest.json").exists()
    assert target.joinpath("arrays.npz").exists()
    assert ("safe", False, True) in calls
    assert ("barrier", "neuralqx:struct:export:pre") in calls
    assert ("barrier", "neuralqx:struct:export:post") in calls


def test_export_distributed_non_master_uses_broadcast_result(
    nqx, tmp_path_writable, monkeypatch
):
    s = nqx.experimental.utils.struct
    from neuralqx.experimental.utils.struct._internals import io

    class _Model(s.Struct):
        x: int

    target = tmp_path_writable / "dist_non_master_export"

    class _FakeDist:
        def safe_replicate_for_io(
            self, value, *, replicate_to_all_processes, block_until_ready
        ):
            return value

        def barrier(self, name):
            return None

        def is_global_master(self):
            return False

        def bcast(self, value, *, root):
            return {"path": str(target), "error": None}

    monkeypatch.setattr(io, "_distributed", lambda: _FakeDist())
    out = _Model(1).export(target)
    assert out == target
    assert not target.exists()


def test_export_distributed_reraises_master_error(nqx, tmp_path_writable, monkeypatch):
    s = nqx.experimental.utils.struct
    from neuralqx.experimental.utils.struct._internals import io

    class _Model(s.Struct):
        x: int

    class _FakeDist:
        def safe_replicate_for_io(
            self, value, *, replicate_to_all_processes, block_until_ready
        ):
            return value

        def barrier(self, name):
            return None

        def is_global_master(self):
            return False

        def bcast(self, value, *, root):
            return {
                "path": None,
                "error": {"type": "FileExistsError", "message": "already exists"},
            }

    monkeypatch.setattr(io, "_distributed", lambda: _FakeDist())
    with pytest.raises(FileExistsError, match="already exists"):
        _Model(1).export(tmp_path_writable / "ignored")


def test_load_distributed_non_master_uses_broadcast_payload(
    nqx, tmp_path_writable, monkeypatch
):
    s = nqx.experimental.utils.struct
    from neuralqx.experimental.utils.struct._internals import io

    class _Model(s.Struct):
        x: np.ndarray

    state = _Model(np.array([3, 4])).to_state_dict()
    payload = {
        "manifest": {k: v for k, v in state.items() if k != "array_data"},
        "arrays": state["array_data"],
        "error": None,
    }

    class _FakeDist:
        def barrier(self, name):
            return None

        def is_global_master(self):
            return False

        def bcast(self, value, *, root):
            return payload

    monkeypatch.setattr(io, "_distributed", lambda: _FakeDist())
    out = s.io.load(tmp_path_writable / "does-not-need-to-exist-on-non-master")
    assert isinstance(out, _Model)
    _assert_array_equal(out.x, np.array([3, 4]))


def test_load_distributed_reraises_master_error(nqx, tmp_path_writable, monkeypatch):
    from neuralqx.experimental.utils.struct._internals import io

    class _FakeDist:
        def barrier(self, name):
            return None

        def is_global_master(self):
            return False

        def bcast(self, value, *, root):
            return {
                "manifest": None,
                "arrays": None,
                "error": {"type": "FileNotFoundError", "message": "missing"},
            }

    monkeypatch.setattr(io, "_distributed", lambda: _FakeDist())
    with pytest.raises(FileNotFoundError, match="missing"):
        nqx.experimental.utils.struct.io.load(tmp_path_writable / "ignored")


def test_jit_tolerates_non_jittable_derived_when_unused(nqx, jax, jnp):
    s = nqx.experimental.utils.struct

    class _GraphState(s.Struct):
        x: jax.Array
        norm: float = s.field(
            static=True,
            init=False,
            derived=lambda self: float(np.asarray(self.x).sum()),
        )

    @jax.jit
    def _square(state: _GraphState):
        return state.x**2

    out = _square(_GraphState(jnp.array([1.0, 2.0, 3.0])))
    _assert_array_equal(out, np.array([1.0, 4.0, 9.0]))


def test_vmap_tolerates_non_jittable_derived_when_unused(nqx, jax, jnp):
    s = nqx.experimental.utils.struct

    class _GraphState(s.Struct):
        x: jax.Array
        norm: float = s.field(
            static=True,
            init=False,
            derived=lambda self: float(np.asarray(self.x).sum()),
        )

    def _square(state: _GraphState):
        return state.x**2

    out = jax.vmap(_square)(_GraphState(jnp.array([[1.0, 2.0], [3.0, 4.0]])))
    _assert_array_equal(out, np.array([[1.0, 4.0], [9.0, 16.0]]))


def test_jit_allows_accessing_derived_field_when_derived_uses_jax_ops(nqx, jax, jnp):
    s = nqx.experimental.utils.struct

    class _GraphState(s.Struct):
        x: jax.Array
        norm: object = s.field(
            pytree=False,
            init=False,
            derived=lambda self: jnp.sum(self.x),
        )

    @jax.jit
    def _norm(state: _GraphState):
        return state.norm

    out = _norm(_GraphState(jnp.array([1.0, 2.0, 3.0])))
    _assert_array_equal(out, np.array(6.0))


def test_replace_with_non_init_field_runs_post_init_exactly_once(nqx):
    """Regression: _restore_non_init must not re-run __post_init__."""
    s = nqx.experimental.utils.struct
    call_count = {"n": 0}

    class _Model(s.Struct):
        x: int
        # Non-init field forces the _restore_non_init path inside replace().
        label: str = s.field(init=False, default="default")

        def __post_init__(self):
            call_count["n"] += 1

    m = _Model(1)
    assert call_count["n"] == 1

    call_count["n"] = 0
    m.replace(x=2)
    assert (
        call_count["n"] == 1
    ), "__post_init__ must run exactly once per replace(), not twice"


def test_replace_with_non_init_field_and_converter_runs_post_init_exactly_once(nqx):
    """Converter on a non-init field must not cause a second __post_init__."""
    s = nqx.experimental.utils.struct
    call_count = {"n": 0}

    class _Model(s.Struct):
        x: int
        tag: str = s.field(init=False, default="0", converter=str)

        def __post_init__(self):
            call_count["n"] += 1

    m = _Model(1)
    call_count["n"] = 0
    m2 = m.replace(x=5, tag=99)
    assert call_count["n"] == 1
    assert m2.tag == "99"


def test_multi_level_inheritance_merges_fields_in_mro_order(nqx):
    s = nqx.experimental.utils.struct

    class _Base(s.Struct):
        x: int

    class _Mid(_Base):
        y: int = s.field(static=True, default=10)

    class _Leaf(_Mid):
        z: int = 0

    leaf = _Leaf(1)
    assert leaf.x == 1
    assert leaf.y == 10
    assert leaf.z == 0
    assert tuple(s.fields(_Leaf).keys()) == ("x", "y", "z")
    assert s.node_fields(_Leaf) == ("x", "z")
    assert s.static_fields(_Leaf) == ("y",)


def test_multi_level_inheritance_replace_and_pytree(nqx, jax, jnp):
    s = nqx.experimental.utils.struct

    class _Base(s.Struct):
        x: jax.Array

    class _Leaf(_Base):
        scale: float = s.field(static=True, default=1.0)

    leaf = _Leaf(jnp.array([1.0, 2.0]))
    leaf2 = leaf.replace(scale=2.0)
    assert leaf2.scale == 2.0

    mapped = jax.tree_util.tree_map(lambda v: v * 2, leaf)
    _assert_array_equal(mapped.x, jnp.array([2.0, 4.0]))
    assert mapped.scale == 1.0


def test_multi_level_inheritance_serialization_roundtrip(nqx):
    s = nqx.experimental.utils.struct

    class _Base(s.Struct):
        x: np.ndarray

    class _Leaf(_Base):
        tag: str = s.field(static=True, default="v1")

    leaf = _Leaf(np.array([3, 4, 5]), "v2")
    restored = _Leaf.from_state_dict(leaf.to_state_dict())
    _assert_array_equal(restored.x, leaf.x)
    assert restored.tag == "v2"


def test_struct_abc_meta_enforces_abstract_methods(nqx):
    s = nqx.experimental.utils.struct
    import abc

    class _AbstractModel(s.Struct, metaclass=s.StructABCMeta):
        x: int

        @abc.abstractmethod
        def compute(self) -> int: ...

    with pytest.raises(TypeError, match="abstract"):
        _AbstractModel(1)


def test_struct_abc_meta_allows_concrete_subclass(nqx):
    s = nqx.experimental.utils.struct
    import abc

    class _AbstractModel(s.Struct, metaclass=s.StructABCMeta):
        x: int

        @abc.abstractmethod
        def compute(self) -> int: ...

    class _Concrete(_AbstractModel):
        def compute(self) -> int:
            return self.x * 2

    c = _Concrete(5)
    assert c.compute() == 10
    assert c.x == 5


def test_register_class_replaces_custom_init_with_struct_generated_one(nqx):
    """register_class always generates a new __init__, raw_cls.__init__ is discarded."""
    s = nqx.experimental.utils.struct

    class _Raw:
        x: int
        y: int

        def __init__(self):
            raise RuntimeError("This custom __init__ must not be called")

    _Registered = s.register_class(_Raw)
    # Should construct fine with the generated Struct __init__.
    obj = _Registered(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_normalize_validators_wraps_generator_as_single_callable_and_fails(nqx):
    """A generator passed as validator is wrapped as one callable and fails at call time."""
    s = nqx.experimental.utils.struct

    def _v1(value):
        return True

    def _v2(value):
        return True

    # A generator expression is not a Sequence, so it gets treated as a single
    # callable. When called with (value,) it raises TypeError because generators
    # are not directly callable.
    class _Model(s.Struct):
        x: int = s.field(validator=(v for v in [_v1, _v2]))

    with pytest.raises(TypeError):
        _Model(1)


def test_rederive_recomputes_opaque_derived_field(nqx):
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        data: list[int]
        # Derived opaque field (not static).
        summary: object = s.field(
            pytree=False,
            init=False,
            derived=lambda self: tuple(self.data),
        )

    m = _Model([1, 2])
    assert m.summary == (1, 2)

    m.data.append(3)
    assert m.summary == (1, 2)  # stale before rederive

    m.rederive()
    assert m.summary == (1, 2, 3)


def test_derived_field_cycle_raises_runtime_error(nqx):
    """A derived field whose callable reads itself must raise RuntimeError, not recurse."""
    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        x: int
        # The derived callable reads `self.cycle`. i.e., its own output.
        cycle: int = s.field(
            static=True,
            init=False,
            derived=lambda self: self.cycle + 1,
        )

    m = object.__new__(_Model)
    object.__setattr__(m, "_Model__struct_initializing__", False)

    from neuralqx.experimental.utils.struct._internals.runtime import (
        DeferredDerivedValue,
    )

    object.__setattr__(m, "__struct_initializing__", False)
    object.__setattr__(m, "x", 1)
    object.__setattr__(m, "cycle", DeferredDerivedValue("cycle"))

    with pytest.raises(RuntimeError, match="Cycle detected"):
        _ = m.cycle


def test_from_state_dict_expected_cls_rejects_list_manifest(nqx):
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Model(s.Struct):
        x: int

    # Manually craft a state dict whose manifest is a list.
    payload = {
        "version": 1,
        "manifest": [1, 2, 3],
        "arrays": {},
        "array_data": {},
    }
    with pytest.raises(SerializationError, match="_Model"):
        s.io.from_state_dict(payload, expected_cls=_Model)


def test_serialize_bytes_field_raises(nqx):
    """bytes values are not serialisable, users must convert or exclude them."""
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Model(s.Struct):
        data: object

    with pytest.raises(SerializationError, match="Cannot serialize"):
        _Model(b"raw bytes").to_state_dict()


def test_serialize_range_raises(nqx):
    """range objects are not serialisable."""
    s = nqx.experimental.utils.struct
    SerializationError = nqx.utils.errors.SerializationError

    class _Model(s.Struct):
        r: object

    with pytest.raises(SerializationError, match="Cannot serialize"):
        _Model(range(10)).to_state_dict()


def test_ordered_dict_loses_subtype_on_roundtrip(nqx):
    """Known limitation: OrderedDict round-trips as plain dict."""
    from collections import OrderedDict

    s = nqx.experimental.utils.struct

    class _Model(s.Struct):
        mapping: object

    m = _Model(OrderedDict([("a", 1), ("b", 2)]))
    restored = _Model.from_state_dict(m.to_state_dict())
    # Values are preserved, but the subtype is lost.
    assert dict(restored.mapping) == {"a": 1, "b": 2}
    assert type(restored.mapping) is dict


def test_io_module_is_in_struct_all(nqx):
    import neuralqx.experimental.utils.struct as struct_pkg

    assert "io" in struct_pkg.__all__
    assert struct_pkg.io is not None


def test_io_module_accessible_via_attribute(nqx):
    s = nqx.experimental.utils.struct

    assert hasattr(s, "io")
    assert callable(s.io.to_state_dict)
    assert callable(s.io.from_state_dict)
    assert callable(s.io.export)
    assert callable(s.io.load)
    assert callable(s.io.to_python)


def test_from_state_dict_rejects_version_zero(nqx):
    SerializationError = nqx.utils.errors.SerializationError

    with pytest.raises(SerializationError, match="Unsupported state dict version"):
        nqx.experimental.utils.struct.io.from_state_dict(
            {"version": 0, "manifest": None, "arrays": {}, "array_data": {}}
        )


def test_from_state_dict_rejects_non_integer_version(nqx):
    SerializationError = nqx.utils.errors.SerializationError

    with pytest.raises(SerializationError, match="Unsupported state dict version"):
        nqx.experimental.utils.struct.io.from_state_dict(
            {"version": "1", "manifest": None, "arrays": {}, "array_data": {}}
        )


def test_export_directory_is_absent_if_write_fails(nqx, tmp_path_writable, monkeypatch):
    """If writing fails, the destination directory must not exist in a partial state."""
    s = nqx.experimental.utils.struct
    from neuralqx.experimental.utils.struct._internals import io as _internal_io

    class _Model(s.Struct):
        x: int

    target = tmp_path_writable / "atomic_export"

    original_write_bundle = _internal_io.write_bundle

    def _failing_write(path, manifest, arrays):
        # Write the manifest but then raise before writing arrays.
        original_write_bundle(path, manifest, arrays)
        raise OSError("Simulated mid-write failure")

    monkeypatch.setattr(_internal_io, "write_bundle", _failing_write)

    with pytest.raises(OSError, match="Simulated mid-write failure"):
        _Model(1).export(target)

    # The destination must not exist, the temp directory was cleaned up.
    assert not target.exists()
