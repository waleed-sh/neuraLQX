---
title: Fields and static data
---

# Fields and static data

Field declarations determine what JAX can transform, what remains fixed during tracing, and what serialisation retains. These decisions are related, but they are not the same decision. A coefficient can remain a dynamic input even when you do not differentiate it, and static metadata can still belong in a saved state.

Use [`field`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.field) to specify these behaviours. An annotated attribute with no explicit declaration is an ordinary node field. Its annotation describes the intended Python type, but does not itself convert the supplied value or validate its numerical meaning.

## Numerical data, static metadata, and opaque references

[`FieldKind`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.FieldKind) distinguishes three roles.

| Declaration | Pytree behaviour | Default persistence |
| --- | --- | --- |
| An annotation alone, or `field()` | Dynamic child traversed recursively | Included |
| `field(static=True)` | Auxiliary metadata compared by value | Included |
| `field(pytree=False)` | Auxiliary reference compared by Python identity | Omitted |

For a node field, use arrays, numerical scalars, or nested pytrees suitable for the transformations you intend. JAX traverses a dictionary or tuple in that field rather than treating the entire field as one indivisible leaf. Strings in ordinary node fields can be inspected by tree utilities, but cannot be passed to a numerical `jit` call as array data.

For a static field, use small immutable descriptions such as an algorithm mode, a fixed axis, or a tuple of dimensions. Static values must be hashable and array-free. Their equality participates in the tree definition and can affect compilation reuse. A value is not static merely because you do not plan to optimise it.

For an opaque field, use a Python object whose reference must accompany the container, such as a runtime interface. The value is excluded from numerical leaves, but its identity remains in auxiliary data. Replacing it with a different object can change tree identity even when the two objects have similar contents. Mutating its contents in place does not tell JAX that anything changed.

This example exposes the three field groups and verifies that a tree transformation changes only the numerical values.

```python
import jax
import jax.numpy as jnp
from neuralqx.utils.struct import Struct, field

class AnalysisState(Struct):
    values: jax.Array
    reduction: str = field(static=True, default="mean")
    context: object = field(pytree=False, default_factory=object, compare=False)

state = AnalysisState(jnp.array([1.0, 2.0]))
scaled = jax.tree.map(lambda value: 2.0 * value, state)

print(AnalysisState.node_fields())
print(AnalysisState.static_fields())
print(AnalysisState.opaque_fields())
print(scaled.values.tolist())
print(scaled.context is state.context)
```

```text
('values',)
('reduction',)
('context',)
[2.0, 4.0]
True
```

An opaque object is not a way to hide changing numerical input from JAX. If compiled code reads an array through an opaque reference, that array is not an explicit dynamic argument. Changing it can leave a compiled computation using assumptions made during tracing. Keep evolving arrays in node fields and pass runtime objects only where their role is understood.

## Defaults and constructor arguments

Use `default=` for a fixed value and `default_factory=` for a zero-argument function called when a value is omitted. A factory is appropriate for arrays or containers that should be created separately for each instance. The two options cannot be combined.

Required positional fields must precede positional fields with defaults, including inherited fields. `kw_only=True` makes a field keyword-only and is useful when a subclass adds a required field after inherited defaults. `init=False` removes a field from the generated constructor. A non-derived field with `init=False` still needs a default or factory.

The convenience [`dataclass`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.dataclass) decorator provides the same struct processing for an annotated class. It is neuraLQX's decorator, distinct from Python's standard dataclass decorator. For new components, directly subclassing `Struct` makes the relationship explicit.

## Convert representations and validate meaning

A converter normalises an assigned value. A validator checks a field after construction has established the object's values and derived data. Both accept either `hook(value)` or `hook(self, value)`. A validator can raise an exception or return the Python value `False` to report failure. It can also return `None` on success.

Here, the input accepts a sequence of axes and stores it as an immutable tuple. The validator checks a fixed chunk size. These are metadata checks, so they do not need to read an array value from a device.

```python
from neuralqx.utils.struct import Struct, field

def positive_integer(value):
    if type(value) is not int or value <= 0:
        raise ValueError("chunk_size must be a positive Python integer")

class ReductionSettings(Struct):
    axes: tuple = field(static=True, converter=lambda value: tuple(value))
    chunk_size: int = field(
        static=True,
        default=32,
        kw_only=True,
        validator=positive_integer,
    )

settings = ReductionSettings([0, 2], chunk_size=16)
updated = settings.replace(chunk_size=64)
print(settings.axes)
print(settings.chunk_size, updated.chunk_size)
```

```text
(0, 2)
16 64
```

You can provide a sequence of validators when several independent checks are useful. Use explicit exceptions for numerical checks outside JAX. Returning a NumPy or JAX boolean is not the same as returning Python `False`, and converting a traced array predicate into a Python boolean is not valid inside a compiled transformation.

Converters also run during reconstruction. A converter that always calls `numpy.asarray`, reads an array value, or assumes a fully realised input can fail when JAX supplies tracers or structural placeholders. Normalise user input before entering numerical transformations when that conversion cannot support those cases. The [JAX behaviour guide](pytrees.md#reconstruction-is-not-an-ordinary-constructor-call) explains this boundary.

## Derived fields

A derived field records a value computed from other fields. It must have `init=False`, must be static or opaque, and cannot specify a default or be explicitly serialised. Its derivation accepts either no arguments or the instance as its only argument.

Use a static derivation for metadata determined by other metadata. The example below records the number of integer charges in a symmetric range. Its source is the static cutoff, so no device value needs to become a Python constant.

```python
from neuralqx.utils.struct import Struct, field

class ChargeRange(Struct):
    cutoff: int = field(static=True)
    local_size: int = field(
        static=True,
        init=False,
        derived=lambda self: 2 * self.cutoff + 1,
    )

    def __post_init__(self):
        if type(self.cutoff) is not int or self.cutoff < 0:
            raise ValueError("cutoff must be a nonnegative Python integer")

small = ChargeRange(cutoff=1)
large = small.replace(cutoff=3)
restored = ChargeRange.from_state_dict(large.to_state_dict())
print(small.local_size, large.local_size, restored.local_size)
print(ChargeRange.derived_fields())
```

```text
3 7 7
('local_size',)
```

This small metadata example is not a replacement for the supplied [U(1) spaces](../../concepts/hilbert/u1.md). It illustrates how a secondary description stays consistent when the source field changes.

Derived fields are recomputed on construction, replacement, and reconstruction. You cannot provide them to the generated constructor or update them directly through `replace`. Update their dependencies instead. Derivations run in field order, so a dependency on another derived field must follow that field's declaration and must not introduce a cycle.

A derived field is not an independent differentiable leaf. For an array-valued function of current numerical data, a property or numerical method is often clearer. If a derived opaque value is appropriate, its computation still needs to obey the transformation rules wherever it is accessed. Deferring a host-only derivation during tracing does not make that derivation executable on a tracer.

## Construction, replacement, and freezing

The generated constructor assigns non-derived values through their converters, computes derived fields, and calls `__post_init__` when one is defined. It then recomputes derived fields if the hook ran, validates field values and static constraints, and freezes attribute assignment. Derivation functions should therefore be deterministic and free of effects such as consuming random state or opening a file.

[`replace`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.Struct.replace) creates a new instance using its constructor, then restores non-init fields as needed. It reruns the relevant conversion, derivation, and validation. The post-init hook runs once through construction. Existing non-init values are preserved unless explicitly replaced, while derived fields are recalculated.

Replacement is not a deep copy. Unchanged nested containers and opaque objects can still be shared with the original. The advanced `rederive()` method recomputes derived values in place, but ordinary numerical code should use replacement rather than depend on mutation of an object hidden inside a field.

Pytree unflattening and deserialisation use a separate reconstruction path that does not call the ordinary constructor or `__post_init__`. Put essential reconstruction rules in field converters, validators, or derivations where appropriate. Do not rely on a post-init side effect to recreate a required attribute after loading.

## Inheritance and the declared schema

Fields are collected from base classes and then combined with the subclass's declarations. A redeclaration replaces the inherited field specification. In particular, re-annotating an inherited static field without repeating `field(static=True)` makes the new declaration an ordinary node field. Repeat the intended field behaviour when overriding a declaration.

Keep all state that must survive transformation or loading in declared fields. An attribute created only inside a custom method is not automatically discovered by the pytree or serialisation machinery. Prefer the generated constructor and `__post_init__` for ordinary subclasses. A custom constructor must deliberately participate in the struct initialisation lifecycle.

[`StructABCMeta`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.StructABCMeta) combines struct processing with abstract-method enforcement. Package interfaces such as the operator and sampler bases already use it. Their concrete subclasses inherit that arrangement.

## Inspection and persistence are separate controls

`Class.fields()` returns the ordered mapping of names to [`FieldSpec`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.FieldSpec). The corresponding `node_fields()`, `static_fields()`, `opaque_fields()`, and `derived_fields()` methods expose each group. They are useful when checking a new component's schema.

`repr=False` hides a field from the generated representation. `compare=False` excludes it from the struct's Python equality method. Neither option changes JAX tree identity or persistence. In particular, `compare=False` on an opaque field does not remove its identity from pytree metadata.

`serialize=False` omits a field from saved state without changing its pytree role. A field omitted from persistence needs a default, factory, or derivation that makes reconstruction possible. Opaque fields are omitted by default, and `serialize=True` can include one only if its value has a supported encoding. Derived fields are always recomputed. Continue with [serialisation](serialisation.md) for worked round trips and file storage.
