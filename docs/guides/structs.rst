.. _structs_guide:

=============================================
Structs and data containers
=============================================

.. warning::

   **Experimental feature.**

   The neuraLQX immutable Struct is not part of the stable neuraLQX public API.
   Its internals, functionality and behaviour may change without prior warning or
   deprecation period. Pin your neuraLQX version if you depend on it in research scripts.

   See :ref:`experimental` for the full experimental-API policy.

``neuralqx.experimental.utils.struct`` is the foundation for **immutable, JAX-native data
containers** in neuraLQX.  It provides a single base class :class:`~neuralqx.experimental.utils.struct.Struct`
whose subclasses are automatically:

* registered as JAX **pytrees** with correct leaf/auxiliary-data split,
* **frozen** after construction so they can be safely used as JAX static
  arguments and in functional code,
* equipped with a stable **serialisation** contract (in-memory state dicts,
  directory bundles, ``.zip`` archives),
* identified by a **portable class reference** so serialised payloads can be
  reconstructed across processes.

The field system that drives all of this behaviour is declared with a single
:func:`~neuralqx.experimental.utils.struct.field` helper that mirrors :func:`dataclasses.field`
but adds JAX-specific controls.

.. code-block:: python

    import neuralqx.experimental.utils.struct as s

    class TrainState(s.Struct):
        params: object  # JAX array tree (node field)
        step: int = s.field(static=True, default=0)  # static metadata
        tag: str = s.field(static=True, default="run")

    state = TrainState(params={"w": jnp.ones(4)})
    state2 = state.replace(step=1)  # new instance, original unchanged

    payload = state.to_state_dict()  # portable mapping
    state3 = TrainState.from_state_dict(payload)

Import path:

.. code-block:: python

    import neuralqx.experimental.utils.struct as s
    # or pick individual names:
    from neuralqx.experimental.utils.struct import Struct, field, FieldKind, register_class


.. contents:: On this page
   :local:
   :depth: 2


--------------------------------------------------------------------
Why Struct?
--------------------------------------------------------------------

Plain Python objects do not compose well with JAX. JAX transformations
(``jit``, ``grad``, ``vmap``) can only trace through values that are
registered as pytrees. If you pass an arbitrary object as an argument to a
jitted function, JAX sees it as a static constant, changes to its fields are
invisible to the tracer, and you lose the ability to differentiate through it.

Python's standard :mod:`dataclasses` module does not know about this.
:func:`dataclasses.dataclass` gives you a nice constructor and ``__repr__``,
but it does not register anything with JAX.

``Struct`` closes this gap:

* **Every ``Struct`` subclass is a JAX pytree** registered automatically at
  class-definition time. JAX can trace through its *node fields* (the pytree
  leaves) while keeping *static fields* as compile-time constants.
* **Instances are immutable** after construction. Attempting to set an
  attribute raises :class:`~neuralqx.utils.errors.FrozenStructError`. This
  makes ``Struct`` instances safe to use as pytree auxiliary data, as ``jit``
  static arguments, and as dictionary keys.
* **Serialisation is built in**.  ``to_state_dict`` / ``from_state_dict`` and
  ``export`` / ``load`` work automatically for any ``Struct``, including nested
  ones, without extra configuration.

The field kinds (node, static, opaque) are the key design decision. Getting
them right is what makes a ``Struct`` behave correctly under JAX
transformations.


--------------------------------------------------------------------
Field kinds
--------------------------------------------------------------------

Every field in a ``Struct`` has a *kind* that controls how it participates in
JAX pytree traversal. The three kinds are expressed by
:class:`~neuralqx.experimental.utils.struct.FieldKind`:

.. list-table::
   :header-rows: 1
   :widths: 12 28 60

   * - Kind
     - ``FieldKind`` value
     - What it means
   * - **Node**
     - ``FieldKind.NODE``
     - The field is a **pytree leaf**. JAX traces through it. Gradients can
       flow through it, ``vmap`` can batch it, and ``jit`` will re-trace if
       its *shape or dtype* changes. This is the default for unannotated
       fields.
   * - **Static**
     - ``FieldKind.STATIC``
     - The field becomes **pytree auxiliary data** (the ``aux_data`` argument
       in JAX's ``tree_unflatten``). It is treated as a compile-time constant
       — JAX re-traces any jitted function whenever a static field value
       changes.  Static fields must be **hashable** and must not contain
       arrays.
   * - **Opaque**
     - ``FieldKind.OPAQUE``
     - The field is **excluded from the pytree entirely**. JAX does not see
       it. Use this for runtime handles, caches, tokens, or any
       process-local object that should not be traced or serialised.

Choosing correctly:

* Use **node** for anything JAX should differentiate through: weight arrays,
  configuration tensors, floating-point scalars that change across calls.
* Use **static** for things that shape your computation but are fixed during a
  training run: hyperparameters, network architecture choices, string labels,
  integer sizes.  Every unique combination of static field values triggers a
  separate ``jit`` compilation.
* Use **opaque** for runtime-only side-data that would confuse JAX: file
  handles, logger objects, reference caches, non-serialisable callbacks.

.. code-block:: python

    class Config(s.Struct):
        weights: object  # NODE: JAX traces this
        n_layers: int = s.field(static=True)  # STATIC: re-traces on change
        cache: object = s.field(pytree=False, default_factory=dict)  # OPAQUE: invisible to JAX


--------------------------------------------------------------------
The ``field()`` declaration helper
--------------------------------------------------------------------

:func:`~neuralqx.experimental.utils.struct.field` is the single function used to control
field behaviour. It returns a :class:`~neuralqx.experimental.utils.struct.FieldSpec` that
the metaclass picks up at class-definition time.

Signature (all arguments are keyword-only):

.. code-block:: python

    s.field(
        *,
        static=False,             # True -> FieldKind.STATIC
        pytree=True,              # False -> FieldKind.OPAQUE
        default=MISSING,          # concrete default value
        default_factory=MISSING,  # zero-arg callable for mutable defaults
        init=True,                # include in generated constructor
        repr=True,                # include in __repr__
        compare=True,             # include in __eq__
        serialize=None,           # override serialisation (None = inferred)
        kw_only=False,            # force keyword-only in constructor
        doc=None,                 # per-field docstring
        metadata=None,            # free-form mapping for tooling
        converter=None,           # normalisation hook (see below)
        validator=None,           # validation hook(s) (see below)
        derived=None,             # computed-field function (see below)
    )

Common patterns:

.. code-block:: python

    class State(s.Struct):
        # plain node field, no annotation needed
        params: object

        # static field with a default
        name: str = s.field(static=True, default="run-1")

        # opaque field, invisible to JAX, not serialised by default
        logger: object = s.field(pytree=False, default_factory=object)

        # mutable container default, avoids shared-state bugs
        history: list = s.field(default_factory=list)

        # keyword-only in constructor
        seed: int = s.field(static=True, kw_only=True, default=0)

        # exclude from repr and equality
        debug_token: str = s.field(
            static=True, repr=False, compare=False, default=""
        )

**Default rules:**

* ``default`` and ``default_factory`` are mutually exclusive.
* Fields without either must receive a value in the constructor.
* ``default_factory`` is called freshly each time a default is needed —
  this is the correct way to declare mutable defaults (lists, dicts, etc.).


--------------------------------------------------------------------
Converter and validator hooks
--------------------------------------------------------------------

Converters
==========

A *converter* is called on the raw value **every time the field is assigned**
in the constructor, in ``replace``, and during deserialisation.  Use it to
normalise input types:

.. code-block:: python

    class Box(s.Struct):
        count: int = s.field(converter=int)  # coerce str to int on input
        label: str = s.field(converter=str.strip)  # strip whitespace

    b = Box(count="3", label="  hello  ")
    b.count  # 3  (int)
    b.label  # "hello"

Converters accept either ``converter(value)`` or ``converter(self, value)``.
The two-argument form lets the converter inspect other already-assigned fields:

.. code-block:: python

    def _clamp(self, value):
        return max(0, min(value, self.maximum))

    class Bounded(s.Struct):
        maximum: int
        value: int = s.field(converter=_clamp)

Validators
==========

A *validator* is called after the converter (if any) with the normalised
value. If it returns ``False``, a :class:`~neuralqx.utils.errors.ValidationError`
is raised. If it raises an exception directly, that exception propagates.

.. code-block:: python

    def _positive(value):
        return value > 0

    class Rate(s.Struct):
        lr: float = s.field(validator=_positive)

    Rate(lr=0.01)  # fine
    Rate(lr=-1.0)  # raises ValidationError

Pass a list to apply multiple validators in order:

.. code-block:: python

    def _finite(value):
        import math
        return math.isfinite(value)

    class Rate(s.Struct):
        lr: float = s.field(validator=[_positive, _finite])

Like converters, validators may accept ``validator(value)`` or
``validator(self, value)``.


--------------------------------------------------------------------
Derived fields
--------------------------------------------------------------------

A *derived field* is computed automatically from other fields. It is never
part of the constructor, never serialised (the value is recomputed instead),
and must be either static or opaque (not a node field).

Declare it with ``init=False`` and a ``derived`` callable:

.. code-block:: python

    class Dataset(s.Struct):
        samples: list
        n: int = s.field(
            static=True,
            init=False,
            derived=lambda self: len(self.samples),
        )

    ds = Dataset(samples=[1, 2, 3])
    ds.n  # 3, computed automatically

The derived callable receives either no arguments (``derived()``) or the
instance (``derived(self)``). It is called:

* after initial construction,
* after ``replace()``,
* after ``from_state_dict()`` / ``load()``,
* explicitly when you call ``rederive()``.

**Derived fields run twice during construction:** once before ``__post_init__``
and once after (in case ``__post_init__`` modifies source fields).


--------------------------------------------------------------------
Construction lifecycle
--------------------------------------------------------------------

When you call ``MyStruct(...)`` the following steps happen in order:

1. **Assign user/default values** to all non-derived, non-init fields and all
   init fields from the provided keyword arguments.  Converters are applied at
   this point.
2. **Compute derived fields** for the first time.
3. **Run ``__post_init__``** if defined. You may freely mutate fields inside
   ``__post_init__``, the instance is not frozen yet.
4. **Recompute derived fields** again (in case ``__post_init__`` changed
   source fields).
5. **Validate** static constraints (static fields must be hashable and
   array-free) and field validators.
6. **Freeze** the instance. Any subsequent ``setattr`` or ``delattr`` raises
   :class:`~neuralqx.utils.errors.FrozenStructError`.

.. code-block:: python

    class Model(s.Struct):
        x: int
        y: int = s.field(static=True, init=False, derived=lambda self: self.x * 2)

        def __post_init__(self):
            # called at step 3, can still mutate
            self.x = self.x + 1

    m = Model(x=2)
    m.x  # 3  (post_init incremented it)
    m.y  # 6  (derived after post_init)


--------------------------------------------------------------------
``replace()`` and ``rederive()``
--------------------------------------------------------------------

replace
=======

Because ``Struct`` instances are frozen, the way to "update" one is to create
a new instance with selected fields changed:

.. code-block:: python

    state2 = state.replace(step=state.step + 1, tag="run-2")

``replace`` runs the full lifecycle again, converters, ``__post_init__``,
derived fields, validators, freeze. You cannot replace a derived field
directly, update its source fields instead.

.. code-block:: python

    # Wrong, raises TypeError:
    ds.replace(n=10)

    # Correct, update the source and n is recomputed:
    ds.replace(samples=[1, 2, 3, 4])

rederive
========

Occasionally a node field holds a **mutable** container (a list, a dict)
that has changed in place. In this case derived fields that depend on it
are stale.  ``rederive()`` recomputes all derived fields without rebuilding
the whole instance:

.. code-block:: python

    class Bag(s.Struct):
        items: list
        size: int = s.field(static=True, init=False, derived=lambda self: len(self.items))

    bag = Bag(items=[1, 2])
    bag.items.append(3)  # mutate in place
    bag.size  # 2  (stale!)

    bag.rederive()
    bag.size  # 3

.. note::

   In most JAX code you would avoid in-place mutation entirely and use
   ``replace`` instead.  ``rederive`` exists for the rare cases where a
   mutable container is intentional (e.g. a growing replay buffer).


--------------------------------------------------------------------
Introspection helpers
--------------------------------------------------------------------

Both class-level and module-level forms exist for convenience:

.. code-block:: python

    # class methods
    State.fields()           # OrderedDict of name -> FieldSpec
    State.node_fields()      # ("params",)
    State.static_fields()    # ("step", "tag")
    State.opaque_fields()    # ()
    State.derived_fields()   # ()

    # module-level wrappers (same result)
    s.fields(State)
    s.node_fields(State)
    s.static_fields(State)
    s.opaque_fields(State)
    s.derived_fields(State)

Inspecting a :class:`~neuralqx.experimental.utils.struct.FieldSpec`:

.. code-block:: python

    spec = State.fields()["step"]
    spec.name              # "step"
    spec.kind              # FieldKind.STATIC
    spec.default           # 0
    spec.has_default       # True
    spec.is_derived        # False
    spec.should_serialize  # True

Counting pytree leaves:

.. code-block:: python

    state.tree_size()   # == len(jax.tree_util.tree_leaves(state))

Converting to a plain dict (for debugging/logging, not persistence):

.. code-block:: python

    state.to_dict()                        # shallow, values are raw Python objects
    state.to_dict(recursive=True)          # recurse through nested Structs
    state.to_dict(include_opaque=False)    # skip opaque fields


--------------------------------------------------------------------
Serialisation
--------------------------------------------------------------------

``Struct`` provides four serialisation methods. All handle nested ``Struct``
values, registered pytrees, and NumPy/JAX arrays automatically.

to_state_dict / from_state_dict
================================

Produces and consumes a portable **in-memory Python mapping**:

.. code-block:: python

    payload = state.to_state_dict()
    # {
    #   "version": 1,
    #   "manifest": {...},   # JSON-safe structural encoding
    #   "arrays": {...},     # shape/dtype metadata
    #   "array_data": {...}, # in-memory NumPy arrays
    # }

    restored = TrainState.from_state_dict(payload)

The payload is suitable for:

* sending over the network (after serialising ``array_data`` separately),
* passing to logging systems,
* checkpointing in memory without writing to disk.

Derived fields are **not** stored, they are recomputed during
``from_state_dict``. Opaque fields are not stored by default.

export / load
=============

Writes a **two-file bundle** to disk and reads it back:

.. code-block:: python

    # Write to a directory:
    state.export("checkpoints/step_100")
    # Creates:
    #   checkpoints/step_100/manifest.json
    #   checkpoints/step_100/arrays.npz

    # Or write a zip archive:
    state.export("checkpoints/step_100.zip")

    # Load back (class is inferred from the manifest):
    restored = TrainState.load("checkpoints/step_100")

    # Optionally enforce the expected class:
    restored = TrainState.load("checkpoints/step_100", load_cls=...)

The ``manifest.json`` contains the structural encoding including the class
reference (``"module:qualname"`` string). The ``arrays.npz`` contains all
array payloads compressed with NumPy's compressed format.  Both files together
are a self-contained checkpoint.

``export`` raises ``FileExistsError`` if the target already exists unless you
pass ``overwrite=True``.

Serialisation policy per field
================================

By default:

* **Node fields** are serialised.
* **Static fields** are serialised.
* **Opaque fields** are **not** serialised.
* **Derived fields** are **never** serialised (recomputed on load).

You can override the policy for any individual field with ``serialize=True``
or ``serialize=False``:

.. code-block:: python

    class State(s.Struct):
        params: object
        cache: object = s.field(pytree=False, serialize=True)  # opaque but saved
        debug: str = s.field(static=True, serialize=False)  # static but skipped


--------------------------------------------------------------------
Registering non-Struct classes
--------------------------------------------------------------------

Not every class needs to (or should) inherit from ``Struct``. The package
provides two APIs to register **existing classes** so they participate in JAX
pytree traversal and struct I/O without changing their inheritance hierarchy.

register_class decorator
=========================

:func:`~neuralqx.experimental.utils.struct.register_class` is a decorator (or direct call)
that promotes an existing class to a ``Struct``.  The original class becomes
the first entry in the MRO so its methods and ``super()`` calls remain valid:

.. code-block:: python

    @s.register_class
    class Params:
        weights: object
        bias: object = s.field(default_factory=lambda: jnp.zeros(1))

    p = Params(weights=jnp.ones(4))
    p.replace(bias=jnp.zeros(4))  # Struct API available

With an optional name override:

.. code-block:: python

    @s.register_class(name="TrainingParams")
    class Params:
        ...

:func:`~neuralqx.experimental.utils.struct.dataclass` is an alias for
``register_class`` provided for readability in codebases that conceptually
think of structs as immutable dataclasses:

.. code-block:: python

    @s.dataclass
    class MyConfig:
        lr: float = s.field(static=True, default=1e-3)

register_pytree_type: full-control API
=======================================

Use :func:`~neuralqx.experimental.utils.struct.register_pytree_type` when you need
complete control over the flatten/unflatten logic for a third-party or
legacy class:

.. code-block:: python

    class Node:
        def __init__(self, data, tag):
            self.data = data
            self.tag = tag

    s.register_pytree_type(
        Node,
        flatten=lambda obj: ([obj.data], obj.tag),
        unflatten=lambda aux, children: Node(children[0], aux),
    )

    # Node is now a JAX pytree:
    import jax
    leaves = jax.tree_util.tree_leaves(Node(jnp.ones(3), "x"))  # [array([1,1,1])]

Optional ``flatten_with_keys`` provides richer key metadata for tools like
``jax.tree_util.tree_map_with_path``:

.. code-block:: python

    s.register_pytree_type(
        Node,
        flatten=lambda obj: ([obj.data], obj.tag),
        unflatten=lambda aux, children: Node(children[0], aux),
        flatten_with_keys=lambda obj: (
            [(jax.tree_util.GetAttrKey("data"), obj.data)],
            obj.tag,
        ),
    )

Optional ``serializer`` / ``deserializer`` pair teaches struct I/O how to
persist instances of this class in ``export`` / ``load`` round-trips when the
raw ``aux`` data is not directly JSON-serialisable:

.. code-block:: python

    s.register_pytree_type(
        Node,
        flatten=lambda obj: ([obj.data], obj.tag),
        unflatten=lambda aux, children: Node(children[0], aux),
        serializer=lambda obj: {"tag": obj.tag},
        deserializer=lambda payload, children: Node(children[0], payload["tag"]),
    )

register_attrs_type: attribute-based shorthand
=================================================

:func:`~neuralqx.experimental.utils.struct.register_attrs_type` is a higher-level
convenience that generates the flatten/unflatten functions automatically from
attribute names.  Use this for attribute-backed classes when you do not want
to write the plumbing manually:

.. code-block:: python

    class Edge:
        def __init__(self, flux, source, target):
            self.flux = flux  # array, pytree leaf
            self.source = source  # int, static metadata
            self.target = target  # int, static metadata

    s.register_attrs_type(
        Edge,
        node_fields=("flux",),
        static_fields=("source", "target"),
    )

    e = Edge(jnp.array(1.5), source=0, target=3)
    # JAX now sees Edge as a pytree with one leaf (flux)
    # and static aux (source, target).

An optional ``constructor`` overrides how instances are rebuilt from the
attribute map (useful when the class constructor has positional-only
arguments or special initialisation logic):

.. code-block:: python

    s.register_attrs_type(
        Edge,
        node_fields=("flux",),
        static_fields=("source", "target"),
        constructor=lambda vals: Edge(vals["flux"], vals["source"], vals["target"]),
    )

Checking and resolving registrations
======================================

.. code-block:: python

    s.is_registered_pytree_type(Node)    # True / False

    # Resolve by class-reference string (used internally by the I/O layer):
    spec = s.resolve_pytree_spec("mymodule:Node")
    spec.cls  # Node
    spec.flatten  # the flatten callable
    spec.unflatten  # the unflatten callable


--------------------------------------------------------------------
The class registry
--------------------------------------------------------------------

neuraLQX serialisation uses **class references** — strings in the format
``"module.path:QualifiedClassName"`` — to identify types portably across
processes.  These references are stored in manifests so that ``load`` can
reconstruct the correct class without hard-coding imports.

.. code-block:: python

    s.class_ref(TrainState)
    # "neuralqx.my_module:TrainState"

    s.resolve_class("neuralqx.my_module:TrainState")
    # <class 'neuralqx.my_module.TrainState'>

You rarely need to call these directly. They are used internally by
``to_state_dict``, ``from_state_dict``, ``export``, and ``load``.  They
become relevant if you build custom I/O tooling on top of the struct layer or
if you need to verify that a class is importable under a given reference.

Resolution follows three steps:

1. In-memory process cache (fastest).
2. Lazy import via ``importlib``.
3. Cache the result for future calls.

All ``Struct`` subclasses and all types registered via
``register_pytree_type`` / ``register_attrs_type`` are automatically added to
the registry at registration time.


--------------------------------------------------------------------
JAX pytree integration in practice
--------------------------------------------------------------------

Because ``Struct`` subclasses are pytrees, they work transparently with all
JAX utilities:

.. code-block:: python

    import jax
    import jax.numpy as jnp

    class Params(s.Struct):
        w: object
        b: object
        lr: float = s.field(static=True, default=1e-3)

    p = Params(w=jnp.ones(4), b=jnp.zeros(4))

    # tree operations
    jax.tree_util.tree_leaves(p)  # [w_array, b_array]
    jax.tree_util.tree_map(jnp.zeros_like, p)  # new Params with zero arrays

    # jit: re-traces only when static field 'lr' changes
    @jax.jit
    def step(params, grad):
        return jax.tree_util.tree_map(lambda p, g: p - params.lr * g, params, grad)

    # grad: differentiates through node fields
    def loss(params):
        return jnp.sum(params.w ** 2)

    g = jax.grad(loss)(p)  # g is a Params with gradient arrays

    # vmap: batches over node fields
    batch = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), p)  # "batch of 2"
    jax.vmap(loss)(batch)

**Static field behaviour under ``jit``:**

Each unique combination of static field values is a separate ``jit``
specialisation. If you change only a static field value and pass the same
function through ``jit``, you will pay a re-compilation cost. This is
expected and correct, static fields encode things like layer sizes or string
keys that change the computation graph.

If a value changes frequently (e.g. a step counter), use a **node field** or
pass it as a separate traced argument, not as a static field.

**Opaque fields under ``jit``:**

Opaque fields are preserved across pytree round-trips (JAX stores them in aux
data via an identity-preserving wrapper) but they are invisible to the tracer.
They are passed through as Python objects and must be the same Python object
(same ``id()``) after unflattening for JAX cache consistency.  Do not store
JAX arrays in opaque fields if you intend to differentiate through them.


--------------------------------------------------------------------
``StructABCMeta``: abstract struct base classes
--------------------------------------------------------------------

If you want to define an **abstract** struct (with ``@abc.abstractmethod``
methods) use :class:`~neuralqx.experimental.utils.struct.StructABCMeta` as the metaclass
instead of the default ``StructMeta``:

.. code-block:: python

    import abc

    class AbstractSolver(s.Struct, metaclass=s.StructABCMeta):
        lr: float = s.field(static=True)

        @abc.abstractmethod
        def step(self, params): ...

    class SGD(AbstractSolver):
        def step(self, params):
            return jax.tree_util.tree_map(lambda p: p - self.lr, params)

``StructABCMeta`` is the combined metaclass for ``StructMeta`` and
``abc.ABCMeta`` and gives you both struct processing and ABC enforcement.
You cannot instantiate an abstract struct class directly. Any concrete
subclass that does not implement all abstract methods will raise
``TypeError`` at instantiation time, exactly like a plain ABC.
