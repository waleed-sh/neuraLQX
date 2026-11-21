---
title: Serialisation
---

# Serialisation

A struct can describe both the state a numerical calculation needs now and the state you want to recover later. These two descriptions need not contain the same fields. Parameters and iteration counters may belong in a saved record, while a cache or an open runtime interface should usually be recreated.

The serialisation tools in [`neuralqx.utils.struct.io`](../../api/public/neuralqx/utils/struct/io/index.md) preserve supported values together with the information needed to reconstruct their classes. They provide an in-memory state dictionary and a file representation containing a JSON manifest and compressed arrays. Both use the [field declarations](fields.md) to decide what is saved.

For a complete variational state, use the dedicated [VQS saving and loading interface](../../concepts/vqs/saving.md). A struct bundle restores the object you supplied. It does not discover the surrounding model, driver, optimiser, or random state and assemble a resumable simulation on your behalf.

## Start with an in-memory round trip

[`to_state_dict()`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.Struct.to_state_dict) encodes an instance, and [`from_state_dict()`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.Struct.from_state_dict) reconstructs it. A round trip is a useful first check when designing a new state container because it exposes missing defaults and unsupported field values before file storage enters the picture.

```python
import jax
import jax.numpy as jnp
from neuralqx.utils.struct import Struct, field

class ParameterRecord(Struct):
    weights: jax.Array
    iteration: jax.Array
    model_name: str = field(static=True, default="example")
    cache: dict = field(pytree=False, default_factory=dict)

original = ParameterRecord(
    weights=jnp.array([0.25, -0.5]),
    iteration=jnp.array(12, dtype=jnp.int32),
)
original.cache["temporary_result"] = 3.0

payload = original.to_state_dict()
restored = ParameterRecord.from_state_dict(payload)

print(sorted(payload))
print(restored.weights.tolist())
print(int(restored.iteration), restored.model_name)
print(restored.cache)
```

```text
['array_data', 'arrays', 'manifest', 'version']
[0.25, -0.5]
12 example
{}
```

The weights, counter, and static model name survive. The opaque cache is omitted and a new empty dictionary is created from its factory. This is a deliberate part of the record's design. It would be inappropriate for a quantity required to continue the calculation, such as an evolving random key, to disappear in this way.

The payload separates four kinds of information.

| Entry | Contents |
| --- | --- |
| `version` | The serialisation schema version |
| `manifest` | Class references, field names, containers, and references to arrays |
| `arrays` | The recorded shape and dtype of each array |
| `array_data` | The actual NumPy array buffers in host memory |

The complete dictionary is not a JSON document because `array_data` contains arrays. Use `export()` when you need files. The inspection helpers `to_dict()` and [`to_python()`](../../api/public/neuralqx/utils/struct/io/index.md#neuralqx.utils.struct.io.to_python) have a different purpose. They provide views of values, but do not retain the reconstruction schema needed for a state-dictionary round trip.

## Save a directory or a ZIP archive

[`export()`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.Struct.export) chooses the file representation from the path. A path ending in `.zip` produces a single archive. Any other suffix, or no suffix, produces a directory. Both contain `manifest.json` and `arrays.npz`, and missing parent directories are created automatically.

The example uses a temporary directory so it can be run repeatedly. In a calculation, replace that directory with your chosen output location, such as `Path("runs/experiment-01")`.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import jax
import jax.numpy as jnp
from neuralqx.utils.struct import Struct, field

class RunRecord(Struct):
    parameters: dict
    iteration: jax.Array
    label: str = field(static=True, default="trial")

record = RunRecord(
    parameters={"weights": jnp.array([1.0, -2.0])},
    iteration=jnp.array(5, dtype=jnp.int32),
)

with TemporaryDirectory() as temporary:
    root = Path(temporary)
    directory = record.export(root / "records" / "step-0005")
    archive = record.export(root / "archives" / "latest.zip")

    from_directory = RunRecord.load(directory)
    from_archive = RunRecord.load(archive)
    print(from_directory.parameters["weights"].tolist())
    print(int(from_archive.iteration), from_archive.label)

    updated = record.replace(iteration=jnp.array(6, dtype=jnp.int32))
    updated.export(archive, overwrite=True)
    print(int(RunRecord.load(archive).iteration))
```

```text
[1.0, -2.0]
5 trial
6
```

An existing destination is protected by default. Pass `overwrite=True` only when you intend to replace it. Export finishes writing a staged bundle before installing it at the destination. ZIP replacement uses a filesystem rename. Replacing a directory first moves the previous directory aside, and restores it if installing the new one fails. Directory replacement is therefore not a single atomic swap for concurrent readers. Arrange for one writer and avoid reading a directory while it is being replaced.

[`load()`](../../api/public/neuralqx/utils/struct/index.md#neuralqx.utils.struct.Struct.load) returns a new object and checks that its type is compatible with the class on which you called it. The module-level [`export`](../../api/public/neuralqx/utils/struct/io/index.md#neuralqx.utils.struct.io.export) and [`load`](../../api/public/neuralqx/utils/struct/io/index.md#neuralqx.utils.struct.io.load) functions also accept supported nested containers and registered pytrees as the root object. For example, a dictionary can hold several related structs in one bundle.

## What is restored, and what is recomputed

Ordinary node fields and static fields are saved by default. Opaque fields are omitted unless you explicitly set `serialize=True`. That setting requests encoding of the value, but does not make an otherwise unsupported object serialisable. Conversely, `serialize=False` omits a node or static field without changing its role in JAX transformations.

A field omitted from the payload must have a default or factory unless it is derived. Required opaque fields without either cannot be reconstructed. A derived field is never stored independently. Its value is recomputed from the restored source fields, as demonstrated in the [derived-field example](fields.md#derived-fields).

Loading follows the struct reconstruction path. It applies converters, recomputes derived fields, and validates the reconstructed values. It does not call the ordinary constructor or `__post_init__`. If a runtime object needs to be recreated, give its field an appropriate factory or derivation. If an invariant must also hold after loading, put the check in a field validator rather than only in `__post_init__`.

Arrays are read into host buffers and then placed into JAX arrays. A NumPy array in the original object therefore returns as a JAX array. The bundle records numerical shape and dtype, but not the original device assignment or sharding. JAX's active dtype policy still applies when placing restored arrays, including whether 64-bit values are enabled.

## Classes and adapters must be available when loading

The manifest stores a class reference using its module and qualified name. It does not contain the Python source implementing the class. A later process must be able to import the class and use a compatible field schema.

For records that will outlive an interactive session, define their classes at module scope in an importable project module. A class created inside a function or notebook cell may round-trip in the current process because it is already registered there, yet be unavailable in a fresh process. The examples on this page demonstrate the mechanics within one running script. Move the class definitions into your project when saving records for later use.

The current format has schema version `1`. Loading checks that version and the stored array metadata. There is no automatic migration of your own renamed classes, moved modules, or changed field meanings. If your record format evolves, keep an explicit conversion path and verify that the reconstructed data still describes the intended calculation.

An existing class can participate through the [pytree registration interface](pytrees.md#adapting-an-existing-class). If its usual flattening representation is also suitable for storage, its children and auxiliary data can be encoded directly. A serializer and deserializer pair provides explicit control when the storage representation needs to differ.

```python
import jax.numpy as jnp
from neuralqx.utils.struct import register_attrs_type
from neuralqx.utils.struct.io import from_state_dict, to_state_dict

class MeasuredSeries:
    def __init__(self, values, unit):
        self.values = values
        self.unit = unit

register_attrs_type(
    MeasuredSeries,
    node_fields=("values",),
    static_fields=("unit",),
    serializer=lambda series: {"values": series.values, "unit": series.unit},
    deserializer=lambda payload: MeasuredSeries(**payload),
)

series = MeasuredSeries(jnp.array([0.5, 1.5]), "normalised")
restored = from_state_dict(to_state_dict(series))
print(type(restored).__name__)
print(restored.values.tolist(), restored.unit)
```

```text
MeasuredSeries
[0.5, 1.5] normalised
```

The serializer must return values that the encoding supports, and its deserializer must be registered in the loading process. Keep the registration beside the class definition, or import the module that performs it before loading. Registering a class only with JAX does not necessarily provide neuraLQX with a storage adapter.

## Supported values and random keys

The encoding handles nested structs, registered pytrees, numerical arrays, Python numerical scalars including complex values, strings, `None`, paths, tuples, lists, dictionaries, sets, and frozen sets. Mapping subclasses are reconstructed as ordinary dictionaries. Shared Python references are not preserved as an object-identity graph, and cyclic containers are not supported. Runtime handles, arbitrary callables, and raw `bytes` need a suitable conversion or an omitted field with a reconstruction rule.

Typed JAX random keys need an explicit representation because their specialised dtype cannot be converted directly to a NumPy buffer. Store the key data together with its generator implementation, and reconstruct the key when you need it. This example keeps both pieces and checks that the restored key produces the same draw.

```python
import jax
import jax.numpy as jnp
from neuralqx.utils.struct import Struct, field

class RandomRecord(Struct):
    key_data: jax.Array
    implementation: str = field(static=True)

key = jax.random.key(17, impl="threefry2x32")
record = RandomRecord(
    key_data=jax.random.key_data(key),
    implementation="threefry2x32",
)
restored = RandomRecord.from_state_dict(record.to_state_dict())
restored_key = jax.random.wrap_key_data(
    restored.key_data, impl=restored.implementation
)

same_draw = jnp.array_equal(
    jax.random.normal(key, (4,)),
    jax.random.normal(restored_key, (4,)),
)
print(bool(same_draw))
```

```text
True
```

This also explains why being a valid JAX pytree does not by itself guarantee that every field can be stored in a struct bundle. Inspect the schema of a custom sampler or numerical component before relying on its serialisation. The [VQS checkpoint interface](../../concepts/vqs/saving.md) handles the supported simulation state through its own reconstruction machinery.

## Multiple processes and memory use

In a distributed JAX calculation, `export()` and `load()` are collective operations. Every process must enter them in the same order, with the same path and compatible metadata. Do not put either call inside a condition that only process zero executes. The implementation coordinates the processes and performs filesystem access only on process zero.

Export checks agreement on the encoded structure, scalar metadata, array shapes and dtypes, and overwrite setting. A global JAX array spanning processes is gathered into its complete value. An ordinary process-local array is saved from process zero. It is not implicitly concatenated with arrays from other processes. Represent data as a global array if every process's contribution must be part of the saved value.

Loading broadcasts the saved buffers and reconstructs the object on each process. Restored arrays are locally addressable and do not recover the original distributed partition. Errors during preparation, filesystem access, or reconstruction are communicated between participating processes so that a recoverable Python failure is reported across the job.

These operations can require the complete array data in each process's host memory. They are useful for manageable state records, but are not a streaming checkpoint mechanism for arrays too large to gather. The in-memory `to_state_dict()` operation is local and does not gather non-addressable global arrays. For simulation checkpoints and placement-aware restoration, continue with [saving variational states](../../concepts/vqs/saving.md) and the [JAX parallelism overview](../../distributed/jax-parallelism.md).
