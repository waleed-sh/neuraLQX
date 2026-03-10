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

"""
A module for serializing/deserializing nearly any Python object, including:
- JAX arrays
- NumPy arrays
- Built-in containers (list, dict, set, tuple)
- Arbitrary user-defined classes
- Fallback for objects lacking __dict__

It also includes:
- A versioning system
- Security checks via SHA-256 checksums
- Post-load hooks
- Optional custom serialization logic with __serializable_attributes__ methods
"""

import importlib
import hashlib
import msgpack

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from neuralqx.debug import errors_only
from .distributed import safe_replicate_for_io

# attempt to import JAX/NumPy if available (we won't crash if they're missing...)
try:
    import jax
    import jax.numpy as jnp

    _HAS_jax = True
except ImportError:
    _HAS_jax = False

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# Module-level serialization format version for backward/forward compatibility
SERIALISATION_FORMAT_VERSION = 1


class SerializerEngine:
    """
    SerializerEngine is an OOP abstraction that encapsulates all logic for serializing and
    deserializing objects (including JAX, NumPy, etc.), as well as saving/loading snapshots to/from
    files with checksums.

    The class provides static methods:
      - serialize(obj): returns a nested dictionary (pytree)
      - deserialize(snapshot): rebuilds the original object from that pytree
      - save_to_file(obj, filename): writes serialized data + checksum to disk
      - load_from_file(filename): reads data from disk and deserializes it

    Internally, it uses helper methods (_serialize_any, _deserialize_any, etc.) to handle
    specialized types like JAX arrays or NumPy arrays.
    """

    @staticmethod
    def dynamic_import_class(class_name: str) -> type:
        """
        Dynamically import a class given its fully qualified string
        (e.g. "my_package.submodule.MyClass").

        :param class_name: the dotted path to the class

        :return: the actual class object
        """

        # split by dots to separate module path from class name
        parts = class_name.split(".")

        # everything except the last part is the module path
        module_name, cls_name = ".".join(parts[:-1]), parts[-1]

        # import the module dynamically
        module = importlib.import_module(module_name)

        # retrieve the class object from the module
        return getattr(module, cls_name)

    @staticmethod
    def compute_sha256(data: bytes) -> str:
        """
        Compute an SHA-256 hash of the given bytes and return it as a hex digest

        :param data: the raw bytes to be hashed

        :return: hexadecimal SHA-256 digest
        """

        # use hashlib.sha256 to get a hasher object
        hasher = hashlib.sha256()

        # feed it the data
        hasher.update(data)

        # return the hex digest
        return hasher.hexdigest()

    @staticmethod
    def get_fully_qualified_name(obj: Any) -> str:
        """
        Returns the fully qualified Python name of obj's class

        :param obj: the object whose class name we want

        :return: e.g. "my_package.submodule.MyClass"
        """

        # extract the __class__ of obj
        cls = obj.__class__

        # return module + class name
        return f"{cls.__module__}.{cls.__name__}"

    @staticmethod
    def maybe_post_load_hook(obj: Any):
        """
        If the object defines a '_post_load_hook()' method, call it.

        This is useful for objects that need to re-initialize ephemeral state.
        """

        # check if the object has post_load_hook and if it's callable
        if hasattr(obj, "_post_load_hook") and callable(obj._post_load_hook):
            # call it if so
            obj._post_load_hook()

    @staticmethod
    def get_serializable_state(obj: Any) -> Dict[str, Any]:
        """
        Get a dictionary of attributes that should be serialized for this object.

        1) If obj.__serializable_attributes__() exists and is callable, use that
        2) Otherwise, if obj.__dict__ exists, store all non-callable attributes
        3) Otherwise, return an empty dict (meaning no attributes are stored)

        :param obj: the object to introspect

        :return: a dictionary of {attr_name: attr_value}
        """

        # check if there's a custom method telling us what to store
        if hasattr(obj, "__serializable_attributes__") and callable(
            obj.__serializable_attributes__
        ):
            # return whatever that method says
            return obj.__serializable_attributes__()

        else:
            # if object has a __dict__, we can gather all non-callable items
            if hasattr(obj, "__dict__"):
                # initialize empty dict
                out = {}

                # iterate through each attribute in obj.__dict__
                for k, v in obj.__dict__.items():

                    # skip methods or other callables
                    if not callable(v):
                        out[k] = v

                # return that dictionary
                return out
            else:
                # if we can't do anything, return an empty dict
                # dev: really?
                return {}

    @staticmethod
    def serialize(obj: Any) -> Dict[str, Any]:
        """
        Serialize a Python object into a nested dict (pytree).

        The result includes:
          - __classname__: the fully qualified class name
          - __version__: the module-level serialization format version
          - __class_version__: optional version from obj.__class_version__
          - __attributes__: the recursively serialized attributes (dict)

        :param obj:the object to serialize

        :return: a nested dictionary with the object's data
        """

        # get any class-specific version tag if present
        class_version = getattr(obj, "__class_version__", 1)

        # build the top-level info, this consists of
        #   - the object's class name
        #   - our global format version
        #   - class-specific version
        #   - a dict which will hold the serialized attributes
        data = {
            "__classname__": SerializerEngine.get_fully_qualified_name(obj),
            "__version__": SERIALISATION_FORMAT_VERSION,
            "__class_version__": class_version,
            "__attributes__": {},
        }

        # get the dictionary of attributes to store
        attr_dict = SerializerEngine.get_serializable_state(obj)

        # recursively serialize each attribute
        data["__attributes__"] = SerializerEngine._serialize_any(attr_dict)

        return data

    @staticmethod
    def _is_jax_array(value: Any) -> bool:
        """Return True if value looks like a JAX array leaf."""
        if not _HAS_jax:
            return False
        try:
            jax_array_type = getattr(jax, "Array", None)
            if jax_array_type is not None and isinstance(value, jax_array_type):
                return True
        except Exception:
            pass
        try:
            return isinstance(value, jnp.ndarray)
        except Exception:
            return False

    @staticmethod
    def _serialize_any(value: Any) -> Any:
        """
        Recursively serialize an arbitrary Python value

        Handles:
          - Primitives (None, bool, int, float, str)
          - JAX arrays (if installed)
          - NumPy arrays (if installed)
          - dict, list, tuple, set
          - Arbitrary user-defined objects (calls serialize(...) again)

        :param value:the object/value to serialize

        :return: a nested dictionary or primitive suitable for msgpack
        """

        # First: handle primitives
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, complex):
            return {"__is_complex__": True, "real": value.real, "imag": value.imag}

        # handle bytes too
        if isinstance(value, bytes):
            return {"__is_bytes__": True, "data": value}

        # Second: check for JAX array
        if _HAS_jax and _HAS_NUMPY and SerializerEngine._is_jax_array(value):
            # convert JAX array to replicated CPU-backed NumPy array for I/O safety
            cpu_arr = np.asarray(
                safe_replicate_for_io(
                    value,
                    replicate_to_all_processes=False,
                    block_until_ready=True,
                )
            )

            # return a dict with markers for shape, dtype, data
            return {
                "__is_jax_array__": True,
                "shape": cpu_arr.shape,
                "dtype": str(cpu_arr.dtype),
                "data": cpu_arr.tolist(),
            }

        # Third: check for NumPy array
        if _HAS_NUMPY and isinstance(value, np.ndarray):
            # return a dict with markers for shape, dtype, data
            return {
                "__is_numpy_array__": True,
                "shape": value.shape,
                "dtype": str(value.dtype),
                "data": value.tolist(),
            }

        # Fourth: if it's a dict, we store each key-value pair
        if isinstance(value, dict):
            return {
                "__is_dict__": True,
                "items": [
                    (k, SerializerEngine._serialize_any(v)) for k, v in value.items()
                ],
            }

        # Fifth: if it's a list, tuple, or set, store them as a list
        if isinstance(value, list):
            return {
                "__is_list__": True,
                "items": [SerializerEngine._serialize_any(v) for v in value],
            }

        if isinstance(value, tuple):
            return {
                "__is_tuple__": True,
                "items": tuple(SerializerEngine._serialize_any(v) for v in value),
            }

        if isinstance(value, set):
            return {
                "__is_set__": True,
                "items": set(SerializerEngine._serialize_any(v) for v in value),
            }

        # Sixth: if it's a NumPy dtype, save it as a str
        if isinstance(value, np.dtype):
            return {"__is_numpy_dtype__": True, "dtype_str": str(value)}

        # Seventh, check if it is a NumPy generic dtype
        if isinstance(value, np.generic):
            return value.item()

        # Eighth: otherwise, treat it as a user-defined or unknown object, so we call
        # serialize(value) again to produce a nested object structure
        return {
            "__is_object__": True,
            "value": SerializerEngine.serialize(value),
        }

    @staticmethod
    def deserialize(snapshot: Dict[str, Any]) -> Any:
        """
        Reconstruct an object from a top-level snapshot (dictionary) previously produced by
        serialize(...).

        :param snapshot:the data structure containing class info and serialized attributes

        :return: the reconstructed (deserialized) object
        """

        # if the snapshot isn't a dict, it might be a primitive or something else
        if not isinstance(snapshot, dict):
            return SerializerEngine._deserialize_any(snapshot)

        # if it doesn't have __classname__, it's not a top-level object
        if "__classname__" not in snapshot:
            # so we parse it as a nested structure
            return SerializerEngine._deserialize_any(snapshot)

        # extract the class name
        class_name = snapshot["__classname__"]

        # dynamically import that class
        class_obj = SerializerEngine.dynamic_import_class(class_name)

        # create an uninitialized instance (skips __init__)
        instance = object.__new__(class_obj)

        # retrieve class version from snapshot
        _class_version_saved = snapshot.get("__class_version__", 1)
        # dev: we can handle migrations here based on version diffs

        # extract the attributes dictionary from the snapshot
        attr_tree = snapshot["__attributes__"]

        # recursively rebuild the attributes
        restored_attrs = SerializerEngine._deserialize_any(attr_tree)

        # assign them onto the instance
        for k, v in restored_attrs.items():
            setattr(instance, k, v)

        # possibly invoke post_load_hook
        SerializerEngine.maybe_post_load_hook(instance)

        return instance

    @staticmethod
    def _deserialize_any(value: Any) -> Any:
        """
        The inverse of _serialize_any: recursively rebuilds Python objects from the nested
        dictionary structure

        :param value: the piece of the snapshot to deserialize

        :return: the corresponding Python object/value
        """

        # First: primitives
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        # Second: if it's a dict, it might contain special markers
        if isinstance(value, dict):

            if value.get("__is_complex__"):
                return complex(value["real"], value["imag"])

            # bytes marker
            if value.get("__is_bytes__"):
                return value["data"]

            # JAX array marker
            if value.get("__is_jax_array__") and _HAS_jax and _HAS_NUMPY:
                shape = value["shape"]
                dtype = value["dtype"]
                data_list = value["data"]
                np_arr = np.array(data_list, dtype=dtype).reshape(shape)

                # convert back to JAX array
                return jnp.array(np_arr)

            # NumPy array marker
            if value.get("__is_numpy_array__") and _HAS_NUMPY:
                shape = value["shape"]
                dtype = value["dtype"]
                data_list = value["data"]
                return np.array(data_list, dtype=dtype).reshape(shape)

            # Dict marker
            if value.get("__is_dict__"):
                items = value["items"]
                return {k: SerializerEngine._deserialize_any(v) for (k, v) in items}

            # List marker
            if value.get("__is_list__"):
                items = value["items"]
                return [SerializerEngine._deserialize_any(x) for x in items]

            if value.get("__is_tuple__"):
                items = value["items"]
                return tuple(SerializerEngine._deserialize_any(x) for x in items)

            if value.get("__is_set__"):
                items = value["items"]
                return set(SerializerEngine._deserialize_any(x) for x in items)

            # object marker
            if value.get("__is_object__"):
                return SerializerEngine.deserialize(value["value"])

            # NumPy dtype marker
            if value.get("__is_numpy_dtype__"):
                return np.dtype(value["dtype_str"])

            # if none of the above markers, treat it as a normal dict
            return {k: SerializerEngine._deserialize_any(v) for k, v in value.items()}

        # Third: if it's neither a dict nor a primitive, we might just return as-is
        return value

    @staticmethod
    def save_to_file(obj: Union[Any, dict], filename: str) -> None:
        """
        Serialize the object and write it to 'filename' using msgpack, along with a SHA-256 checksum
        to detect tampering. You can also pass it an already serialised dict to directly save

        :param obj: the Python object to be serialized or a serialised dict
        :param filename:the path to the file where data will be saved
        """

        # ensure all JAX-backed leaves are materialised on host before serialisation.
        # this keeps I/O deterministic in distributed runs.
        obj = safe_replicate_for_io(
            obj,
            replicate_to_all_processes=False,
            block_until_ready=True,
        )

        # convert object to a nested dictionary
        if not isinstance(obj, dict):
            snapshot = SerializerEngine.serialize(obj)
        else:
            # or it is an already serialised dict
            snapshot = SerializerEngine._serialize_any(obj)

        # pack that dictionary into msgpack bytes
        snapshot_bytes = msgpack.packb(snapshot, use_bin_type=True)

        # compute a checksum
        checksum = SerializerEngine.compute_sha256(snapshot_bytes)

        # build a final structure that includes snapshot + checksum
        final_structure = {
            "snapshot": snapshot,
            "checksum": checksum,
            "format_version": SERIALISATION_FORMAT_VERSION,
        }

        # pack this final structure
        final_bytes = msgpack.packb(final_structure, use_bin_type=True)

        # write to file in binary mode
        with open(filename, "wb") as f:
            f.write(final_bytes)

    @staticmethod
    def load_from_file(
        filename: str,
        raw: bool = False,
    ) -> Any:
        """
        Read a snapshot (and its checksum) from 'filename' and deserialize it.

        :param filename: the path to the file containing the serialized data
        :param raw: a flag to determine whether the given snapshot is already a serialised dict
                    that we do not want to set into an object
        :return: the reconstructed object from the file
        """

        # read all bytes from the file
        with open(filename, "rb") as f:
            file_bytes = f.read()

        # unpack the outer structure from msgpack
        try:
            final_structure = msgpack.unpackb(file_bytes, raw=False)
        except Exception as e:
            raise ValueError(f"Failed to unpack msgpack data: {e}")

        # validate the presence of required keys
        if not all(
            k in final_structure for k in ["snapshot", "checksum", "format_version"]
        ):
            raise ValueError(
                "Malformed serialization file: missing one or more required keys"
            )

        snapshot = final_structure["snapshot"]
        stored_checksum = final_structure["checksum"]
        format_version = final_structure["format_version"]

        # recompute checksum of the snapshot
        snapshot_bytes = msgpack.packb(snapshot, use_bin_type=True)
        computed_checksum = SerializerEngine.compute_sha256(snapshot_bytes)

        # compare checksums
        if computed_checksum != stored_checksum:
            raise ValueError(
                "Checksum mismatch. The file may be corrupted or tampered with."
            )

        # warn if format versions differ
        if format_version != SERIALISATION_FORMAT_VERSION:
            import warnings

            warnings.warn(
                f"Loaded snapshot with format_version={format_version}, "
                f"current is {SERIALISATION_FORMAT_VERSION}.\n\n"
                f"There might be errors arising due to incompatible versions."
            )

        # finally, rebuild the object from the snapshot
        if raw:
            return SerializerEngine._deserialize_any(snapshot)

        return SerializerEngine.deserialize(snapshot)


# Top level functions


@errors_only(tag="SERIALIZER:SERIALIZE")
def serialize(obj: Any) -> Dict[str, Any]:
    """
    Public function to serialize a Python object into a nested dictionary.
    This delegates to the SerializerEngine class.

    :param obj: the object to serialize

    :return: a nested dictionary with the object's data
    """

    # the previous implementation assumed the serialised object is always a class instance
    if isinstance(obj, (dict, list, tuple, set)):
        return SerializerEngine._serialize_any(obj)

    return SerializerEngine.serialize(obj)


@errors_only(tag="SERIALIZER:DESERIALIZE")
def deserialize(snapshot: Dict[str, Any]) -> Any:
    """
    Public function to reconstruct an object from a snapshot dictionary.
    Delegates to the SerializerEngine class.

    :param snapshot: the data structure containing class info and serialized attributes

    :return: the reconstructed (deserialized) object
    """

    # the previous implementation assumed the serialised object is always a class instance
    # if isinstance(snapshot, (dict, list, tuple, set)):
    #     return SerializerEngine._deserialize_any(snapshot)
    #
    # return SerializerEngine.deserialize(snapshot)

    # if this looks like a top-level object snapshot, use the class-aware path.
    if isinstance(snapshot, dict) and "__classname__" in snapshot:
        return SerializerEngine.deserialize(snapshot)

    # otherwise, fall back to generic container deserialization
    return SerializerEngine._deserialize_any(snapshot)


@errors_only(tag="SERIALIZER:SAVE")
def save_to_file(obj: Any, filename: str) -> None:
    """
    Public function to serialize an object and save it to a file with
    a checksum. Delegates to the SerializerEngine class.

    :param obj: the Python object to serialize
    :param filename: path to the output file
    """

    SerializerEngine.save_to_file(obj, filename)


@errors_only(tag="SERIALIZER:LOAD")
def load_from_file(
    filename: str,
    raw: bool = False,
) -> Any:
    """
    Public function to load a snapshot from a file, verify its checksum,
    and deserialize the object. Delegates to the SerializerEngine class.

    :param filename: path to the file to read
    :param raw: a flag to determine whether the given snapshot is already a serialised dict
                    that we do not want to set into an object
    :return: the reconstructed object
    """

    return SerializerEngine.load_from_file(filename, raw)


def auto_serializable(cls):
    """
    A decorator that automatically adds a default __serializable_attributes__()
    method to a class if none is defined. It will basically return all class attributes.
    """

    # check if the class already defines __serializable_attributes__
    if not hasattr(cls, "__serializable_attributes__"):

        # define a default method that gathers all non-callable attributes
        def __serializable_attributes__(self):
            out = {}
            for k, v in self.__dict__.items():
                if not callable(v):
                    out[k] = v
            return out

        # attach that method to the class
        setattr(cls, "__serializable_attributes__", __serializable_attributes__)

    # return the (potentially modified) class
    return cls


def partial_serialize(obj: Any, attr_names: List[str]) -> dict:
    """
    Collects a subset of `obj`'s attributes (those in `attr_names`) and
    serializes them into a dictionary that can be saved to disk.

    :param obj: the Python object from which we want to extract certain attributes
    :param attr_names: list of attribute names (strings) to be serialized

    :returns: a dict mapping each attribute name to its serialized representation
    """

    partial_data = {}
    for name in attr_names:
        # grab the attribute value if it exists (or None if missing)
        value = getattr(obj, name, None)

        # use the same logic the engine uses to turn that value into a nested dictionary
        # (primitives, JAX arrays, etc.)
        partial_data[name] = SerializerEngine._serialize_any(value)

    return partial_data


def partial_deserialize(partial_data: dict, existing_obj: Any) -> None:
    """
    Given a dictionary of serialized attributes (produced by partial_serialize), deserialize them
    and set them onto `existing_obj` in place.

    :param partial_data: a dict of {attr_name -> serialized_value}
    :param existing_obj: the object whose attributes should be updated in-place
    """

    for name, serialized_value in partial_data.items():
        # rebuild the actual Python value
        real_value = SerializerEngine._deserialize_any(serialized_value)

        # assign it onto the existing_obj
        setattr(existing_obj, name, real_value)


def save_multi_partial(
    obj_map: Dict[str, Tuple[Any, List[str]]], filename: str
) -> None:
    """
    For each key in obj_map, we partial-serialize its (obj, attr_list), then store them all in one
    structure. We'll also compute a checksum.

    :param obj_map: a dict like:
            {
              "sampler": (sampler_obj, ["chains", "rng_state"]),
              "optimizer": (optimizer_obj, ["learning_rate", "momentum"]),
              ...
            }
    :param filename: The path where we'll write the msgpack data.
    """

    # build a dict that collects partial_data for each label
    master_data = {}

    for label, (the_obj, attr_names) in obj_map.items():
        # partial_serialize for each object
        partial_dict = partial_serialize(the_obj, attr_names)
        master_data[label] = partial_dict

    # ensure all leaves are host-backed before packing bytes
    master_data = safe_replicate_for_io(
        master_data,
        replicate_to_all_processes=False,
        block_until_ready=True,
    )

    # convert master_data to msgpack bytes
    snapshot_bytes = msgpack.packb(master_data, use_bin_type=True)

    # compute a checksum
    checksum = SerializerEngine.compute_sha256(snapshot_bytes)

    # build the final structure
    final_structure = {
        "multi_partial_data": master_data,
        "checksum": checksum,
        "format_version": 1,
    }

    # pack again
    final_bytes = msgpack.packb(final_structure, use_bin_type=True)

    # write to disk
    with open(filename, "wb") as f:
        f.write(final_bytes)


def load_multi_partial(obj_map: Dict[str, Any], filename: str) -> Dict[str, dict]:
    """
    Loads the multi-partial snapshot from 'filename'. Then for each label in 'obj_map', merges the
    corresponding partial data into the provided object in-place.

    :param obj_map: a dict like:
            {
              "sampler": sampler_obj,
              "optimizer": optimizer_obj,
              "network": network_obj,
            }
            (Note: no attribute lists here; we already have the partial_data stored in the file)

    :param filename: path to the msgpack file that has multi_partial_data

    :return: a dict of {label -> partial_data}, in case we want to do custom merges or just to see
     the raw data. We do the in-place assignment too
    """

    # read the file
    with open(filename, "rb") as f:
        file_bytes = f.read()

    # unpack the structure
    try:
        final_structure = msgpack.unpackb(file_bytes, raw=False)
    except Exception as e:
        raise ValueError(f"Failed to unpack msgpack data: {e}")

    # validate
    if not all(
        k in final_structure
        for k in ["multi_partial_data", "checksum", "format_version"]
    ):
        raise ValueError(
            "Malformed multi-partial file: missing one or more required keys"
        )

    master_data = final_structure["multi_partial_data"]
    stored_checksum = final_structure["checksum"]

    # recompute the checksum
    master_bytes = msgpack.packb(master_data, use_bin_type=True)
    computed_checksum = SerializerEngine.compute_sha256(master_bytes)

    if computed_checksum != stored_checksum:
        raise ValueError(
            "Checksum mismatch. The file may be corrupted or tampered with."
        )

    # dev: optionally check format_version, etc. here

    # now, master_data is a dict of {label -> partial_dict}.
    # for each label that also exists in obj_map, we do partial_deserialize
    for label, partial_dict in master_data.items():

        if label in obj_map:
            # we have an object in obj_map for this label
            existing_obj = obj_map[label]
            partial_deserialize(partial_dict, existing_obj)
        else:
            # we might warn or ignore if there's partial data for a label we didn't supply
            pass

    # return the raw partial dict in case we want to do something custom afterwards
    return master_data
