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
This file contains implementation of parsing related helper functions
"""

import ast
import inspect
import types
import functools

from typing import Union
from typing import Callable
from typing import Any

import flax.linen as fln


def required_kwargs(*required_args, conditional_args=None):
    """
    Decorator enforcing the presence of required keyword arguments.

    The returned decorator validates that specified keyword arguments are provided when the
    decorated function is called. It supports:

    - unconditional requirements: names listed in ``required_args`` must always appear in ``kwargs``
    - conditional requirements: extra required kwargs depending on the value of another kwarg

    Conditional requirements are specified as::

        conditional_args = {
            "<condition_key>": {
                "<condition_value_1>": ["req_kw_1", "req_kw_2", ...],
                "<condition_value_2>": [...],
            },
            ...
        }

    If ``condition_key`` is present in ``kwargs`` and its value matches one of the configured
    ``condition_value`` entries, then the corresponding list of required kwargs must also be
    present.

    :param required_args: Names of keyword arguments that must always be present.
    :param conditional_args: Optional mapping
                             ``{condition_key: {condition_value: [required_kw, ...]}}`` defining
                             conditional requirements.
    :returns: A decorator that wraps the target function and raises :class:`ValueError` when
              required kwargs are missing.
    :raises ValueError: When any unconditional or conditional required keyword argument is missing.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # check "unconditional" required kwargs
            missing = [arg for arg in required_args if arg not in kwargs]
            if missing:
                raise ValueError(
                    f"Missing required argument {', '.join(missing)} for '{func.__name__}()'"
                )

            # check "conditional" required kwargs
            #    e.g. if conditional_args = {
            #           "sampler_type": {
            #               "Weighted Sampler": ["rules", "probabilities"],
            #               ...
            #           }
            #         }
            if conditional_args:
                for condition_key, condition_map in conditional_args.items():
                    if condition_key in kwargs:
                        condition_value = kwargs[condition_key]
                        # if we have a list of required fields for this value
                        if condition_value in condition_map:
                            extra_required = condition_map[condition_value]
                            # check them
                            for req in extra_required:
                                if req not in kwargs:
                                    raise ValueError(
                                        f"The argument '{req}' is required when "
                                        f"'{condition_key}' = '{condition_value}' "
                                        f"for '{func.__name__}()'."
                                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_module_attributes(solver, module, prefix=""):
    """
    Recursively log attributes of a Flax module into the solver logger.

    This function iterates over the public attributes of a Flax ``linen.Module`` instance and logs
    each attribute into the solver's logger under the "Network Configs" section. For nested module
    attributes, it recurses to capture the full hierarchy.

    Wrapper modules:
        If ``module`` has an attribute ``base`` that is itself a Flax module (a common wrapper
        pattern), the attributes of ``module.base`` are also logged under the ``<prefix>.base``
        namespace.

    Logging details:
    - The logger is retrieved as ``solver.logger`` if available, otherwise ``solver._logger``.
    - Attributes starting with ``"_"`` and the attribute named ``"name"`` are skipped.
    - Values are converted to readable strings using :func:`get_attr_val`.

    :param solver: Object providing a logger via ``solver.logger`` or ``solver._logger``.
    :param module: Flax module instance whose attributes should be logged.
    :param prefix: Optional namespace prefix for the logged attribute names. If empty, the module
                   class name is used.
    :returns: ``None``.
    """

    try:
        logger = solver.logger
    except AttributeError:
        logger = solver._logger

    # get the name used in logs
    mod_name = prefix or module.__class__.__name__

    for attrib_name, attrib_val in vars(module).items():
        # skip private and internal Flax members
        if attrib_name.startswith("_") or attrib_name == "name":
            continue

        # log wrapper/base attributes
        field_name = f"{mod_name}.{attrib_name}"

        logger.add_field("Network Configs", field_name)
        logger.log(field_name, get_attr_val(attrib_val))

        # if this attribute is itself a module, recurse
        if isinstance(attrib_val, fln.Module):
            # important recursion: log nested modules
            log_module_attributes(solver, attrib_val, prefix=field_name)

    # special case: wrapper with attribute `base`
    if hasattr(module, "base") and isinstance(module.base, fln.Module):
        base = module.base
        base_prefix = f"{mod_name}.base"
        log_module_attributes(solver, base, prefix=base_prefix)


def get_attr_val(attr) -> str:
    """
    Return a robust string representation for an attribute value.

    This helper is designed for logging configuration-like objects where naïve ``str(attr)``
    may fail or be uninformative. It handles:

    - Flax ``linen.Module`` instances (returns the class name)
    - Python modules (returns ``"Module '<name>'"``)
    - Classes (returns the class name when available)
    - Common containers (lists/tuples/dicts), recursively formatting their contents
    - ``functools.partial`` (returns ``"partial(<funcname>)"``)
    - Generic callables without ``__name__`` (returns a best-effort type/name)
    - Fallback to ``str(attr)``, with a safe type-based fallback if string conversion fails

    :param attr: Attribute value to stringify.
    :returns: A human-readable string representation suitable for logs.
    """

    # Flax Modules (instances)
    #    They are callable, but do not have __name__.
    try:
        import flax.linen as nn

        if isinstance(attr, nn.Module):
            return type(attr).__name__
    except Exception:
        # flax not available or other import issues
        pass

    # Python modules
    if isinstance(attr, types.ModuleType):
        return f"Module '{attr.__name__}'"

    # classes
    if inspect.isclass(attr):
        return str(getattr(attr, "__name__", type(attr).__name__))

    # common containers
    if isinstance(attr, list):
        return str([get_attr_val(item) for item in attr])
    if isinstance(attr, tuple):
        return str(tuple(get_attr_val(item) for item in attr))
    if isinstance(attr, dict):
        return str({k: get_attr_val(v) for k, v in attr.items()})

    # functools.partial (optional nicety)
    if isinstance(attr, functools.partial):
        fn_name = getattr(attr.func, "__name__", type(attr.func).__name__)
        return f"partial({fn_name})"

    # generic callables
    if callable(attr):
        return str(getattr(attr, "__name__", type(attr).__name__))

    # fallback stringify
    try:
        return str(attr)
    except Exception:
        return f"'{type(attr).__name__}'"


def str_edge_parser(s: str) -> Union[tuple[int, int], tuple[int, int, int]]:
    """
    Parse an edge represented as a string into a Python tuple.

    The input must be a string that looks like a tuple (starts with ``"("`` and ends with ``")"``).
    The contents are parsed with :func:`ast.literal_eval` for safety.

    Expected formats:
    - Planar graphs: a tuple of integers ``(a, b)``
    - Non-planar graphs (coordinate vertices): a tuple of tuples
      ``((a, b, c), (d, e, f))`` describing the coordinates of the two vertices.

    :param s: String representation of an edge tuple.
    :returns: Parsed tuple representing the edge.
    :raises ValueError: If the string is not tuple-like, cannot be parsed, or does not evaluate to a
                        Python tuple.
    """

    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError(
            "Invalid edge format. The input edge must be a tuple of integers. "
            "For a planar graph, this is a tuple of integers (a, b). For non-planar graphs, "
            "this is a tuple of "
            "tuples ((a, b, c), (d, e, f)) where (a, b, c), (d, e, f) specify the coordinates of "
            "the vertices "
            "the edge is connecting."
        )

    try:
        result = ast.literal_eval(s)
        if not isinstance(result, tuple):
            raise ValueError(
                f"Invalid edge format. Your edge is of type {type(result)} but type tuple is "
                f"required."
            )
        return result
    except ValueError as exc:
        raise ValueError("Invalid edge format.") from exc


def calculate_total_terms(current_level_terms: dict) -> int:
    """
    Compute the total number of term entries across all vertices/triplets for a TRC level.

    The input is expected to be a dictionary keyed by vertex (or similar identifiers) whose values
    contain a ``"triplets"`` list. Each triplet entry is expected to have a ``"terms"`` field that
    is a list of term objects. The total is the sum of ``len(triplet["terms"])`` over all triplets
    in the structure.

    :param current_level_terms: Nested contributions structure containing triplets and their term
                                lists.
    :returns: Total number of terms across all triplets in all contributions.
    :raises KeyError: If the expected keys (``"triplets"`` / ``"terms"``) are missing.
    :raises TypeError: If the values do not have the expected container structure.
    """

    return sum(
        len(triplet["terms"])
        for contributions in current_level_terms.values()
        for triplet in contributions["triplets"]
    )


def strict_type(obj):
    """
    Return the "semantic base class" of an object whose runtime type may be a
    parameterised Generic specialization.

    This helper is meant for libraries that use `typing.Generic` and produce runtime-generated subclasses with names
    like ``Foo[Bar]`` (for example, lazy-operator wrappers or generic specializations). In such cases, ``type(obj)`` is
    not the unparameterised class ``Foo``, but a distinct dynamically-created subclass whose ``__name__`` contains
    ``[...]``.

    The function attempts to map such a specialised runtime class back to its corresponding unparameterised base class
    by:

    1. Taking the runtime class ``cls`` (or using the input directly if it is a class).
    2. If ``cls.__name__`` does not contain ``'['``, returning ``cls`` unchanged.
    3. Otherwise, extracting the base name before the bracket (e.g. ``"Foo"`` from ``"Foo[Bar]"``).
    4. Walking the method resolution order (MRO) and returning the first base class whose ``__name__`` matches that
        extracted base name.

    If no matching base class name is found in the MRO, the function falls back to returning ``cls``.

    Notes:
    - This is a pragmatic runtime-introspection utility. It relies on the naming convention ``BaseName[...]`` and may
       not apply to all generic implementations.

    Examples:
        >>> type(op)  # doctest: +SKIP
        <class 'mypkg.Foo[Bar]'>
        >>> strict_type(op)  # doctest: +SKIP
        <class 'mypkg.Foo'>
        >>> strict_type(op) is mypkg.Foo  # doctest: +SKIP
        True
    """
    cls = obj if isinstance(obj, type) else type(obj)

    name = cls.__name__
    if "[" not in name:
        return cls

    base_name = name.split("[", 1)[0]

    # walk MRO and return the first class whose name matches the unparameterised base
    for b in cls.__mro__[1:]:
        if b.__name__ == base_name:
            return b

    # fallback
    return cls
