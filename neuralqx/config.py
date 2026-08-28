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


from __future__ import annotations

import contextlib
import enum
import hashlib
import json
import logging
import os
import sys
import threading

from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime

from typing import Any
from typing import Generic
from typing import Iterator
from typing import Mapping
from typing import MutableMapping
from typing import Sequence
from typing import TextIO
from typing import cast

from neuralqx.utils.typing import ConfigValueT
from neuralqx.utils.typing import MutationHook
from neuralqx.utils.typing import Parser
from neuralqx.utils.typing import Validator

_LOGGER = logging.getLogger("neuralqx.config")


class ConfigError(RuntimeError):
    """Base class for all neuraLQX configuration errors."""


class UnknownOptionError(ConfigError):
    """Raised when accessing or mutating an unknown configuration option."""


class ConfigValidationError(ConfigError):
    """Raised when a configuration value fails parsing or validation."""


class ReadOnlyDict(dict[str, Any]):
    """Immutable dictionary used for package-wide static values."""

    def _readonly(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("This configuration dictionary is read-only.")

    __setitem__ = _readonly
    __delitem__ = _readonly
    clear = _readonly
    pop = _readonly
    popitem = _readonly
    setdefault = _readonly
    update = _readonly


class ConfigMutability(str, enum.Enum):
    """Defines when a configuration option may be changed.

    Attributes:
        IMMUTABLE: Cannot be changed at runtime via the Python API.
            Value is determined exclusively by defaults and environment.
        STARTUP: Can be changed programmatically only before runtime is locked.
            Suitable for compile-signature or execution-topology options.
        RUNTIME: Can be changed at any time and can be overridden thread-locally.
    """

    IMMUTABLE = "immutable"
    STARTUP = "startup"
    RUNTIME = "runtime"


class ConfigSource(str, enum.Enum):
    """Describes where an observed value came from.

    Attributes:
        DEFAULT: Value originated from the option's built-in default.
        ENV_DEFAULT: Value originated from an environment "default" variable.
        ENV_FORCE: Value originated from an environment "force" variable.
        USER: Value originated from a direct programmatic update.
        PATCH: Value originated from a temporary patch context.
        THREAD_LOCAL: Value originated from a thread-local runtime override.
        RESET: Value originated from removal of an override.
    """

    DEFAULT = "default"
    ENV_DEFAULT = "env_default"
    ENV_FORCE = "env_force"
    USER = "user"
    PATCH = "patch"
    THREAD_LOCAL = "thread_local"
    RESET = "reset"


class _UnsetType:
    """Sentinel used for uninitialized optional values."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"


UNSET = _UnsetType()
_MISSING = _UnsetType()


@dataclass(frozen=True, slots=True)
class ConfigMutation:
    """Represents a single effective config mutation event.

    Attributes:
        name: Option name that changed.
        old_value: Value before mutation.
        new_value: Value after mutation.
        source: Mutation source.
        thread_local: Whether the mutation was thread-local.
        mutability: Declared mutability policy for the option.
    """

    name: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    thread_local: bool
    mutability: ConfigMutability


@dataclass(frozen=True, slots=True)
class ConfigOption(Generic[ConfigValueT]):
    """Static declaration of one config option.

    Attributes:
        name: Public option name.
        default: Built-in default value.
        doc: User-facing help text.
        value_type: Runtime type contract after parsing. May be a single type
            or a tuple of accepted types.
        parser: Optional parser that converts raw inputs into canonical values.
            It is used for both environment variables and runtime updates.
        validator: Optional semantic validator run after parsing.
        env_default: Ordered environment variable names used to override the
            default value. The first present variable wins.
        env_force: Ordered environment variable names that force the effective
            value and prevent runtime mutation. The first present variable wins.
        role: Functional role/category for this option.
        mutability: Option mutability policy.
        include_in_fingerprint: Whether this option participates in
            :meth:`ConfigManager.fingerprint`.
    """

    name: str
    default: ConfigValueT
    doc: str
    value_type: type[Any] | tuple[type[Any], ...] | None = None
    parser: Parser | None = None
    validator: Validator | None = None
    env_default: tuple[str, ...] = ()
    env_force: tuple[str, ...] = ()
    role: str = "general"
    mutability: ConfigMutability = ConfigMutability.STARTUP
    include_in_fingerprint: bool = True

    def parse(self, raw_value: Any) -> ConfigValueT:
        """Parses and validates a raw option value.

        Args:
            raw_value: Input value from environment or user code.

        Returns:
            Parsed and validated value.

        Raises:
            ConfigValidationError: If parsing, type checking, or validation
                fails.
        """
        parsed = self.parser(raw_value) if self.parser is not None else raw_value
        if self.value_type is not None and not isinstance(parsed, self.value_type):
            expected = self.value_type
            raise ConfigValidationError(
                f"Option '{self.name}' expects value of type {expected}, "
                f"got {type(parsed)} with value {parsed!r}."
            )
        if self.validator is not None:
            self.validator(parsed)
        return cast(ConfigValueT, parsed)


@dataclass(slots=True)
class _OptionState(Generic[ConfigValueT]):
    """Mutable runtime state for a single option."""

    spec: ConfigOption[ConfigValueT]
    env_default_value: ConfigValueT | _UnsetType = UNSET
    env_force_value: ConfigValueT | _UnsetType = UNSET
    user_override: ConfigValueT | _UnsetType = UNSET


def parse_bool(value: Any) -> bool:
    """Parses a boolean with permissive string handling.

    Args:
        value: Value to parse.

    Returns:
        Parsed boolean value.

    Raises:
        ConfigValidationError: If the value cannot be interpreted as a boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ConfigValidationError(f"Cannot parse boolean from value {value!r}.")


def parse_int(value: Any) -> int:
    """Parses an integer value.

    Args:
        value: Value to parse.

    Returns:
        Parsed integer.

    Raises:
        ConfigValidationError: If conversion fails or value is a boolean.
    """
    if isinstance(value, bool):
        raise ConfigValidationError("Boolean values are not valid integers here.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"Cannot parse int from value {value!r}.") from exc


def parse_float(value: Any) -> float:
    """Parses a float value.

    Args:
        value: Value to parse.

    Returns:
        Parsed float.

    Raises:
        ConfigValidationError: If conversion fails or value is a boolean.
    """
    if isinstance(value, bool):
        raise ConfigValidationError("Boolean values are not valid floats here.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(
            f"Cannot parse float from value {value!r}."
        ) from exc


def parse_optional_int(value: Any) -> int | None:
    """Parses an optional integer.

    Args:
        value: Value to parse.

    Returns:
        Parsed integer or ``None``.

    Raises:
        ConfigValidationError: If value is neither empty/none-like nor integer.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return parse_int(value)


def parse_optional_string(value: Any) -> str | None:
    """Parses an optional string.

    Args:
        value: Value to parse.

    Returns:
        Parsed string or ``None``.

    Raises:
        ConfigValidationError: If value is not string-like.
    """
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.lower() in {"", "none", "null"}:
            return None
        return trimmed
    raise ConfigValidationError(f"Cannot parse optional string from value {value!r}.")


def parse_csv_tuple(value: Any) -> tuple[str, ...]:
    """Parses a comma-separated value list into a tuple of strings.

    Args:
        value: CSV string or sequence of strings.

    Returns:
        Tuple of normalized non-empty feature names.

    Raises:
        ConfigValidationError: If value is not parseable.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return tuple(item for item in items if item)
    if isinstance(value, Sequence):
        parsed: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ConfigValidationError(
                    f"CSV tuple entries must be strings, got {type(item)}."
                )
            stripped = item.strip()
            if stripped:
                parsed.append(stripped)
        return tuple(parsed)
    raise ConfigValidationError(f"Cannot parse CSV tuple from value {value!r}.")


def parse_optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    """Parses an optional comma-separated integer tuple.

    This is used for settings like JAX local device IDs, where an omitted value
    should defer to JAX's own runtime defaults.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "null"}:
            return None
        return tuple(
            parse_int(item.strip()) for item in stripped.split(",") if item.strip()
        )
    if isinstance(value, int):
        return (parse_int(value),)
    if isinstance(value, Sequence):
        return tuple(parse_int(item) for item in value)
    raise ConfigValidationError(
        f"Cannot parse optional int tuple from value {value!r}."
    )


_SHARDING_AXIS_ORDER = ("samples", "parameters", "operators")
_SHARDING_AXIS_ALIASES = {
    "sample": "samples",
    "samples": "samples",
    "chain": "samples",
    "chains": "samples",
    "data": "samples",
    "parameter": "parameters",
    "parameters": "parameters",
    "param": "parameters",
    "params": "parameters",
    "model": "parameters",
    "operator": "operators",
    "operators": "operators",
    "op": "operators",
    "ops": "operators",
}
_SHARDING_PRESETS = {
    "default": ("samples",),
    "sample": ("samples",),
    "samples": ("samples",),
    "data": ("samples",),
    "parameter": ("samples", "parameters"),
    "parameters": ("samples", "parameters"),
    "param": ("samples", "parameters"),
    "params": ("samples", "parameters"),
    "sample_parameter": ("samples", "parameters"),
    "samples_parameters": ("samples", "parameters"),
    "sample_parameters": ("samples", "parameters"),
    "samples_parameter": ("samples", "parameters"),
    "full": ("samples", "parameters", "operators"),
    "all": ("samples", "parameters", "operators"),
    "sample_parameter_operator": ("samples", "parameters", "operators"),
    "samples_parameters_operators": ("samples", "parameters", "operators"),
    "sample_parameters_operators": ("samples", "parameters", "operators"),
    "samples_parameter_operator": ("samples", "parameters", "operators"),
}


def parse_sharding_axes(value: Any) -> tuple[str, ...]:
    """Parses the high-level sharding policy into canonical mesh axes.

    Accepted examples include ``"samples"``, ``"parameters"``,
    ``"samples,parameters"``, ``"samples+parameters+operators"``, and
    ``"all"``. Parameter/operator-only inputs are interpreted additively on
    top of the default sample sharding.
    """
    if value is None:
        return ("samples",)
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"", "none", "off", "false", "replicated"}:
            return ()
        if normalized in _SHARDING_PRESETS:
            return _SHARDING_PRESETS[normalized]
        raw_items = normalized.replace("+", ",").replace(";", ",").replace(" ", ",")
        items = tuple(item for item in raw_items.split(",") if item)
    elif isinstance(value, Sequence):
        items = tuple(str(item).strip().lower().replace("-", "_") for item in value)
    else:
        raise ConfigValidationError(f"Cannot parse sharding axes from value {value!r}.")

    parsed: set[str] = set()
    for item in items:
        try:
            parsed.add(_SHARDING_AXIS_ALIASES[item])
        except KeyError as exc:
            raise ConfigValidationError(
                f"Unknown sharding axis {item!r}; expected samples, parameters, or operators."
            ) from exc

    if parsed and "samples" not in parsed:
        parsed.add("samples")
    return tuple(axis for axis in _SHARDING_AXIS_ORDER if axis in parsed)


def positive_int_validator(value: int) -> None:
    """Validates positive integers.

    Args:
        value: Integer value.

    Raises:
        ConfigValidationError: If value is not strictly positive.
    """
    if value <= 0:
        raise ConfigValidationError(f"Expected positive integer, got {value}.")


def non_negative_int_validator(value: int) -> None:
    """Validates non-negative integers.

    Args:
        value: Integer value.

    Raises:
        ConfigValidationError: If value is negative.
    """
    if value < 0:
        raise ConfigValidationError(f"Expected non-negative integer, got {value}.")


def non_empty_string_validator(value: str) -> None:
    """Validates non-empty strings after parsing/normalization."""
    if not value:
        raise ConfigValidationError("Expected a non-empty string.")


def non_negative_int_tuple_validator(value: tuple[int, ...] | None) -> None:
    """Validates optional integer tuples as non-negative values."""
    if value is None:
        return
    for item in value:
        non_negative_int_validator(item)


class ConfigManager:
    """Typed and hookable configuration manager for neuraLQX.

    This class stores option declarations and runtime state, including
    environment-derived values, user overrides, and thread-local overrides.

    The effective value precedence is:

    1. Forced environment value (if present)
    2. Thread-local override (runtime-mutable options only)
    3. User global override
    4. Environment default value (if present)
    5. Built-in option default

    Notes:
        - Forced environment values are intentionally immutable from Python.
        - Startup options become immutable once :meth:`lock_runtime` is called.
        - Runtime options may be changed globally or per-thread.
    """

    def __init__(self) -> None:
        """Initializes an empty configuration manager."""
        object.__setattr__(self, "_options", {})
        object.__setattr__(self, "_hooks", {})
        object.__setattr__(self, "_global_hooks", [])
        object.__setattr__(self, "_runtime_locked", False)
        object.__setattr__(self, "_thread_local", threading.local())
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_statics", self._build_statics())

    def __repr__(self) -> str:
        """Returns concise user-facing config-manager summary."""
        return (
            "ConfigManager("
            f"options={len(self._options)}, "
            f"runtime_locked={self._runtime_locked})"
        )

    def __getattr__(self, name: str) -> Any:
        """Reads an option as an attribute.

        Args:
            name: Option name.

        Returns:
            Effective option value.

        Raises:
            AttributeError: If ``name`` is unknown.
        """
        if name in self._options:
            return self.get(name)
        raise AttributeError(f"{self.__class__.__name__} has no option '{name}'.")

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets an option as an attribute.

        Args:
            name: Option name.
            value: New value.

        Raises:
            AttributeError: If ``name`` is unknown.
        """
        internal_attr = name.startswith("_") or name in {
            "register_option",
            "set",
            "get",
            "patch",
            "thread_local",
        }
        if internal_attr:
            object.__setattr__(self, name, value)
            return
        if "_options" in self.__dict__ and name in self._options:
            self.set(name, value)
            return
        raise AttributeError(f"{self.__class__.__name__} has no option '{name}'.")

    @staticmethod
    def _build_statics() -> ReadOnlyDict:
        """Builds read-only package-wide static values."""
        date_str = datetime.now().strftime("%Y%m%d")
        log_dir = os.path.join(os.getcwd(), ".neuralqx_logs", f"neuralqx_{date_str}")
        return ReadOnlyDict(
            {
                "Errors Directory": (
                    "https://neuralqx.readthedocs.io/en/latest/guides/errors.html"
                ),
                "Cache Directory": os.path.join(os.getcwd(), ".neuralqx_cache"),
                "Log Directory": log_dir,
                "Leach Directory": log_dir,
                "Profiling Directory": os.path.join(
                    os.getcwd(),
                    ".neuralqx_profiling",
                    f"neuralqx_{date_str}",
                ),
            }
        )

    @property
    def statics(self) -> ReadOnlyDict:
        """Read-only package-wide static values."""
        return self._statics

    def get_static(self, name: str) -> Any:
        """Returns a registered static value.

        Args:
            name: Static value name.

        Returns:
            The static value.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        try:
            return self._statics[name]
        except KeyError as exc:
            raise KeyError(f"Unknown static variable {name!r}.") from exc

    @staticmethod
    def _normalize_env_names(names: str | Sequence[str] | None) -> tuple[str, ...]:
        """Normalizes env var name declarations.

        Args:
            names: Single name, sequence of names, or ``None``.

        Returns:
            Tuple of normalized environment variable names.

        Raises:
            ConfigValidationError: If names are not strings.
        """
        if names is None:
            return ()
        if isinstance(names, str):
            return (names,)
        normalized: list[str] = []
        for name in names:
            if not isinstance(name, str):
                raise ConfigValidationError(
                    f"Environment variable names must be strings, got {type(name)}."
                )
            normalized.append(name)
        return tuple(normalized)

    @staticmethod
    def _enum_parser(values: Sequence[str], *, case_sensitive: bool) -> Parser:
        """Builds a parser for enum-like string options.

        Args:
            values: Allowed values.
            case_sensitive: Whether matching should be case-sensitive.

        Returns:
            Parser function.
        """
        if case_sensitive:
            allowed = tuple(values)
            allowed_set = set(allowed)

            def parser(raw: Any) -> str:
                if not isinstance(raw, str):
                    raise ConfigValidationError(
                        f"Expected string enum value in {allowed}, got {raw!r}."
                    )
                if raw not in allowed_set:
                    raise ConfigValidationError(
                        f"Invalid enum value {raw!r}; expected one of {allowed}."
                    )
                return raw

            return parser

        canonical = tuple(value.lower() for value in values)
        canonical_set = set(canonical)

        def parser(raw: Any) -> str:
            if not isinstance(raw, str):
                raise ConfigValidationError(
                    f"Expected string enum value in {canonical}, got {raw!r}."
                )
            normalized = raw.strip().lower()
            if normalized not in canonical_set:
                raise ConfigValidationError(
                    f"Invalid enum value {raw!r}; expected one of {canonical}."
                )
            return normalized

        return parser

    def register_option(self, option: ConfigOption[Any]) -> None:
        """Registers a fully specified option.

        Args:
            option: Option declaration.

        Raises:
            ConfigValidationError: If option name is duplicated or invalid.
        """
        with self._lock:
            if not option.name or not isinstance(option.name, str):
                raise ConfigValidationError("Option name must be a non-empty string.")
            if option.name in self._options:
                raise ConfigValidationError(
                    f"Option '{option.name}' is already registered."
                )
            canonical_default = option.parse(option.default)
            if canonical_default != option.default:
                option = replace(option, default=canonical_default)

            state: _OptionState[Any] = _OptionState(spec=option)
            state.env_default_value = self._read_env_value(
                option=option,
                env_names=option.env_default,
            )
            state.env_force_value = self._read_env_value(
                option=option,
                env_names=option.env_force,
            )

            self._options[option.name] = state
            self._hooks[option.name] = []

    def read(self, name: str, *, thread_local: bool = True) -> Any:
        """Alias for :meth:`get`.

        Args:
            name: Option name.
            thread_local: Whether thread-local overrides are considered.

        Returns:
            Effective option value.
        """
        return self.get(name, thread_local=thread_local)

    def update(
        self,
        name: str,
        value: Any,
        *,
        source: ConfigSource = ConfigSource.USER,
        thread_local: bool = False,
    ) -> Any:
        """Alias for :meth:`set`.

        Args:
            name: Option name.
            value: New value.
            source: Source label.
            thread_local: Whether update is thread-local.

        Returns:
            New effective value.
        """
        return self.set(name, value, source=source, thread_local=thread_local)

    def define_option(
        self,
        name: str,
        *,
        default: Any,
        doc: str,
        value_type: type[Any] | tuple[type[Any], ...] | None = None,
        parser: Parser | None = None,
        validator: Validator | None = None,
        env_default: str | Sequence[str] | None = None,
        env_force: str | Sequence[str] | None = None,
        role: str = "general",
        mutability: ConfigMutability = ConfigMutability.STARTUP,
        include_in_fingerprint: bool = True,
    ) -> None:
        """Defines and registers a generic option.

        Args:
            name: Option name.
            default: Built-in default value.
            doc: User-facing description.
            value_type: Expected runtime type after parsing.
            parser: Optional parsing function.
            validator: Optional semantic validator.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force effective value.
            role: Functional role/category for this option.
            mutability: Option mutability policy.
            include_in_fingerprint: Whether to include this option in
                :meth:`fingerprint`.
        """
        option = ConfigOption(
            name=name,
            default=default,
            doc=doc,
            value_type=value_type,
            parser=parser,
            validator=validator,
            env_default=self._normalize_env_names(env_default),
            env_force=self._normalize_env_names(env_force),
            role=role,
            mutability=mutability,
            include_in_fingerprint=include_in_fingerprint,
        )
        self.register_option(option)

    def define_bool(
        self,
        name: str,
        *,
        default: bool,
        doc: str,
        env_default: str | Sequence[str] | None = None,
        env_force: str | Sequence[str] | None = None,
        role: str = "general",
        mutability: ConfigMutability = ConfigMutability.STARTUP,
        include_in_fingerprint: bool = True,
    ) -> None:
        """Defines a boolean option.

        Args:
            name: Option name.
            default: Default boolean.
            doc: User-facing description.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force effective value.
            role: Functional role/category for this option.
            mutability: Mutability policy.
            include_in_fingerprint: Fingerprint participation toggle.
        """
        self.define_option(
            name,
            default=default,
            doc=doc,
            value_type=bool,
            parser=parse_bool,
            env_default=env_default,
            env_force=env_force,
            role=role,
            mutability=mutability,
            include_in_fingerprint=include_in_fingerprint,
        )

    def define_int(
        self,
        name: str,
        *,
        default: int,
        doc: str,
        env_default: str | Sequence[str] | None = None,
        env_force: str | Sequence[str] | None = None,
        role: str = "general",
        mutability: ConfigMutability = ConfigMutability.STARTUP,
        validator: Validator | None = None,
        include_in_fingerprint: bool = True,
    ) -> None:
        """Defines an integer option.

        Args:
            name: Option name.
            default: Default integer.
            doc: User-facing description.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force effective value.
            role: Functional role/category for this option.
            mutability: Mutability policy.
            validator: Optional validator for semantic constraints.
            include_in_fingerprint: Fingerprint participation toggle.
        """
        self.define_option(
            name,
            default=default,
            doc=doc,
            value_type=int,
            parser=parse_int,
            validator=validator,
            env_default=env_default,
            env_force=env_force,
            role=role,
            mutability=mutability,
            include_in_fingerprint=include_in_fingerprint,
        )

    def define_float(
        self,
        name: str,
        *,
        default: float,
        doc: str,
        env_default: str | Sequence[str] | None = None,
        env_force: str | Sequence[str] | None = None,
        role: str = "general",
        mutability: ConfigMutability = ConfigMutability.STARTUP,
        validator: Validator | None = None,
        include_in_fingerprint: bool = True,
    ) -> None:
        """Defines a float option.

        Args:
            name: Option name.
            default: Default float.
            doc: User-facing description.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force effective value.
            role: Functional role/category for this option.
            mutability: Mutability policy.
            validator: Optional semantic validator.
            include_in_fingerprint: Fingerprint participation toggle.
        """
        self.define_option(
            name,
            default=default,
            doc=doc,
            value_type=float,
            parser=parse_float,
            validator=validator,
            env_default=env_default,
            env_force=env_force,
            role=role,
            mutability=mutability,
            include_in_fingerprint=include_in_fingerprint,
        )

    def define_string(
        self,
        name: str,
        *,
        default: str,
        doc: str,
        env_default: str | Sequence[str] | None = None,
        env_force: str | Sequence[str] | None = None,
        role: str = "general",
        mutability: ConfigMutability = ConfigMutability.STARTUP,
        validator: Validator | None = None,
        include_in_fingerprint: bool = True,
    ) -> None:
        """Defines a string option.

        Args:
            name: Option name.
            default: Default string value.
            doc: User-facing description.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force effective value.
            role: Functional role/category for this option.
            mutability: Mutability policy.
            validator: Optional semantic validator.
            include_in_fingerprint: Fingerprint participation toggle.
        """
        self.define_option(
            name,
            default=default,
            doc=doc,
            value_type=str,
            parser=lambda raw: str(raw).strip(),
            validator=validator,
            env_default=env_default,
            env_force=env_force,
            role=role,
            mutability=mutability,
            include_in_fingerprint=include_in_fingerprint,
        )

    def define_optional_string(
        self,
        name: str,
        *,
        default: str | None,
        doc: str,
        env_default: str | Sequence[str] | None = None,
        env_force: str | Sequence[str] | None = None,
        role: str = "general",
        mutability: ConfigMutability = ConfigMutability.STARTUP,
        validator: Validator | None = None,
        include_in_fingerprint: bool = True,
    ) -> None:
        """Defines an optional string option.

        Args:
            name: Option name.
            default: Default string or ``None``.
            doc: User-facing description.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force effective value.
            role: Functional role/category for this option.
            mutability: Mutability policy.
            validator: Optional semantic validator.
            include_in_fingerprint: Fingerprint participation toggle.
        """
        self.define_option(
            name,
            default=default,
            doc=doc,
            value_type=(str, type(None)),
            parser=parse_optional_string,
            validator=validator,
            env_default=env_default,
            env_force=env_force,
            role=role,
            mutability=mutability,
            include_in_fingerprint=include_in_fingerprint,
        )

    def define_optional_int(
        self,
        name: str,
        *,
        default: int | None,
        doc: str,
        env_default: str | Sequence[str] | None = None,
        env_force: str | Sequence[str] | None = None,
        role: str = "general",
        mutability: ConfigMutability = ConfigMutability.STARTUP,
        validator: Validator | None = None,
        include_in_fingerprint: bool = True,
    ) -> None:
        """Defines an optional integer option.

        Args:
            name: Option name.
            default: Default integer or ``None``.
            doc: User-facing description.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force effective value.
            role: Functional role/category for this option.
            mutability: Mutability policy.
            validator: Optional semantic validator.
            include_in_fingerprint: Fingerprint participation toggle.
        """
        self.define_option(
            name,
            default=default,
            doc=doc,
            value_type=(int, type(None)),
            parser=parse_optional_int,
            validator=validator,
            env_default=env_default,
            env_force=env_force,
            role=role,
            mutability=mutability,
            include_in_fingerprint=include_in_fingerprint,
        )

    def define_enum(
        self,
        name: str,
        *,
        default: str,
        values: Sequence[str],
        doc: str,
        case_sensitive: bool = False,
        env_default: str | Sequence[str] | None = None,
        env_force: str | Sequence[str] | None = None,
        role: str = "general",
        mutability: ConfigMutability = ConfigMutability.STARTUP,
        include_in_fingerprint: bool = True,
    ) -> None:
        """Defines an enum-like string option.

        Args:
            name: Option name.
            default: Default enum value.
            values: Allowed enum values.
            doc: User-facing description.
            case_sensitive: Whether value matching is case-sensitive.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force effective value.
            role: Functional role/category for this option.
            mutability: Mutability policy.
            include_in_fingerprint: Fingerprint participation toggle.
        """
        parser = self._enum_parser(values=values, case_sensitive=case_sensitive)
        normalized_default = parser(default)
        self.define_option(
            name,
            default=normalized_default,
            doc=doc,
            value_type=str,
            parser=parser,
            env_default=env_default,
            env_force=env_force,
            role=role,
            mutability=mutability,
            include_in_fingerprint=include_in_fingerprint,
        )

    def list_options(self) -> tuple[str, ...]:
        """Returns all registered option names.

        Returns:
            Sorted tuple of option names.
        """
        with self._lock:
            return tuple(sorted(self._options.keys()))

    @property
    def values(self) -> dict[str, Any]:
        """Returns effective global values for all options.

        Returns:
            Snapshot of global effective values, ignoring thread-local overrides.
        """
        return self.snapshot(thread_local=False)

    def options_by_mutability(self) -> dict[ConfigMutability, tuple[str, ...]]:
        """Groups options by mutability category.

        Returns:
            Mapping from mutability kind to option-name tuples.
        """
        grouped: dict[ConfigMutability, list[str]] = {
            ConfigMutability.IMMUTABLE: [],
            ConfigMutability.STARTUP: [],
            ConfigMutability.RUNTIME: [],
        }
        for name, state in self._options.items():
            grouped[state.spec.mutability].append(name)
        return {key: tuple(sorted(values)) for key, values in grouped.items()}

    def options_by_role(self) -> dict[str, tuple[str, ...]]:
        """Groups option names by role.

        Returns:
            Mapping from role name to sorted option-name tuples.
        """
        grouped: dict[str, list[str]] = {}
        for name, state in self._options.items():
            grouped.setdefault(state.spec.role, []).append(name)
        return {key: tuple(sorted(values)) for key, values in grouped.items()}

    def is_runtime_mutable(self, name: str) -> bool:
        """Reports whether an option can be mutated at runtime.

        Args:
            name: Option name.

        Returns:
            ``True`` if the option is runtime-mutable and not force-locked by
            environment; otherwise ``False``.
        """
        state = self._get_state(name)
        return (
            state.spec.mutability is ConfigMutability.RUNTIME
            and state.env_force_value is UNSET
        )

    @property
    def runtime_locked(self) -> bool:
        """Whether startup options are currently locked."""
        return self._runtime_locked

    def lock_runtime(self) -> None:
        """Locks startup-only options from further mutation."""
        with self._lock:
            self._runtime_locked = True

    def unlock_runtime_for_testing(self) -> None:
        """Unlocks startup options.

        This method is intentionally explicit and should only be used in tests.
        """
        with self._lock:
            self._runtime_locked = False

    def add_hook(
        self,
        name: str,
        hook: MutationHook,
        *,
        run_immediately: bool = False,
        thread_local: bool = False,
    ) -> None:
        """Registers a mutation hook for one option.

        Hooks are invoked whenever the effective value for the option changes.

        Args:
            name: Option name.
            hook: Callback receiving a :class:`ConfigMutation` event.
            run_immediately: If ``True``, executes ``hook`` once immediately
                with the current effective value.
            thread_local: When ``run_immediately`` is ``True``, controls whether
                the immediate event reflects thread-local state.
        """
        state = self._get_state(name)
        with self._lock:
            self._hooks[name].append(hook)
        if run_immediately:
            current = self.get(name, thread_local=thread_local)
            event = ConfigMutation(
                name=name,
                old_value=current,
                new_value=current,
                source=ConfigSource.DEFAULT,
                thread_local=thread_local,
                mutability=state.spec.mutability,
            )
            hook(event)

    def add_global_hook(self, hook: MutationHook) -> None:
        """Registers a global mutation hook.

        Args:
            hook: Callback receiving all :class:`ConfigMutation` events.
        """
        with self._lock:
            self._global_hooks.append(hook)

    def describe_option(self, name: str) -> dict[str, Any]:
        """Returns rich metadata for one option.

        Args:
            name: Option name.

        Returns:
            Dictionary with declaration metadata, current value, mutability,
            and environment binding information.
        """
        state = self._get_state(name)
        return {
            "name": state.spec.name,
            "doc": state.spec.doc,
            "role": state.spec.role,
            "default": state.spec.default,
            "mutability": state.spec.mutability.value,
            "env_default": state.spec.env_default,
            "env_force": state.spec.env_force,
            "env_default_value": state.env_default_value,
            "env_force_value": state.env_force_value,
            "effective_value": self.get(name),
            "runtime_locked": self._runtime_locked,
            "runtime_mutable": self.is_runtime_mutable(name),
            "include_in_fingerprint": state.spec.include_in_fingerprint,
        }

    @staticmethod
    def _format_table_cell(value: Any, *, max_width: int) -> str:
        """Formats one table cell for :meth:`show`.

        Args:
            value: Value to render.
            max_width: Maximum rendered width for one cell.

        Returns:
            String-safe cell value.
        """
        if value is UNSET:
            text = "<unset>"
        else:
            text = repr(value)
        text = " ".join(text.split())
        if len(text) <= max_width:
            return text
        if max_width <= 3:
            return text[:max_width]
        return f"{text[: max_width - 3]}..."

    def show(
        self,
        *,
        file: TextIO | None = None,
        include_current: bool = True,
        include_env: bool = False,
        thread_local: bool = True,
        max_cell_width: int = 48,
    ) -> str:
        """Renders and prints a tabular configuration summary.

        This method is intended for quick runtime introspection in notebooks,
        scripts, and debugging sessions.

        Args:
            file: Output stream. Defaults to :data:`sys.stdout`.
            include_current: Whether to include effective value and source
                columns.
            include_env: Whether to include environment binding columns.
            thread_local: Whether current value/source should include
                thread-local overrides when ``include_current`` is true.
            max_cell_width: Maximum width for each cell before truncation.

        Returns:
            The full rendered table text.
        """
        column_specs: list[tuple[str, str]] = [
            ("name", "name"),
            ("role", "role"),
            ("default", "default"),
            ("mutability", "mutability"),
            ("runtime_mutable", "runtime_mutable"),
        ]
        if include_current:
            column_specs.extend(
                [
                    ("effective", "effective"),
                    ("source", "source"),
                ]
            )
        if include_env:
            column_specs.extend(
                [
                    ("env_default", "env_default"),
                    ("env_force", "env_force"),
                ]
            )

        rows: list[dict[str, str]] = []
        for name in self.list_options():
            state = self._get_state(name)
            row: dict[str, str] = {
                "name": name,
                "role": state.spec.role,
                "default": self._format_table_cell(
                    state.spec.default,
                    max_width=max_cell_width,
                ),
                "mutability": state.spec.mutability.value,
                "runtime_mutable": ("yes" if self.is_runtime_mutable(name) else "no"),
            }
            if include_current:
                row["effective"] = self._format_table_cell(
                    self.get(name, thread_local=thread_local),
                    max_width=max_cell_width,
                )
                row["source"] = self.value_source(
                    name,
                    thread_local=thread_local,
                ).value
            if include_env:
                row["env_default"] = (
                    ", ".join(state.spec.env_default) if state.spec.env_default else "-"
                )
                row["env_force"] = (
                    ", ".join(state.spec.env_force) if state.spec.env_force else "-"
                )
            rows.append(row)

        widths: dict[str, int] = {}
        for key, title in column_specs:
            max_row = max((len(row[key]) for row in rows), default=0)
            widths[key] = max(len(title), max_row)

        separator = (
            "+-" + "-+-".join("-" * widths[key] for key, _ in column_specs) + "-+"
        )
        header = (
            "| "
            + " | ".join(title.ljust(widths[key]) for key, title in column_specs)
            + " |"
        )
        lines = [separator, header, separator]
        for row in rows:
            line = (
                "| "
                + " | ".join(row[key].ljust(widths[key]) for key, _ in column_specs)
                + " |"
            )
            lines.append(line)
        lines.append(separator)

        table = "\n".join(lines)
        stream = sys.stdout if file is None else file
        print(table, file=stream)
        return table

    def get(self, name: str, *, thread_local: bool = True) -> Any:
        """Reads the effective value of an option.

        Args:
            name: Option name.
            thread_local: Whether thread-local overrides are considered.

        Returns:
            Effective option value.
        """
        state = self._get_state(name)
        if state.env_force_value is not UNSET:
            return state.env_force_value
        if thread_local:
            overrides = self._thread_local_overrides(create=False)
            if overrides is not None and name in overrides:
                return overrides[name]
        if state.user_override is not UNSET:
            return state.user_override
        if state.env_default_value is not UNSET:
            return state.env_default_value
        return state.spec.default

    def value_source(self, name: str, *, thread_local: bool = True) -> ConfigSource:
        """Reports which source currently provides an option's value.

        Args:
            name: Option name.
            thread_local: Whether thread-local overrides are considered.

        Returns:
            Effective value source.
        """
        state = self._get_state(name)
        if state.env_force_value is not UNSET:
            return ConfigSource.ENV_FORCE
        if thread_local:
            overrides = self._thread_local_overrides(create=False)
            if overrides is not None and name in overrides:
                return ConfigSource.THREAD_LOCAL
        if state.user_override is not UNSET:
            return ConfigSource.USER
        if state.env_default_value is not UNSET:
            return ConfigSource.ENV_DEFAULT
        return ConfigSource.DEFAULT

    def set(
        self,
        name: str,
        value: Any,
        *,
        source: ConfigSource = ConfigSource.USER,
        thread_local: bool = False,
    ) -> Any:
        """Sets a configuration value.

        Args:
            name: Option name.
            value: New value.
            source: Mutation source label.
            thread_local: If ``True``, applies as thread-local override and
                requires runtime mutability.

        Returns:
            New effective value.

        Raises:
            ConfigError: If mutation is disallowed.
        """
        state = self._get_state(name)
        spec = state.spec
        parsed = spec.parse(value)

        with self._lock:
            if state.env_force_value is not UNSET:
                raise ConfigError(
                    f"Option '{name}' is forced by environment "
                    f"({state.spec.env_force}) and cannot be changed from Python."
                )

            if thread_local:
                if spec.mutability is not ConfigMutability.RUNTIME:
                    raise ConfigError(
                        f"Option '{name}' is {spec.mutability.value} and cannot be "
                        "overridden thread-locally."
                    )
                old_value = self.get(name, thread_local=True)
                overrides = self._thread_local_overrides(create=True)
                overrides[name] = parsed
                new_value = self.get(name, thread_local=True)
                self._maybe_emit(
                    name=name,
                    old_value=old_value,
                    new_value=new_value,
                    source=(
                        ConfigSource.THREAD_LOCAL
                        if source is ConfigSource.USER
                        else source
                    ),
                    thread_local=True,
                    mutability=spec.mutability,
                )
                return new_value

            self._assert_can_mutate_globally(spec)
            old_value = self.get(name, thread_local=False)
            state.user_override = parsed
            new_value = self.get(name, thread_local=False)
            self._maybe_emit(
                name=name,
                old_value=old_value,
                new_value=new_value,
                source=source,
                thread_local=False,
                mutability=spec.mutability,
            )
            return new_value

    def clear_override(
        self,
        name: str,
        *,
        source: ConfigSource = ConfigSource.RESET,
        thread_local: bool = False,
    ) -> Any:
        """Removes a user or thread-local override for an option.

        Args:
            name: Option name.
            source: Source label for emitted hook events.
            thread_local: If ``True``, clears thread-local override.

        Returns:
            New effective value.
        """
        state = self._get_state(name)
        with self._lock:
            if thread_local:
                overrides = self._thread_local_overrides(create=False)
                old_value = self.get(name, thread_local=True)
                if overrides is not None:
                    overrides.pop(name, None)
                new_value = self.get(name, thread_local=True)
                self._maybe_emit(
                    name=name,
                    old_value=old_value,
                    new_value=new_value,
                    source=source,
                    thread_local=True,
                    mutability=state.spec.mutability,
                )
                return new_value

            self._assert_can_mutate_globally(state.spec)
            old_value = self.get(name, thread_local=False)
            state.user_override = UNSET
            new_value = self.get(name, thread_local=False)
            self._maybe_emit(
                name=name,
                old_value=old_value,
                new_value=new_value,
                source=source,
                thread_local=False,
                mutability=state.spec.mutability,
            )
            return new_value

    def set_many(
        self,
        updates: Mapping[str, Any],
        *,
        source: ConfigSource = ConfigSource.USER,
        thread_local: bool = False,
    ) -> None:
        """Applies multiple updates.

        Args:
            updates: Mapping from option names to new values.
            source: Source label for mutation events.
            thread_local: Whether updates are thread-local.
        """
        for key, value in updates.items():
            self.set(key, value, source=source, thread_local=thread_local)

    @contextlib.contextmanager
    def patch(
        self,
        arg1: str | Mapping[str, Any] | None = None,
        arg2: Any = _MISSING,
        *,
        thread_local: bool = False,
        **kwargs: Any,
    ) -> Iterator[None]:
        """Temporarily patches one or more options.

        Supported forms:

        - ``patch("name", value)``
        - ``patch({"name": value, "other": value2})``
        - ``patch(name=value, other=value2)``

        Args:
            arg1: Key or mapping of updates.
            arg2: Value for the two-argument form.
            thread_local: Whether to patch thread-locally.
            **kwargs: Keyword-form updates.

        Yields:
            ``None``.
        """
        updates = self._normalize_patch_args(arg1=arg1, arg2=arg2, kwargs=kwargs)

        if thread_local:
            overrides = self._thread_local_overrides(create=True)
            previous: dict[str, Any] = {k: overrides.get(k, UNSET) for k in updates}
            try:
                self.set_many(updates, source=ConfigSource.PATCH, thread_local=True)
                yield
            finally:
                for key, old in previous.items():
                    if old is UNSET:
                        self.clear_override(
                            key, source=ConfigSource.PATCH, thread_local=True
                        )
                    else:
                        self.set(
                            key,
                            old,
                            source=ConfigSource.PATCH,
                            thread_local=True,
                        )
            return

        previous_global: dict[str, Any] = {}
        for key in updates:
            state = self._get_state(key)
            previous_global[key] = state.user_override
        try:
            self.set_many(updates, source=ConfigSource.PATCH, thread_local=False)
            yield
        finally:
            for key, old in previous_global.items():
                if old is UNSET:
                    self.clear_override(
                        key, source=ConfigSource.PATCH, thread_local=False
                    )
                else:
                    self.set(key, old, source=ConfigSource.PATCH, thread_local=False)

    def thread_local(
        self, name: str, value: Any
    ) -> contextlib.AbstractContextManager[None]:
        """Creates a thread-local context override for one runtime option.

        Args:
            name: Option name.
            value: Temporary value.

        Returns:
            Context manager that applies the override for the active thread.
        """
        return self.patch(name, value, thread_local=True)

    def snapshot(self, *, thread_local: bool = True) -> dict[str, Any]:
        """Exports a snapshot of effective option values.

        Args:
            thread_local: Whether to include thread-local overrides.

        Returns:
            Mapping of option names to effective values.
        """
        return {
            name: self.get(name, thread_local=thread_local)
            for name in self.list_options()
        }

    def user_overrides(self, *, thread_local: bool = False) -> dict[str, Any]:
        """Returns explicitly overridden values.

        Args:
            thread_local: If ``True``, returns current thread-local overrides.
                Otherwise returns global user overrides.

        Returns:
            Mapping of overridden values.
        """
        if thread_local:
            overrides = self._thread_local_overrides(create=False)
            return {} if overrides is None else dict(overrides)
        out: dict[str, Any] = {}
        for name, state in self._options.items():
            if state.user_override is not UNSET:
                out[name] = state.user_override
        return out

    def fingerprint(self, *, thread_local: bool = False) -> str:
        """Computes a stable fingerprint of selected config values.

        This is primarily useful for compile cache keying and diagnostics.

        Args:
            thread_local: Whether to include thread-local overrides.

        Returns:
            Hex digest string.
        """
        values: dict[str, Any] = {}
        for name, state in self._options.items():
            if not state.spec.include_in_fingerprint:
                continue
            values[name] = self.get(name, thread_local=thread_local)
        payload = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _normalize_patch_args(
        self,
        *,
        arg1: str | Mapping[str, Any] | None,
        arg2: Any,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Normalizes supported patch call forms."""
        if arg1 is None:
            if arg2 is not _MISSING:
                raise TypeError(
                    "Second positional argument is valid only when first argument is a key string."
                )
            return dict(kwargs)

        if arg2 is not _MISSING:
            if kwargs:
                raise TypeError("Cannot mix positional and keyword patch arguments.")
            if not isinstance(arg1, str):
                raise TypeError(
                    "First positional argument must be a key string in two-argument form."
                )
            return {arg1: arg2}

        if not isinstance(arg1, Mapping):
            raise TypeError("Single-argument patch form expects a mapping of updates.")
        if kwargs:
            raise TypeError("Cannot combine mapping-form patch with keyword arguments.")
        return dict(arg1)

    def _read_env_value(
        self,
        *,
        option: ConfigOption[Any],
        env_names: Sequence[str],
    ) -> Any | _UnsetType:
        """Reads first present environment value for an option."""
        for env_name in env_names:
            if env_name not in os.environ:
                continue
            raw_value = os.environ[env_name]
            try:
                return option.parse(raw_value)
            except ConfigValidationError as exc:
                raise ConfigValidationError(
                    f"Invalid value from environment variable {env_name!r} for "
                    f"option {option.name!r}: {raw_value!r}."
                ) from exc
        return UNSET

    def _thread_local_overrides(
        self, *, create: bool
    ) -> MutableMapping[str, Any] | None:
        """Returns thread-local override dictionary."""
        overrides = getattr(self._thread_local, "overrides", None)
        if overrides is None and create:
            overrides = {}
            self._thread_local.overrides = overrides
        return overrides

    def _get_state(self, name: str) -> _OptionState[Any]:
        """Returns state object for an option.

        Args:
            name: Option name.

        Returns:
            Option state.

        Raises:
            UnknownOptionError: If option does not exist.
        """
        try:
            return self._options[name]
        except KeyError as exc:
            raise UnknownOptionError(f"Unknown option '{name}'.") from exc

    def _assert_can_mutate_globally(self, spec: ConfigOption[Any]) -> None:
        """Checks global mutability constraints."""
        if spec.mutability is ConfigMutability.IMMUTABLE:
            raise ConfigError(
                f"Option '{spec.name}' is immutable and cannot be changed."
            )
        if spec.mutability is ConfigMutability.STARTUP and self._runtime_locked:
            raise ConfigError(
                f"Option '{spec.name}' is startup-only and runtime is locked. "
                "Set this option before calling lock_runtime()."
            )

    def _maybe_emit(
        self,
        *,
        name: str,
        old_value: Any,
        new_value: Any,
        source: ConfigSource,
        thread_local: bool,
        mutability: ConfigMutability,
    ) -> None:
        """Emits mutation hooks if value changed."""
        if old_value == new_value:
            return
        event = ConfigMutation(
            name=name,
            old_value=old_value,
            new_value=new_value,
            source=source,
            thread_local=thread_local,
            mutability=mutability,
        )
        hooks = list(self._hooks.get(name, ()))
        global_hooks = list(self._global_hooks)
        for hook in hooks:
            hook(event)
        for hook in global_hooks:
            hook(event)


def _register_default_options(cfg: ConfigManager) -> None:
    """Registers neuraLQX's default option set.

    Args:
        cfg: Configuration manager to initialize.
    """
    # DType and precision policy.
    cfg.define_enum(
        "dtype_real",
        default="float64",
        values=("float16", "bfloat16", "float32", "float64"),
        doc="Default real dtype used for model and graph feature tensors.",
        env_default="NEURALQX_DTYPE_REAL",
        role="dtype",
        mutability=ConfigMutability.STARTUP,
    )
    cfg.define_enum(
        "dtype_complex",
        default="complex128",
        values=("complex64", "complex128"),
        doc="Default complex dtype for wavefunction outputs and matrix elements.",
        env_default="NEURALQX_DTYPE_COMPLEX",
        role="dtype",
        mutability=ConfigMutability.STARTUP,
    )
    cfg.define_enum(
        "dtype_index",
        default="int32",
        values=("int32", "int64"),
        doc="Index dtype for graph connectivity arrays.",
        env_default="NEURALQX_DTYPE_INDEX",
        role="dtype",
        mutability=ConfigMutability.STARTUP,
    )
    cfg.define_bool(
        "enable_x64",
        default=True,
        doc=(
            "Enables higher precision defaults where supported. "
            "Intended to align with JAX high-precision execution modes."
        ),
        env_default="NEURALQX_ENABLE_X64",
        role="precision",
        mutability=ConfigMutability.STARTUP,
    )

    # Runtime and compilation behavior.
    cfg.define_enum(
        "backend_target",
        default="auto",
        values=("auto", "cpu", "gpu", "tpu"),
        doc="Preferred execution backend target.",
        env_default="NEURALQX_BACKEND_TARGET",
        env_force="NEURALQX_FORCE_BACKEND_TARGET",
        role="runtime",
        mutability=ConfigMutability.STARTUP,
    )
    cfg.define_bool(
        "jit_enabled",
        default=True,
        doc="Global toggle for compiled execution paths.",
        env_default="NEURALQX_JIT_ENABLED",
        role="runtime",
        mutability=ConfigMutability.RUNTIME,
    )
    cfg.define_enum(
        "profile_level",
        default="off",
        values=("off", "basic", "trace", "full"),
        doc="Profiling detail level for runtime diagnostics and performance traces.",
        env_default="NEURALQX_PROFILE_LEVEL",
        role="runtime",
        mutability=ConfigMutability.RUNTIME,
    )
    cfg.define_enum(
        "log_level",
        default="warning",
        values=("critical", "error", "warning", "info", "debug"),
        doc="Logging verbosity level for the neuraLQX logger hierarchy.",
        env_default="NEURALQX_LOG_LEVEL",
        role="runtime",
        mutability=ConfigMutability.RUNTIME,
        include_in_fingerprint=False,
    )
    cfg.define_bool(
        "compile_cache_enabled",
        default=True,
        doc="Enables on-disk and in-memory compilation artifact caching.",
        env_default="NEURALQX_COMPILE_CACHE_ENABLED",
        env_force="NEURALQX_FORCE_COMPILE_CACHE_ENABLED",
        role="compilation",
        mutability=ConfigMutability.RUNTIME,
    )
    cfg.define_optional_string(
        "compile_cache_dir",
        default=None,
        doc="Directory for persisted compilation artifacts.",
        env_default="NEURALQX_COMPILE_CACHE_DIR",
        role="compilation",
        mutability=ConfigMutability.STARTUP,
    )

    # JAX distributed runtime startup.
    cfg.define_bool(
        "distributed",
        default=False,
        doc=(
            "Initializes JAX's multi-process distributed runtime during "
            "neuraLQX startup. Process launch is still handled externally."
        ),
        env_default=("NEURALQX_DISTRIBUTED", "NEURALQX_DISTRIBUTED_ENABLED"),
        env_force=("NEURALQX_FORCE_DISTRIBUTED", "NEURALQX_FORCE_DISTRIBUTED_ENABLED"),
        role="distributed",
        mutability=ConfigMutability.STARTUP,
    )
    cfg.define_bool(
        "distributed_auto_initialize",
        default=True,
        doc="Automatically calls jax.distributed.initialize() when distributed mode is enabled.",
        env_default="NEURALQX_DISTRIBUTED_AUTO_INITIALIZE",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
    )
    cfg.define_optional_string(
        "distributed_coordinator_address",
        default=None,
        doc="Coordinator host:port passed to jax.distributed.initialize().",
        env_default="NEURALQX_DISTRIBUTED_COORDINATOR_ADDRESS",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
    )
    cfg.define_optional_int(
        "distributed_num_processes",
        default=None,
        doc="Total process count passed to jax.distributed.initialize().",
        env_default="NEURALQX_DISTRIBUTED_NUM_PROCESSES",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
        validator=lambda value: (
            None if value is None else positive_int_validator(value)
        ),
    )
    cfg.define_optional_int(
        "distributed_process_id",
        default=None,
        doc="Current process ID passed to jax.distributed.initialize().",
        env_default="NEURALQX_DISTRIBUTED_PROCESS_ID",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
        validator=lambda value: (
            None if value is None else non_negative_int_validator(value)
        ),
    )
    cfg.define_option(
        "distributed_local_device_ids",
        default=None,
        doc="Optional comma-separated local device IDs for jax.distributed.initialize().",
        value_type=(tuple, type(None)),
        parser=parse_optional_int_tuple,
        validator=non_negative_int_tuple_validator,
        env_default="NEURALQX_DISTRIBUTED_LOCAL_DEVICE_IDS",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
    )
    cfg.define_int(
        "distributed_initialization_timeout",
        default=300,
        doc="Distributed initialization timeout in seconds.",
        env_default="NEURALQX_DISTRIBUTED_INITIALIZATION_TIMEOUT",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
        validator=positive_int_validator,
    )

    # JAX mesh and sharding policy.
    # TODO: do we also allow this to be runtime mutable? There could be use cases for it,
    #       and in principle there is no obstruction to switching the parallel execution
    #       model in runtime, assuming upstream jaxPP is okay with that
    #
    # TODO: if the above is adopted, all *_enabled flags below should have hooks to update
    #       themselves once the sharding policy is changed as neuralqx.jax depends frequently
    #       on these flags and not directly querying the sharding_axes envvar.
    #
    # TODO: Have a more explicit docs maybe? Also verify if samples sharding is always on,
    #       if not, then also include that in the docs. If not, also we need to understand
    #       that behaviour, and if it is truly mirrored in the code or not, and if we
    #       actually want that or not, not all users do not need complete freedom but others
    #       should have the ability to do so if they want. Right now, samples are always
    #       chosen by default.
    cfg.define_option(
        "sharding_axes",
        default=("samples",),
        doc=(
            "High-level sharding policy. Use 'samples' for sample-only sharding, "
            "'parameters' or 'samples,parameters' for sample+parameter sharding, "
            "and 'all' or 'samples,parameters,operators' for the full mesh."
        ),
        value_type=tuple,
        parser=parse_sharding_axes,
        env_default=("NEURALQX_SHARDING", "NEURALQX_SHARDING_AXES"),
        role="distributed",
        mutability=ConfigMutability.STARTUP,
    )
    active_sharding_axes = cfg.get("sharding_axes", thread_local=False)
    cfg.define_bool(
        "sample_sharding_enabled",
        default="samples" in active_sharding_axes,
        doc=(
            "Shards Monte Carlo samples/chains across the sample mesh axis. "
            "Defaults to whether 'samples' is active in sharding_axes."
        ),
        env_default="NEURALQX_SAMPLE_SHARDING_ENABLED",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
    )
    cfg.define_bool(
        "parameter_sharding_enabled",
        default="parameters" in active_sharding_axes,
        doc=(
            "Shards model parameters across the parameter mesh axis. "
            "Defaults to whether 'parameters' is active in sharding_axes."
        ),
        env_default="NEURALQX_PARAMETER_SHARDING_ENABLED",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
    )
    cfg.define_bool(
        "operator_sharding_enabled",
        default="operators" in active_sharding_axes,
        doc=(
            "Shards independent operator terms in operator lists and SumOperator "
            "wrappers across the operator mesh axis. Defaults to whether "
            "'operators' is active in sharding_axes."
        ),
        env_default="NEURALQX_OPERATOR_SHARDING_ENABLED",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
    )
    cfg.define_bool(
        "operator_streaming_enabled",
        default=False,
        doc=(
            "Streams/chunks connected-component generation inside one operator. "
            "This bounds peak memory independently of operator-term sharding."
        ),
        env_default="NEURALQX_OPERATOR_STREAMING_ENABLED",
        role="distributed",
        mutability=ConfigMutability.RUNTIME,
    )
    cfg.define_int(
        "operator_streaming_chunk_size",
        default=1024,
        doc=(
            "Number of flattened input states processed per operator streaming chunk. "
            "This controls memory use when operator_streaming_enabled is active."
        ),
        env_default="NEURALQX_OPERATOR_STREAMING_CHUNK_SIZE",
        role="distributed",
        mutability=ConfigMutability.RUNTIME,
        validator=positive_int_validator,
    )
    cfg.define_string(
        "mesh_sample_axis_name",
        default="samples",
        doc="Logical mesh axis name for sample/chain sharding.",
        env_default="NEURALQX_MESH_SAMPLE_AXIS_NAME",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
        validator=non_empty_string_validator,
    )
    cfg.define_string(
        "mesh_parameter_axis_name",
        default="parameters",
        doc="Logical mesh axis name for model-parameter sharding.",
        env_default="NEURALQX_MESH_PARAMETER_AXIS_NAME",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
        validator=non_empty_string_validator,
    )
    cfg.define_string(
        "mesh_operator_axis_name",
        default="operators",
        doc="Logical mesh axis name for operator-work sharding.",
        env_default="NEURALQX_MESH_OPERATOR_AXIS_NAME",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
        validator=non_empty_string_validator,
    )
    cfg.define_optional_int(
        "mesh_sample_axis_size",
        default=None,
        doc=(
            "Device count for the sample mesh axis. "
            "None lets mesh construction infer the remaining devices."
        ),
        env_default="NEURALQX_MESH_SAMPLE_AXIS_SIZE",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
        validator=lambda value: (
            None if value is None else positive_int_validator(value)
        ),
    )
    cfg.define_optional_int(
        "mesh_parameter_axis_size",
        default=None,
        doc=(
            "Device count for the parameter mesh axis. None lets mesh construction "
            "infer the size when parameter sharding is active."
        ),
        env_default="NEURALQX_MESH_PARAMETER_AXIS_SIZE",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
        validator=lambda value: (
            None if value is None else positive_int_validator(value)
        ),
    )
    cfg.define_optional_int(
        "mesh_operator_axis_size",
        default=None,
        doc=(
            "Device count for the operator mesh axis. None lets mesh construction "
            "infer the size when operator sharding is active."
        ),
        env_default="NEURALQX_MESH_OPERATOR_AXIS_SIZE",
        role="distributed",
        mutability=ConfigMutability.STARTUP,
        validator=lambda value: (
            None if value is None else positive_int_validator(value)
        ),
    )

    # Experimental and feature gating.
    cfg.define_bool(
        "experimental",
        default=False,
        doc="Master switch for experimental APIs and code paths.",
        env_default=("NEURALQX_EXPERIMENTAL", "NEURALQX_EXPERIMENTAL_ENABLED"),
        env_force="NEURALQX_FORCE_EXPERIMENTAL",
        role="experimental",
        mutability=ConfigMutability.RUNTIME,
    )
    cfg.define_option(
        "experimental_features",
        default=(),
        doc=(
            "Comma-separated feature gates for opt-in experimental behavior "
            "(e.g. 'new_operator_ir,pallas_branch_emit')."
        ),
        value_type=tuple,
        parser=parse_csv_tuple,
        env_default="NEURALQX_EXPERIMENTAL_FEATURES",
        role="experimental",
        mutability=ConfigMutability.RUNTIME,
    )
    cfg.define_bool(
        "allow_ffi_kernels",
        default=False,
        doc="Permits loading optional native FFI kernel extensions.",
        env_default="NEURALQX_ALLOW_FFI_KERNELS",
        role="kernel",
        mutability=ConfigMutability.STARTUP,
    )

    mesh_axis_names = (
        cfg.get("mesh_sample_axis_name", thread_local=False),
        cfg.get("mesh_parameter_axis_name", thread_local=False),
        cfg.get("mesh_operator_axis_name", thread_local=False),
    )
    if len(set(mesh_axis_names)) != len(mesh_axis_names):
        raise ConfigValidationError(
            "Mesh axis names must be unique. "
            f"Received sample={mesh_axis_names[0]!r}, "
            f"parameter={mesh_axis_names[1]!r}, "
            f"operator={mesh_axis_names[2]!r}."
        )

    manual_distributed_args = {
        "distributed_coordinator_address": cfg.get(
            "distributed_coordinator_address", thread_local=False
        ),
        "distributed_num_processes": cfg.get(
            "distributed_num_processes", thread_local=False
        ),
        "distributed_process_id": cfg.get("distributed_process_id", thread_local=False),
    }
    provided_manual_args = {
        name: value
        for name, value in manual_distributed_args.items()
        if value is not None
    }
    if provided_manual_args and len(provided_manual_args) != len(
        manual_distributed_args
    ):
        missing = tuple(
            name for name, value in manual_distributed_args.items() if value is None
        )
        raise ConfigValidationError(
            "Manual distributed initialization requires coordinator address, "
            f"process count, and process ID together. Missing {missing}."
        )
    num_processes = manual_distributed_args["distributed_num_processes"]
    process_id = manual_distributed_args["distributed_process_id"]
    if (
        num_processes is not None
        and process_id is not None
        and process_id >= num_processes
    ):
        raise ConfigValidationError(
            "Invalid distributed initialization: process ID must be less than "
            f"process count. Received process_id={process_id}, "
            f"num_processes={num_processes}."
        )


def _register_default_hooks(cfg: ConfigManager) -> None:
    """Registers built-in runtime hooks for default options.

    Args:
        cfg: Configuration manager to configure.
    """

    def _log_level_hook(event: ConfigMutation) -> None:
        level_name = str(event.new_value).upper()
        level = getattr(logging, level_name, None)
        if isinstance(level, int):
            logging.getLogger("neuralqx").setLevel(level)
            _LOGGER.debug(
                "Applied log level %s from %s mutation.",
                level_name,
                event.source.value,
            )

    def _experimental_hook(event: ConfigMutation) -> None:
        if bool(event.new_value):
            _LOGGER.warning(
                "Experimental mode enabled. APIs and behaviors may change without notice."
            )

    def _jax_x64_hook(event: ConfigMutation) -> None:
        import jax

        enabled = bool(event.new_value)
        jax.config.update("jax_enable_x64", enabled)
        _LOGGER.debug(
            "Applied jax_enable_x64=%s from %s mutation.",
            enabled,
            event.source.value,
        )

    def _jax_distributed_hook(event: ConfigMutation) -> None:
        if not bool(cfg.get("distributed", thread_local=False)):
            return
        if not bool(cfg.get("distributed_auto_initialize", thread_local=False)):
            return

        import jax

        if jax.distributed.is_initialized():
            return

        try:
            jax.distributed.initialize(
                coordinator_address=cfg.get(
                    "distributed_coordinator_address",
                    thread_local=False,
                ),
                num_processes=cfg.get("distributed_num_processes", thread_local=False),
                process_id=cfg.get("distributed_process_id", thread_local=False),
                local_device_ids=cfg.get(
                    "distributed_local_device_ids",
                    thread_local=False,
                ),
                initialization_timeout=cfg.get(
                    "distributed_initialization_timeout",
                    thread_local=False,
                ),
            )
        except Exception as exc:
            raise ConfigError(
                "Failed to initialize JAX distributed runtime. Either run in a "
                "JAX auto-detectable environment, or provide "
                "NEURALQX_DISTRIBUTED_COORDINATOR_ADDRESS, "
                "NEURALQX_DISTRIBUTED_NUM_PROCESSES, and "
                "NEURALQX_DISTRIBUTED_PROCESS_ID."
            ) from exc

        _LOGGER.info(
            "Initialized JAX distributed runtime: process_index=%s, process_count=%s.",
            jax.process_index(),
            jax.process_count(),
        )

    cfg.add_hook("log_level", _log_level_hook, run_immediately=True)
    cfg.add_hook("experimental", _experimental_hook, run_immediately=False)
    cfg.add_hook("enable_x64", _jax_x64_hook, run_immediately=True)
    cfg.add_hook("distributed", _jax_distributed_hook, run_immediately=True)
    cfg.add_hook(
        "distributed_auto_initialize",
        _jax_distributed_hook,
        run_immediately=True,
    )


def create_default_config() -> ConfigManager:
    """Builds a fully initialized configuration manager.

    Returns:
        Config manager pre-loaded with standard neuraLQX options and hooks.
    """
    manager = ConfigManager()
    _register_default_options(manager)
    _register_default_hooks(manager)
    return manager


config = create_default_config()


def get_static(name: str) -> Any:
    """Returns a registered package-wide static value."""
    return config.get_static(name)


__all__ = [
    "ConfigError",
    "ConfigManager",
    "ConfigMutation",
    "ConfigMutability",
    "ConfigOption",
    "ConfigSource",
    "ConfigValidationError",
    "ReadOnlyDict",
    "UnknownOptionError",
    "config",
    "create_default_config",
    "get_static",
    "non_negative_int_validator",
    "non_negative_int_tuple_validator",
    "non_empty_string_validator",
    "parse_bool",
    "parse_csv_tuple",
    "parse_float",
    "parse_int",
    "parse_optional_int",
    "parse_optional_int_tuple",
    "parse_optional_string",
    "parse_sharding_axes",
    "positive_int_validator",
]
