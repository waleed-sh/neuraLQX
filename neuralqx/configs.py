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

"""Configuration management for neuraLQX.

This module provides a typed, environment-aware configuration system.

All neuraLQX options are bound to environment variables with the prefix
``NQX_`` and are matched case-insensitively for convenience.

The public singleton exported by this module is :data:`cfg`.
"""

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
from textwrap import dedent

from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import Mapping
from typing import MutableMapping
from typing import Sequence
from typing import TextIO
from typing import TypeVar
from typing import cast

from tabulate import tabulate

T = TypeVar("T")

_LOGGER = logging.getLogger("neuralqx.configs")


def new_error_content(
    message: str,
    *args,
    **kwargs,
):
    return (
        f"{dedent(message)}"
        f"\n"
        f"\n============================================================================="
        f"\n"
        f"You can find a list of all neuraLQX errors and warnings including their "
        f"\ndetailed explanations at:"
        f"\n\t https://neuralqx.readthedocs.io/en/latest/guides/errors.html."
        f"\n============================================================================="
        f"\n"
    )


class ConfigError(Exception):
    """Base class for all neuraLQX configuration errors."""

    def __init__(self, msg: str):
        super().__init__(new_error_content(msg))


class UnknownOptionError(ConfigError):
    """Raised when accessing or mutating an unknown configuration option."""


class ConfigValidationError(ConfigError):
    """Raised when a configuration value fails parsing or validation."""


class ReadOnlyDict(Dict[str, Any]):
    """An immutable dictionary used for package-wide static variables."""

    def __setitem__(self, key, value):
        raise TypeError("This configuration dictionary is read-only.")

    def __delitem__(self, key):
        raise TypeError("This configuration dictionary is read-only.")


class ConfigMutability(str, enum.Enum):
    """
    Defines when a configuration option may be changed.

    Attributes:
        IMMUTABLE: Cannot be changed at runtime via the Python API.
            Value is determined exclusively by defaults and environment.
        STARTUP: Can be changed programmatically only before runtime is locked
            via :meth:`ConfigManager.lock_runtime`. Suitable for options that
            influence compilation or process topology.
        RUNTIME: Can be changed at any time and supports thread-local overrides.
    """

    IMMUTABLE = "immutable"
    STARTUP = "startup"
    RUNTIME = "runtime"


class ConfigSource(str, enum.Enum):
    """
    Describes where an observed configuration value originated.

    Attributes:
        DEFAULT: Value originated from the option's built-in default.
        ENV_DEFAULT: Value originated from an environment *default* variable.
        ENV_FORCE: Value originated from an environment *force* variable.
        USER: Value originated from a direct programmatic update.
        PATCH: Value originated from a temporary :meth:`ConfigManager.patch`
            context.
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
    """Sentinel for values that have not been provided."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"


UNSET = _UnsetType()
_MISSING = _UnsetType()


@dataclass(frozen=True, slots=True)
class ConfigMutation:
    """
    Represents a single effective configuration mutation event.

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


MutationHook = Callable[[ConfigMutation], None]
Parser = Callable[[Any], Any]
Validator = Callable[[Any], None]


@dataclass(frozen=True, slots=True)
class ConfigOption(Generic[T]):
    """
    Static declaration of one configuration option.

    Attributes:
        name: Public option name.
        default: Built-in default value.
        doc: User-facing help text.
        value_type: Runtime type contract after parsing. May be a single type
            or a tuple of accepted types.
        parser: Optional parser that converts raw inputs into canonical values.
            Applied for both environment variables and runtime updates.
        validator: Optional semantic validator run after parsing.
        env_default: Ordered environment variable names used to override the
            built-in default. The first present variable wins.
        env_force: Ordered environment variable names that force the effective
            value and prevent runtime mutation. The first present variable wins.
        role: Functional role/category for grouping this option.
        mutability: Option mutability policy.
        include_in_fingerprint: Whether this option participates in
            :meth:`ConfigManager.fingerprint`.
    """

    name: str
    default: T
    doc: str
    value_type: type[Any] | tuple[type[Any], ...] | None = None
    parser: Parser | None = None
    validator: Validator | None = None
    env_default: tuple[str, ...] = ()
    env_force: tuple[str, ...] = ()
    role: str = "general"
    mutability: ConfigMutability = ConfigMutability.STARTUP
    include_in_fingerprint: bool = True

    def parse(self, raw_value: Any) -> T:
        """
        Parses and validates a raw option value.

        Args:
            raw_value: Input value from environment or user code.

        Returns:
            Parsed and validated value.

        Raises:
            ConfigValidationError: If parsing, type-checking, or validation
                fails.
        """
        parsed = self.parser(raw_value) if self.parser is not None else raw_value
        if self.value_type is not None and not isinstance(parsed, self.value_type):
            raise ConfigValidationError(
                f"Option '{self.name}' expects value of type {self.value_type}, "
                f"got {type(parsed)} with value {parsed!r}."
            )
        if self.validator is not None:
            self.validator(parsed)
        return cast(T, parsed)


@dataclass(slots=True)
class _OptionState(Generic[T]):
    """Mutable runtime state for a single option."""

    spec: ConfigOption[T]
    env_default_value: T | _UnsetType = UNSET
    env_force_value: T | _UnsetType = UNSET
    user_override: T | _UnsetType = UNSET


#
#
#   Standalone parsers


def parse_bool(value: Any) -> bool:
    """
    Parses a boolean with permissive string handling.

    Args:
        value: Value to parse. Accepts ``bool``, ``int`` 0/1, or common
            string representations (``"true"``, ``"yes"``, ``"1"``, …).

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
    """
    Parses an integer value.

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
    """
    Parses a float value.

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
    """
    Parses an optional integer.

    Args:
        value: Value to parse.

    Returns:
        Parsed integer or ``None``.

    Raises:
        ConfigValidationError: If value is neither empty/none-like nor a valid
            integer.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return parse_int(value)


def parse_optional_string(value: Any) -> str | None:
    """
    Parses an optional string.

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
    """
    Parses a comma-separated value list into a tuple of strings.

    Args:
        value: CSV string or sequence of strings.

    Returns:
        Tuple of normalised, non-empty strings.

    Raises:
        ConfigValidationError: If value is not parseable.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return tuple(item for item in items if item)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
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


#
#
#   Standalone validators


def positive_int_validator(value: int) -> None:
    """
    Validates strictly positive integers.

    Args:
        value: Integer value.

    Raises:
        ConfigValidationError: If value is not strictly positive.
    """
    if value <= 0:
        raise ConfigValidationError(f"Expected positive integer, got {value}.")


def non_negative_int_validator(value: int) -> None:
    """
    Validates non-negative integers.

    Args:
        value: Integer value.

    Raises:
        ConfigValidationError: If value is negative.
    """
    if value < 0:
        raise ConfigValidationError(f"Expected non-negative integer, got {value}.")


def positive_float_validator(value: float) -> None:
    """
    Validates strictly positive floats.

    Args:
        value: Float value.

    Raises:
        ConfigValidationError: If value is not strictly positive.
    """
    if value <= 0:
        raise ConfigValidationError(f"Expected positive float, got {value}.")


class ConfigManager:
    """
    Typed, hookable, and singleton configuration manager for neuraLQX.

    Options are declared with typed helpers (e.g. :meth:`define_bool`,
    :meth:`define_enum`) and bound to ``NQX_``-prefixed environment variables
    which are matched case-insensitively.

    Effective value precedence (highest to lowest):

    1. Forced environment value (``env_force``)
    2. Thread-local override (:attr:`ConfigMutability.RUNTIME` options only)
    3. User global override
    4. Environment default value (``env_default``)
    5. Built-in option default

    Notes:
        - Forced environment values are intentionally immutable from Python.
        - :attr:`ConfigMutability.STARTUP` options become immutable once
          :meth:`lock_runtime` is called.
        - :attr:`ConfigMutability.RUNTIME` options can be changed globally or
          per-thread via :meth:`patch` and :meth:`thread_local_override`.
    """

    _instance = None
    """Holds the singleton instance."""

    _class_lock = threading.Lock()
    """Lock used exclusively for double-checked singleton initialisation."""

    PREFIX = "NQX_"
    """Standard neuraLQX prefix for environment variables."""

    _PROFILING_EXTRAS: list[str] = ["PROFILE_JAX_ANNOTATIONS"]
    """Additional profiling env-var suffixes accepted for backward compatibility."""

    def __new__(cls) -> "ConfigManager":
        """Ensures only one instance exists across the package."""
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialises the singleton. Subsequent calls are no-ops."""
        if self._initialized:
            return

        object.__setattr__(self, "_options", {})
        object.__setattr__(self, "_hooks", {})
        object.__setattr__(self, "_global_hooks", [])
        object.__setattr__(self, "_runtime_locked", False)
        object.__setattr__(self, "_thread_local_store", threading.local())
        object.__setattr__(self, "_lock", threading.RLock())

        self._register_statics()
        self._register_default_configs()
        self._sync_external_envars()
        self._warn_unused()

        self._initialized = True

    def __repr__(self) -> str:
        return (
            "ConfigManager("
            f"options={len(self._options)}, "
            f"runtime_locked={self._runtime_locked})"
        )

    def __getattr__(self, name: str) -> Any:
        """
        Reads a registered option as an attribute.

        Args:
            name: Option name.

        Returns:
            Effective option value.

        Raises:
            AttributeError: If *name* is not a registered option.
        """
        if "_options" in self.__dict__ and name in self._options:
            return self.get(name)
        raise AttributeError(f"{self.__class__.__name__} has no option '{name}'.")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets a registered option as an attribute.

        Args:
            name: Option name.
            value: New value.

        Raises:
            AttributeError: If *name* is not a registered option or internal
                attribute.
        """
        if name.startswith("_") or name in {
            "register_option",
            "set",
            "get",
            "patch",
            "thread_local_override",
        }:
            object.__setattr__(self, name, value)
            return
        if "_options" in self.__dict__ and name in self._options:
            self.set(name, value)
            return
        raise AttributeError(f"{self.__class__.__name__} has no option '{name}'.")

    def _register_statics(self) -> None:
        """Builds the read-only package-wide static variable dictionary."""
        date_str = datetime.now().strftime("%Y%m%d")
        _s = {
            "Errors Directory": "https://neuralqx.readthedocs.io/en/latest/guides/errors.html",
            "Cache Directory": os.path.join(os.getcwd(), ".neuralqx_cache"),
            "Leach Directory": os.path.join(
                os.getcwd(), ".neuralqx_logs", f"neuralqx_{date_str}"
            ),
            "Profiling Directory": os.path.join(
                os.getcwd(), ".neuralqx_profiling", f"neuralqx_{date_str}"
            ),
        }
        object.__setattr__(self, "_statics", ReadOnlyDict(_s))

    @property
    def statics(self) -> ReadOnlyDict:
        """Read-only dictionary of package-wide static variables."""
        return self._statics

    def get_static(self, name: str) -> Any:
        """
        Retrieves a static variable by name.

        Args:
            name: Static variable name.

        Returns:
            Static variable value.

        Raises:
            KeyError: If *name* is not a registered static variable.
        """
        if name not in self._statics:
            raise KeyError(
                f"You have tried to access a static variable `{name}` which does not exist."
            )
        return self._statics[name]

    @staticmethod
    def _normalize_env_names(
        names: str | Sequence[str] | None,
    ) -> tuple[str, ...]:
        """
        Normalises environment variable name declarations.

        Args:
            names: Single name, sequence of names, or ``None``.

        Returns:
            Tuple of normalised environment variable names.

        Raises:
            ConfigValidationError: If any name is not a string.
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

            def _cs_parser(raw: Any) -> str:
                if not isinstance(raw, str):
                    raise ConfigValidationError(
                        f"Expected string enum value in {allowed}, got {raw!r}."
                    )
                if raw not in allowed_set:
                    raise ConfigValidationError(
                        f"Invalid enum value {raw!r}, expected one of {allowed}."
                    )
                return raw

            return _cs_parser

        canonical = tuple(value.lower() for value in values)
        canonical_set = set(canonical)

        def _ci_parser(raw: Any) -> str:
            if not isinstance(raw, str):
                raise ConfigValidationError(
                    f"Expected string enum value in {canonical}, got {raw!r}."
                )
            normalized = raw.strip().lower()
            if normalized not in canonical_set:
                raise ConfigValidationError(
                    f"Invalid enum value {raw!r}, expected one of {canonical}."
                )
            return normalized

        return _ci_parser

    def register_option(self, option: ConfigOption[Any]) -> None:
        """
        Registers a fully specified option.

        Args:
            option: Option declaration.

        Raises:
            ConfigValidationError: If the option name is invalid or duplicated.
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

            # write the canonical effective value back to os.environ
            # so that downstream tools and the case-insensitive lookup find a
            # consistently cased key.
            effective = self._direct_effective_value(state)
            os.environ[self.PREFIX + option.name] = str(effective)

    def _direct_effective_value(self, state: _OptionState[Any]) -> Any:
        """Returns effective value directly from state (no thread-local check)."""
        if state.env_force_value is not UNSET:
            return state.env_force_value
        if state.user_override is not UNSET:
            return state.user_override
        if state.env_default_value is not UNSET:
            return state.env_default_value
        return state.spec.default

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
        """
        Defines and registers a generic option.

        Args:
            name: Option name.
            default: Built-in default value.
            doc: User-facing description.
            value_type: Expected runtime type after parsing.
            parser: Optional parsing function.
            validator: Optional semantic validator.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force the effective value.
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
        """
        Defines a boolean option.

        Args:
            name: Option name.
            default: Default boolean.
            doc: User-facing description.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force the effective value.
            role: Functional role/category.
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
        """
        Defines an integer option.

        Args:
            name: Option name.
            default: Default integer.
            doc: User-facing description.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force the effective value.
            role: Functional role/category.
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
        """
        Defines a float option.

        Args:
            name: Option name.
            default: Default float.
            doc: User-facing description.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force the effective value.
            role: Functional role/category.
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
        """
        Defines a string option.

        Args:
            name: Option name.
            default: Default string value.
            doc: User-facing description.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force the effective value.
            role: Functional role/category.
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
        """
        Defines an optional string option.

        Args:
            name: Option name.
            default: Default string or ``None``.
            doc: User-facing description.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force the effective value.
            role: Functional role/category.
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
        """
        Defines an optional integer option.

        Args:
            name: Option name.
            default: Default integer or ``None``.
            doc: User-facing description.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force the effective value.
            role: Functional role/category.
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
        """
        Defines an enum-like string option constrained to a fixed value set.

        Args:
            name: Option name.
            default: Default enum value.
            values: Allowed enum values.
            doc: User-facing description.
            case_sensitive: Whether value matching is case-sensitive.
                When ``False`` (default), values are normalised to lower-case.
            env_default: Environment variables that override the default.
            env_force: Environment variables that force the effective value.
            role: Functional role/category.
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

    def get(self, name: str, *, thread_local: bool = True) -> Any:
        """
        Reads the effective value of an option.

        Args:
            name: Option name (case-sensitive, uppercase by convention).
            thread_local: Whether thread-local overrides are considered.

        Returns:
            Effective option value.

        Raises:
            UnknownOptionError: If *name* is not registered.
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

    def read(self, name: str, *, thread_local: bool = True) -> Any:
        """
        Alias for :meth:`get`.

        Args:
            name: Option name.
            thread_local: Whether thread-local overrides are considered.

        Returns:
            Effective option value.
        """
        return self.get(name, thread_local=thread_local)

    def set(
        self,
        name: str,
        value: Any,
        *,
        source: ConfigSource = ConfigSource.USER,
        thread_local: bool = False,
    ) -> Any:
        """
        Sets a configuration value.

        Args:
            name: Option name.
            value: New value (will be parsed and validated).
            source: Mutation source label emitted to hooks.
            thread_local: If ``True``, applies as a thread-local override and
                requires :attr:`ConfigMutability.RUNTIME`.

        Returns:
            New effective value.

        Raises:
            ConfigError: If mutation is disallowed (forced, immutable, or
                runtime-locked startup option).
            UnknownOptionError: If *name* is not registered.
        """
        state = self._get_state(name)
        spec = state.spec
        parsed = spec.parse(value)

        with self._lock:
            if state.env_force_value is not UNSET:
                raise ConfigError(
                    f"Option '{name}' is forced by environment "
                    f"({spec.env_force}) and cannot be changed from Python."
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

            # keep os.environ in sync with the canonical casing
            os.environ[self.PREFIX + name] = str(parsed)
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

    def update(
        self,
        name: str,
        value: Any,
        *,
        source: ConfigSource = ConfigSource.USER,
        thread_local: bool = False,
    ) -> Any:
        """
        Alias for :meth:`set`.

        Args:
            name: Option name.
            value: New value.
            source: Source label.
            thread_local: Whether update is thread-local.

        Returns:
            New effective value.
        """
        return self.set(name, value, source=source, thread_local=thread_local)

    def clear_override(
        self,
        name: str,
        *,
        source: ConfigSource = ConfigSource.RESET,
        thread_local: bool = False,
    ) -> Any:
        """
        Removes a user or thread-local override for an option.

        Args:
            name: Option name.
            source: Source label for emitted hook events.
            thread_local: If ``True``, clears the thread-local override.

        Returns:
            New effective value after the override is removed.
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
            os.environ[self.PREFIX + name] = str(self._direct_effective_value(state))
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
        """
        Applies multiple updates atomically.

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
        """
        Temporarily patches one or more options and restores them on exit.

        Supported call forms::

            cfg.patch("NAME", value)
            cfg.patch({"NAME": value, "OTHER": value2})
            cfg.patch(NAME=value, OTHER=value2)

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
                        self.set(key, old, source=ConfigSource.PATCH, thread_local=True)
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

    def thread_local_override(
        self, name: str, value: Any
    ) -> contextlib.AbstractContextManager[None]:
        """
        Creates a thread-local context override for one :attr:`ConfigMutability.RUNTIME` option.

        Args:
            name: Option name.
            value: Temporary value for the calling thread.

        Returns:
            Context manager that applies and reverts the override.
        """
        return self.patch(name, value, thread_local=True)

    def list_options(self) -> tuple[str, ...]:
        """
        Returns all registered option names in sorted order.

        Returns:
            Sorted tuple of option names.
        """
        with self._lock:
            return tuple(sorted(self._options.keys()))

    @property
    def values(self) -> dict[str, Any]:
        """Effective global values for all options (ignoring thread-local overrides)."""
        return self.snapshot(thread_local=False)

    def snapshot(self, *, thread_local: bool = True) -> dict[str, Any]:
        """
        Exports a snapshot of all effective option values.

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
        """
        Returns only the explicitly overridden values.

        Args:
            thread_local: If ``True``, returns thread-local overrides for the
                calling thread, otherwise returns global user overrides.

        Returns:
            Mapping of overridden option names to their current values.
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
        """
        Computes a stable SHA-256 fingerprint of fingerprint-enabled options.

        Primarily useful for compile-cache keying and run diagnostics.

        Args:
            thread_local: Whether to include thread-local overrides.

        Returns:
            Hex-digest string.
        """
        values: dict[str, Any] = {}
        for name, state in self._options.items():
            if not state.spec.include_in_fingerprint:
                continue
            values[name] = self.get(name, thread_local=thread_local)
        payload = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def describe_option(self, name: str) -> dict[str, Any]:
        """
        Returns rich metadata for one option.

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

    def options_by_mutability(self) -> dict[ConfigMutability, tuple[str, ...]]:
        """
        Groups options by mutability category.

        Returns:
            Mapping from :class:`ConfigMutability` to sorted option-name tuples.
        """
        grouped: dict[ConfigMutability, list[str]] = {
            ConfigMutability.IMMUTABLE: [],
            ConfigMutability.STARTUP: [],
            ConfigMutability.RUNTIME: [],
        }
        for name, state in self._options.items():
            grouped[state.spec.mutability].append(name)
        return {key: tuple(sorted(vals)) for key, vals in grouped.items()}

    def options_by_role(self) -> dict[str, tuple[str, ...]]:
        """
        Groups option names by their declared role.

        Returns:
            Mapping from role name to sorted option-name tuples.
        """
        grouped: dict[str, list[str]] = {}
        for name, state in self._options.items():
            grouped.setdefault(state.spec.role, []).append(name)
        return {key: tuple(sorted(vals)) for key, vals in grouped.items()}

    def is_runtime_mutable(self, name: str) -> bool:
        """
        Reports whether an option can be mutated at runtime.

        Args:
            name: Option name.

        Returns:
            ``True`` if the option is :attr:`ConfigMutability.RUNTIME` and not
            force-locked by an environment variable, otherwise ``False``.
        """
        state = self._get_state(name)
        return (
            state.spec.mutability is ConfigMutability.RUNTIME
            and state.env_force_value is UNSET
        )

    def value_source(self, name: str, *, thread_local: bool = True) -> ConfigSource:
        """
        Reports which source currently provides an option's effective value.

        Args:
            name: Option name.
            thread_local: Whether thread-local overrides are considered.

        Returns:
            Effective :class:`ConfigSource`.
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

    @property
    def runtime_locked(self) -> bool:
        """Whether :attr:`ConfigMutability.STARTUP` options are currently locked."""
        return self._runtime_locked

    def lock_runtime(self) -> None:
        """Locks startup-only options from further programmatic mutation."""
        with self._lock:
            self._runtime_locked = True

    def unlock_runtime_for_testing(self) -> None:
        """
        Unlocks startup options.

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
        """
        Registers a mutation hook for one option.

        Hooks are invoked whenever the effective value for the option changes.

        Args:
            name: Option name.
            hook: Callable receiving a :class:`ConfigMutation` event.
            run_immediately: If ``True``, invokes *hook* once immediately with
                the current effective value.
            thread_local: When *run_immediately* is ``True``, controls whether
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
        """
        Registers a global mutation hook that fires on any option change.

        Args:
            hook: Callable receiving all :class:`ConfigMutation` events.
        """
        with self._lock:
            self._global_hooks.append(hook)

    @staticmethod
    def _format_table_cell(value: Any, *, max_width: int) -> str:
        """Formats one table cell for :meth:`show`."""
        text = "<unset>" if value is UNSET else repr(value)
        text = " ".join(text.split())
        if len(text) <= max_width:
            return text
        if max_width <= 3:
            return text[:max_width]
        return f"{text[:max_width - 3]}..."

    def show(
        self,
        *,
        file: TextIO | None = None,
        include_current: bool = True,
        include_env: bool = False,
        thread_local: bool = True,
        max_cell_width: int = 48,
    ) -> str:
        """
        Renders and prints a tabular configuration summary.

        Args:
            file: Output stream. Defaults to :data:`sys.stdout`.
            include_current: Whether to include effective value and source
                columns.
            include_env: Whether to include environment binding columns.
            thread_local: Whether current value/source should reflect
                thread-local overrides.
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
            column_specs.extend([("effective", "effective"), ("source", "source")])
        if include_env:
            column_specs.extend(
                [("env_default", "env_default"), ("env_force", "env_force")]
            )

        rows: list[dict[str, str]] = []
        for name in self.list_options():
            state = self._get_state(name)
            row: dict[str, str] = {
                "name": name,
                "role": state.spec.role,
                "default": self._format_table_cell(
                    state.spec.default, max_width=max_cell_width
                ),
                "mutability": state.spec.mutability.value,
                "runtime_mutable": "yes" if self.is_runtime_mutable(name) else "no",
            }
            if include_current:
                row["effective"] = self._format_table_cell(
                    self.get(name, thread_local=thread_local),
                    max_width=max_cell_width,
                )
                row["source"] = self.value_source(name, thread_local=thread_local).value
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

    def _warn_unused(self) -> None:
        """Warns about ``NQX_``-prefixed environment variables not in the schema."""
        registered_keys = set(self._options.keys()) | set(self._PROFILING_EXTRAS)
        env_vars = {k for k in os.environ.keys() if k.upper().startswith(self.PREFIX)}

        unused_vars = []
        for var in env_vars:
            key_no_prefix = var[len(self.PREFIX) :].upper()
            if key_no_prefix not in registered_keys:
                unused_vars.append(var)

        if not unused_vars:
            return

        table_data = [[var, os.environ.get(var, "")] for var in sorted(unused_vars)]
        warning_message = dedent(f"""
            The following environment variables with the `{self.PREFIX}` prefix are not
            recognized by neuraLQX and will be ignored:
            """)
        formatted_table = tabulate(
            table_data, headers=["Variable", "Value"], tablefmt="grid"
        )
        full_message = new_error_content(f"{warning_message}\n{formatted_table}")
        print(full_message)

    def _sync_external_envars(self) -> None:
        """Propagates neuraLQX flags to external environment variables.

        The critical pre-import env vars (JAX_PLATFORMS, XLA memory,
        OMP_NUM_THREADS, NETKET_NO_TIPS, TF_CPP_MIN_LOG_LEVEL) are set by
        ``neuralqx._boot`` via the ``neuralqx_boot.pth`` site-packages hook
        before any user import runs. The setdefault calls below are a
        belt-and-suspenders fallback for editable/source installs where the
        .pth file may not yet be in site-packages.

        x64 precision is applied programmatically via ``jax.config.update`` /
        ``nk.config.update`` when JAX/NetKet are already imported, so it
        always takes effect regardless of import order.
        """
        import sys as _sys

        jax_already_imported = "jax" in _sys.modules
        netket_already_imported = "netket" in _sys.modules
        enable_x64: bool = self.get("ENABLE_X64", thread_local=False)

        if not jax_already_imported:
            # Belt-and-suspenders: _boot.py should have done these already.
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("JAX_PLATFORMS", "")
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            # Set x64 via env var before JAX reads it at its import time.
            os.environ["JAX_ENABLE_X64"] = "1" if enable_x64 else "0"
            os.environ["NETKET_ENABLE_X64"] = "1" if enable_x64 else "0"
        else:
            # JAX already imported: env vars for platform/memory are too late,
            # but x64 can be applied programmatically at any time.
            import jax as _jax

            if bool(_jax.config.jax_enable_x64) != enable_x64:
                _jax.config.update("jax_enable_x64", enable_x64)

            if netket_already_imported:
                import netket as _nk

                try:
                    _nk.config.update("netket_enable_x64", enable_x64)
                except Exception:
                    pass  # Just in case older NetKet versions not exposing this key.

            # Keep env vars in sync for child processes/subinterpreters
            os.environ["JAX_ENABLE_X64"] = "1" if enable_x64 else "0"
            os.environ["NETKET_ENABLE_X64"] = "1" if enable_x64 else "0"

    def _register_default_configs(self) -> None:
        """Registers the complete default option set for neuraLQX."""

        #
        #
        #   Debugging and logging

        self.define_bool(
            "DEBUG",
            default=False,
            doc="Enable or disable debugging mode throughout various parts of neuraLQX.",
            env_default="NQX_DEBUG",
            role="debugging",
            mutability=ConfigMutability.RUNTIME,
        )
        self.define_bool(
            "VERBOSE",
            default=True,
            doc="Enable or disable Rich console printing.",
            env_default="NQX_VERBOSE",
            role="debugging",
            mutability=ConfigMutability.RUNTIME,
        )
        self.define_enum(
            "LOG_LEVEL",
            default="INFO",
            values=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
            doc="Set the logging level for the neuraLQX logger hierarchy.",
            case_sensitive=True,
            env_default="NQX_LOG_LEVEL",
            role="debugging",
            mutability=ConfigMutability.RUNTIME,
            include_in_fingerprint=False,
        )
        self.define_bool(
            "EXPERIMENTAL",
            default=False,
            doc="Enable experimental functions throughout the neuralqx and netket packages.",
            env_default="NQX_EXPERIMENTAL",
            role="debugging",
            mutability=ConfigMutability.RUNTIME,
        )
        self.define_bool(
            "TESTING",
            default=False,
            doc="Relax some neuraLQX features for testing purposes.",
            env_default="NQX_TESTING",
            role="debugging",
            mutability=ConfigMutability.RUNTIME,
        )

        #
        #
        #   Profiling

        self.define_int(
            "PROFILE",
            default=0,
            doc="Enable profiling in neuraLQX.",
            env_default="NQX_PROFILE",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_int(
            "PROFILE_NVTX",
            default=0,
            doc="Enable nvidia-smi profiling in neuraLQX.",
            env_default="NQX_PROFILE_NVTX",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_int(
            "PROFILE_METRICS",
            default=0,
            doc=(
                "Enable profiling telemetry (CPU/mem + GPU via NVML if available) "
                "in neuraLQX."
            ),
            env_default="NQX_PROFILE_METRICS",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_int(
            "PROFILE_TRACE",
            default=0,
            doc="Enable Perfetto UI traces in neuraLQX.",
            env_default="NQX_PROFILE_TRACE",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_int(
            "PROFILE_SYNC",
            default=0,
            doc=(
                "Enable device-accurate timing (forces synchronisation, can slow) "
                "in neuraLQX."
            ),
            env_default="NQX_PROFILE_SYNC",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_string(
            "PROFILE_DIR",
            default=self.get_static("Profiling Directory"),
            doc="Override the directory where profiling artefacts are written.",
            env_default="NQX_PROFILE_DIR",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_string(
            "PROFILE_RUN_ID",
            default="",
            doc=(
                "Optional stable run identifier for profiling outputs. "
                "If non-empty, profiler artefacts are written under run_<PROFILE_RUN_ID>."
            ),
            env_default="NQX_PROFILE_RUN_ID",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_int(
            "PROFILE_JAX_ANNOTATE",
            default=1,
            doc="Enable JAX trace annotations for profiling regions.",
            env_default="NQX_PROFILE_JAX_ANNOTATE",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_int(
            "PROFILE_JAX_TRACE",
            default=0,
            doc=(
                "Enable JAX profiler trace capture in the profiling output directory."
            ),
            env_default="NQX_PROFILE_JAX_TRACE",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_float(
            "PROFILE_SAMPLE_PERIOD_S",
            default=0.1,
            doc=(
                "Sampling period in seconds for profiling telemetry when metrics "
                "are enabled."
            ),
            env_default="NQX_PROFILE_SAMPLE_PERIOD_S",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
            validator=positive_float_validator,
        )
        self.define_int(
            "PROFILE_MAX_EVENTS",
            default=2_000_000,
            doc="Maximum number of in-memory profiling trace events before export.",
            env_default="NQX_PROFILE_MAX_EVENTS",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
            validator=positive_int_validator,
        )
        self.define_int(
            "PROFILE_MPI_AGG",
            default=0,
            doc=(
                "Aggregate profiling summaries across ranks at exit "
                "(can block until all ranks finish)."
            ),
            env_default="NQX_PROFILE_MPI_AGG",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_int(
            "PROFILE_PY_CALLS",
            default=0,
            doc=(
                "Enable deep Python call tracing for profiled wrappers/calls "
                "(high overhead)."
            ),
            env_default="NQX_PROFILE_PY_CALLS",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_string(
            "PROFILE_PY_CALLS_INCLUDE",
            default="netket,neuralqx",
            doc="Comma-separated module prefixes included by deep Python call tracing.",
            env_default="NQX_PROFILE_PY_CALLS_INCLUDE",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_string(
            "PROFILE_PY_CALLS_EXCLUDE",
            default="neuralqx.profile",
            doc="Comma-separated module prefixes excluded by deep Python call tracing.",
            env_default="NQX_PROFILE_PY_CALLS_EXCLUDE",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_int(
            "PROFILE_PY_CALLS_MAX_DEPTH",
            default=6,
            doc="Maximum Python call depth captured by deep tracing (<=0 disables limit).",
            env_default="NQX_PROFILE_PY_CALLS_MAX_DEPTH",
            role="profiling",
            mutability=ConfigMutability.STARTUP,
        )

        #
        #
        #   Precision and hardware
        self.define_bool(
            "ENABLE_X64",
            default=parse_bool(os.getenv("JAX_ENABLE_X64", "True")),
            doc="Enable 64-bit floating point operations in neuraLQX.",
            env_default=("NQX_ENABLE_X64", "JAX_ENABLE_X64"),
            role="precision",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_int(
            "CACHE",
            default=0,
            doc="Enable caching when possible.",
            env_default="NQX_CACHE",
            role="compilation",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_enum(
            "DTYPE_REAL",
            default="float64",
            values=("float16", "bfloat16", "float32", "float64"),
            doc="Default real-valued array dtype used by neuraLQX array factories.",
            env_default="NQX_DTYPE_REAL",
            role="precision",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_enum(
            "DTYPE_COMPLEX",
            default="complex128",
            values=("complex64", "complex128"),
            doc="Default complex-valued array dtype used by neuraLQX array factories.",
            env_default="NQX_DTYPE_COMPLEX",
            role="precision",
            mutability=ConfigMutability.STARTUP,
        )
        self.define_enum(
            "DTYPE_INDEX",
            default="int64",
            values=("int32", "int64"),
            doc="Integer dtype for index and topology arrays in neuraLQX.",
            env_default="NQX_DTYPE_INDEX",
            role="precision",
            mutability=ConfigMutability.STARTUP,
        )

    def _get_state(self, name: str) -> _OptionState[Any]:
        """
        Returns state object for a registered option.

        Args:
            name: Option name.

        Raises:
            UnknownOptionError: If the option does not exist.
        """
        try:
            return self._options[name]
        except KeyError as exc:
            raise UnknownOptionError(f"Unknown option '{name}'.") from exc

    def _assert_can_mutate_globally(self, spec: ConfigOption[Any]) -> None:
        """Checks global mutability constraints before a write."""
        if spec.mutability is ConfigMutability.IMMUTABLE:
            raise ConfigError(
                f"Option '{spec.name}' is immutable and cannot be changed."
            )
        if spec.mutability is ConfigMutability.STARTUP and self._runtime_locked:
            raise ConfigError(
                f"Option '{spec.name}' is startup-only and runtime is locked. "
                "Set this option before calling lock_runtime()."
            )

    def _read_env_value(
        self,
        *,
        option: ConfigOption[Any],
        env_names: Sequence[str],
    ) -> Any | _UnsetType:
        """
        Reads first present environment value using case-insensitive matching.

        env vars are matched case-insensitively to accommodate varied launcher conventions.
        """
        for env_name in env_names:
            raw_value = self._get_env_case_insensitive(env_name)
            if raw_value is None:
                continue
            try:
                return option.parse(raw_value)
            except ConfigValidationError as exc:
                raise ConfigValidationError(
                    f"Invalid value from environment variable {env_name!r} for "
                    f"option {option.name!r}: {raw_value!r}."
                ) from exc
        return UNSET

    @staticmethod
    def _get_env_case_insensitive(env_key: str) -> str | None:
        """
        Returns the value of *env_key* using a case-insensitive search.

        Args:
            env_key: Target environment variable name.

        Returns:
            Value string if found, ``None`` otherwise.
        """
        for key, value in os.environ.items():
            if key.upper() == env_key.upper():
                return value
        return None

    def _thread_local_overrides(
        self, *, create: bool
    ) -> MutableMapping[str, Any] | None:
        """Returns the per-thread override dictionary."""
        overrides = getattr(self._thread_local_store, "overrides", None)
        if overrides is None and create:
            overrides = {}
            self._thread_local_store.overrides = overrides
        return overrides

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
        """Fires per-option and global hooks when a value changes."""
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

    def _normalize_patch_args(
        self,
        *,
        arg1: str | Mapping[str, Any] | None,
        arg2: Any,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Normalises the overloaded :meth:`patch` call forms."""
        if arg1 is None:
            if arg2 is not _MISSING:
                raise TypeError(
                    "Second positional argument is valid only when first argument "
                    "is a key string."
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


def _register_default_hooks(cfg: ConfigManager) -> None:
    """
    Registers built-in runtime hooks for the default option set.

    Args:
        cfg: Configuration manager to configure.
    """

    def _refresh_debug_runtime(*, reinit: bool) -> None:
        import sys as _sys

        # Avoid importing neuralqx.debug from inside configs import bootstrap
        if "neuralqx.debug" not in _sys.modules:
            return
        try:
            import neuralqx.debug as _dbg

            if reinit:
                _dbg.initialise(force=True)
            else:
                _dbg.refresh_settings(reinit=False)
        except Exception:
            pass

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
            _refresh_debug_runtime(reinit=False)

    def _debug_hook(event: ConfigMutation) -> None:
        enabled = bool(event.new_value)
        # Enabling needs full init to create logfile/handlers
        _refresh_debug_runtime(reinit=enabled)

    def _experimental_hook(event: ConfigMutation) -> None:
        if bool(event.new_value):
            _LOGGER.warning(
                "neuraLQX: Experimental mode enabled. APIs and behaviours may change without notice."
            )

    def _x64_hook(event: ConfigMutation) -> None:
        import sys as _sys

        enabled = bool(event.new_value)
        os.environ["NETKET_ENABLE_X64"] = "1" if enabled else "0"
        os.environ["JAX_ENABLE_X64"] = "1" if enabled else "0"

        if "jax" in _sys.modules:
            import jax as _jax

            _jax.config.update("jax_enable_x64", enabled)

        if "netket" in _sys.modules:
            import netket as _nk

            try:
                _nk.config.update("netket_enable_x64", enabled)
            except Exception:
                pass

        _LOGGER.debug(
            "Applied ENABLE_X64=%s from %s mutation.",
            enabled,
            event.source.value,
        )

    cfg.add_hook("LOG_LEVEL", _log_level_hook, run_immediately=True)
    cfg.add_hook("DEBUG", _debug_hook, run_immediately=False)
    cfg.add_hook("EXPERIMENTAL", _experimental_hook, run_immediately=False)
    cfg.add_hook("ENABLE_X64", _x64_hook, run_immediately=True)


def _should_init_jax_distributed() -> bool:
    def _env_int(name, default=0):
        try:
            return int(os.environ.get(name, default))
        except Exception:
            return default

    if os.environ.get("JAX_COORDINATOR_ADDRESS"):
        return True
    if _env_int("JAX_PROCESS_COUNT", 1) > 1:
        return True
    if _env_int("SLURM_NTASKS", 1) > 1:
        return True
    if _env_int("OMPI_COMM_WORLD_SIZE", 1) > 1:
        return True
    if _env_int("PMI_SIZE", 1) > 1:
        return True
    return False


cfg = ConfigManager()
_register_default_hooks(cfg)

__all__ = [
    "cfg",
    "_should_init_jax_distributed",
]
