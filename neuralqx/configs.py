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
This file contains some logic to read and interpret environment variables
"""

import inspect
import os
import threading
from datetime import datetime

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import List

from dataclasses import dataclass
from dataclasses import field

from tabulate import tabulate

from textwrap import dedent


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
    def __init__(self, msg: str):
        super().__init__(new_error_content(msg))


class ReadOnlyDict(Dict[str, Any]):

    def __setitem__(self, key, value):
        raise TypeError("This configuration dictionary is read-only.")

    def __delitem__(self, key):
        raise TypeError("This configuration dictionary is read-only.")


@dataclass
class ConfigItem:
    """
    A data class to represent a single configuration item
    """

    default: Any
    """The default value of the configuration variable."""

    mutable: bool
    """A flag indicating whether the variable is mutable at runtime."""

    parser: Callable[[str], Any]
    """A callable to parse the environment variable string into the desired type."""

    description: str
    """The description of the configuration variable."""

    callback: Optional[Callable[[Any, Any], None]] = field(default=None)
    """The callback function to execute when the variable's value changes."""


class ConfigSchema:
    """
    Helper to manage the schema of configuration variables. It handles registration and retrieval
    of configuration items.
    """

    def __init__(self):
        """
        Initialises the ConfigSchema with an empty dictionary.
        """
        self._schema: Dict[str, ConfigItem] = {}

    def register(
        self,
        env_name: str,
        env_val_type: Callable[[str], Any],
        env_default_val: Any,
        env_desc: str,
        runtime: bool,
        callback: Optional[Callable[[Any, Any], None]] = None,
    ) -> None:
        """
        Registers a new configuration variable in the schema.

        :param env_name: name of the environment variable following the structure NQX_{ENV NAME}
        :param env_val_type: type of the environment variable's value (e.g., bool, int, str)
        :param env_default_val: default value for the environment variable
        :param env_desc: description of the environment variable
        :param runtime: flag indicating whether the variable can be modified at runtime
        :param callback: optional callback function to execute when the variable's value changes
        """

        key_upper = env_name.upper()

        config_item = ConfigItem(
            default=env_default_val,
            mutable=runtime,
            parser=env_val_type,
            description=env_desc,
            callback=callback,
        )

        self._schema[key_upper] = config_item

    def get_schema(self) -> Dict[str, ConfigItem]:
        """
        Retrieves the entire configuration schema

        :return: dictionary mapping environment variable names to ConfigItem instances
        """
        return self._schema


class ConfigManager:
    """
    A singleton class to manage package-specific environment variables with the prefix 'NQX_'

    It handles initialisation, synchronisation with os.environ, and provides access to configuration
    variables throughout the package
    """

    _instance = None
    """A class variable to hold the singleton instance."""

    _lock = threading.Lock()
    """A lock to ensure thread-safe singleton instantiation."""

    PREFIX = "NQX_"
    """The standard neuraLQX prefix for environment variables managed by ConfigManager."""

    _PROFILING_EXTRAS: List[str] = [
        "NQX_PROFILE_RUN_ID",
        "NQX_PROFILE_JAX_ANNOTATE",
        "NQX_PROFILE_SAMPLE_PERIOD_S",
        "NQX_PROFILE_MAX_EVENTS",
        "PROFILE_MPI_AGG",
        "PROFILE_JAX_ANNOTATIONS",
        "PROFILE_JAX_TRACE",
    ]
    """Additional envvars which may be set for profiling but we do not need to store them explicitly."""

    def __new__(cls):
        """
        Controls the instantiation of the singleton instance by ensuring only one instance exists
        across the package

        :return: The singleton instance of ConfigManager.
        """

        if cls._instance is None:
            with cls._lock:

                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance._initialized = False

        return cls._instance

    def __init__(self):
        """
        Initialises the ConfigManager instance. It sets up the configuration schema and synchronises
        environment variables
        """

        if self._initialized:
            return

        self._schema = ConfigSchema()

        self._register_statics()

        self._register_default_configs()

        self._config: Dict[str, Any] = {}
        self._callbacks: Dict[str, Callable[[Any, Any], None]] = {}
        self._initialize_config()

        # propagate neuraLQX to NetKet flags before anybody might import netket
        # all it needs to run is os.environ, so we keep it here
        self._sync_external_envars()

        self._warn_unused()

        self._initialized = True

    def _warn_unused(self) -> None:
        """
        Searches for unused environment variables which begin with the prefix `NQX_`
        and are not registered in the schema and not a known package config or static variable
        and emits a runtime non-interrupting warning to the user.
        """

        registered_keys = set(self._schema.get_schema().keys()) | set(
            self._PROFILING_EXTRAS
        )

        env_vars = {k for k in os.environ.keys() if k.upper().startswith(self.PREFIX)}

        unused_vars = []
        for var in env_vars:
            key_no_prefix = var[len(self.PREFIX) :].upper()
            if key_no_prefix not in registered_keys:
                unused_vars.append(var)

        if not unused_vars:
            return

        table_data = []
        for var in sorted(unused_vars):
            table_data.append([var, os.environ.get(var, "")])

        warning_message = dedent(f"""
            The following environment variables with the `{self.PREFIX}` prefix are not
            recognized by neuraLQX and will be ignored:
            """)

        formatted_table = tabulate(
            table_data, headers=["Variable", "Value"], tablefmt="grid"
        )

        # emit a runtime warning
        full_message = new_error_content(f"{warning_message}\n{formatted_table}")
        print(full_message)
        return

    def _sync_external_envars(self):
        """
        Propagate runtime-related flags into external environment variables once.

        neuraLQX now relies on JAX sharding/distributed execution and no longer
        manages an MPI execution mode from config.
        """

        os.environ.setdefault("OMP_NUM_THREADS", "1")

        os.environ.setdefault("JAX_PLATFORMS", "")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("NETKET_EXPERIMENTAL_SHARDING", "True")

        if bool(int(self._config.get("ENABLE_X64", 0))):
            # set NetKet to x64, which will set JAX to x64
            os.environ["NETKET_ENABLE_X64"] = "1"
            os.environ["JAX_ENABLE_X64"] = "1"
        else:
            os.environ["NETKET_ENABLE_X64"] = "0"
            os.environ["JAX_ENABLE_X64"] = "0"

    def _register_statics(self) -> None:
        """
        Registers the statics "global variables" of neuraLQX which can be used package wide
        """

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

        # a read-only dict to hold all neuraLQX "global" variables
        self._statics = ReadOnlyDict(_s)

    @property
    def statics(self):
        return self._statics

    def get_static(self, name) -> Any:
        if name not in self.statics:
            raise KeyError(
                f"You have tried to access a static variable `{name}` which does not exist."
            )
        return self._statics[name]

    def _register_default_configs(self) -> None:
        """
        Registers the default configuration variables in the schema.
        """

        self._schema.register(
            env_name="DEBUG",
            env_val_type=self._parse_bool,
            env_default_val=False,
            env_desc="Enable or disable debugging mode throughout various parts of neuraLQX.",
            runtime=True,
        )

        self._schema.register(
            env_name="PROFILE",
            env_val_type=int,
            env_default_val=0,
            env_desc="Enable profiling in neuraLQX.",
            runtime=False,
        )

        self._schema.register(
            env_name="PROFILE_NVTX",
            env_val_type=int,
            env_default_val=0,
            env_desc="Enable nvidia-smi profiling in neuraLQX.",
            runtime=False,
        )

        self._schema.register(
            env_name="PROFILE_METRICS",
            env_val_type=int,
            env_default_val=0,
            env_desc="Enable profiling telemetry (CPU/mem + GPU via NVML if available) in neuraLQX.",
            runtime=False,
        )

        self._schema.register(
            env_name="PROFILE_TRACE",
            env_val_type=int,
            env_default_val=0,
            env_desc="Enable Perfetto UI traces in neuraLQX.",
            runtime=False,
        )

        self._schema.register(
            env_name="PROFILE_SYNC",
            env_val_type=int,
            env_default_val=0,
            env_desc="Enable device-accurate timing (forces synchronization, can slow) in neuraLQX.",
            runtime=False,
        )

        self._schema.register(
            env_name="PROFILE_DIR",
            env_val_type=str,
            env_default_val=self.get_static("Profiling Directory"),
            env_desc="Override the directory where profiling artifacts are written.",
            runtime=False,
        )

        self._schema.register(
            env_name="VERBOSE",
            env_val_type=self._parse_bool,
            env_default_val=True,
            env_desc="Enable or disable Rich console printing.",
            runtime=True,
        )

        self._schema.register(
            env_name="LOG_LEVEL",
            env_val_type=str,
            env_default_val="INFO",
            env_desc="Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).",
            runtime=False,
        )

        self._schema.register(
            env_name="EXPERIMENTAL",
            env_val_type=int,
            env_default_val=0,
            env_desc="Enable experimental functions throughout the neuralqx and netket packages.",
            runtime=True,
        )

        self._schema.register(
            env_name="TESTING",
            env_val_type=int,
            env_default_val=0,
            env_desc="Relax some neuraLQX features for testing purposes.",
            runtime=True,
        )

        self._schema.register(
            env_name="CACHE",
            env_val_type=int,
            env_default_val=0,
            env_desc="Enable caching when possible.",
            runtime=False,
        )

        self._schema.register(
            env_name="ENABLE_X64",
            env_val_type=int,
            env_default_val=self._parse_bool(os.getenv("JAX_ENABLE_X64", "True")),
            env_desc="Enable 64-bit floating point operations in neuraLQX.",
            runtime=False,
        )

    @staticmethod
    def _parse_bool(value: str) -> bool:
        """
        Parses a string to a boolean value. Accepts various representations of true and false.

        :param value: string representation of the boolean value
        :return: the boolean value
        :raises ValueError: the string cannot be parsed to a boolean.
        """

        # define sets of true and false representations
        true_set = {"true", "1", "yes", "on", "t", "y"}
        false_set = {"false", "0", "no", "off", "f", "n"}

        val_lower = value.strip().lower()

        if val_lower in true_set:
            return True
        elif val_lower in false_set:
            return False
        else:
            raise ValueError(f"Cannot parse boolean value from '{value}'.")

    def _initialize_config(self):
        """
        Initialises the configuration by scanning os.environ for relevant variables. Sets variables
        found in the environment or assigns default values otherwise.
        """

        schema = self._schema.get_schema()

        for key, config_item in schema.items():

            env_key = self.PREFIX + key.upper()

            env_value = self._get_env_case_insensitive(env_key)

            if env_value is not None:
                try:
                    parsed_value = config_item.parser(env_value)

                except ValueError as e:
                    # raise a ConfigError if parsing fails
                    raise ConfigError(
                        f"Error parsing environment variable '{env_key}': {e}"
                    )

                # assign the parsed value to the internal configuration dictionary
                self._config[key] = parsed_value

                # update os.environ to ensure consistent casing and representation
                os.environ[env_key] = str(parsed_value)

            else:
                # assign the default value if the environment variable is not set
                self._config[key] = config_item.default

                # set the environment variable to the default value
                os.environ[env_key] = str(config_item.default)

            # if a callback is defined for this variable, store it
            if config_item.callback:
                self._callbacks[key] = config_item.callback

    @staticmethod
    def _get_env_case_insensitive(env_key: str) -> Optional[str]:
        """
        Retrieves the value of an environment variable in a case-insensitive manner.

        :param env_key: the exact name of the environment variable to retrieve
        :return: the value of the environment variable if found; otherwise, None.
        """

        for key, value in os.environ.items():
            if key.upper() == env_key.upper():
                return value
        # return None if the environment variable is not found
        return None

    def get(self, key: str) -> Any:
        """
        Retrieves the value of a configuration variable.

        :param key: the name of the configuration variable (case-insensitive)
        :return: the current value of the configuration variable
        :raises ConfigError: if the configuration key is not defined
        """

        key_upper = key.upper()

        schema = self._schema.get_schema()

        if key_upper not in schema:
            # raise an error if the key is not defined
            raise ConfigError(f"Configuration key '{key}' is not defined.")

        config_item = schema[key_upper]

        # for mutable variables, check if their value has been updated since it has been set
        if config_item.mutable:

            env_key = self.PREFIX + key_upper
            env_val = self._get_env_case_insensitive(env_key)

            if env_val is not None:
                # if it is not empty, try to parse its value from os.environ
                try:
                    parsed_val = config_item.parser(env_val)
                except ValueError as e:
                    raise ConfigError(
                        f"Error parsing environment variable {env_key}: {e}"
                    )

                # check if the current os.environ value doesnt match the saved value
                if parsed_val != self._config.get(key_upper):
                    # if so, change the saved value
                    self.set(key, parsed_val)

        return self._config.get(key_upper)

    def set(self, key: str, value: Any) -> None:
        """
        Sets the value of a mutable (runtime editable) configuration variable. Updates both the
        internal state and the corresponding environment variable. Executes any associated callback
        functions upon successful update.

        :param key: the name of the configuration variable to set (case-insensitive)
        :param value: the new value to assign to the configuration variable
        :raises ConfigError: if the key is not defined or is immutable, or if parsing fails
        """

        # to ensure thread safety across different submodules
        with self._lock:

            key_upper = key.upper()
            schema = self._schema.get_schema()

            if key_upper not in schema:
                # raise an error if the key is not defined
                raise ConfigError(f"Configuration key '{key}' is not defined.")

            # retrieve the ConfigItem for the given key
            config_item = schema[key_upper]

            # check if the configuration variable is mutable
            if not config_item.mutable:
                raise ConfigError(
                    f"Configuration key '{key}' is not runtime editable and cannot be modified."
                )

            try:
                # parse the new value using the specified parser
                parsed_value = config_item.parser(str(value))
            except ValueError as e:
                # raise a ConfigError if parsing fails
                raise ConfigError(f"Error parsing value for '{key}': {e}")

            # retrieve the old value for callback purposes
            old_value = self._config.get(key_upper)

            # update the internal configuration dictionary with the new value
            self._config[key_upper] = parsed_value

            # update the corresponding environment variable
            env_key = self.PREFIX + key_upper
            os.environ[env_key] = str(parsed_value)

            # check if a callback is associated with this configuration variable
            if key_upper in self._callbacks:
                # execute the callback with old and new values
                self._callbacks[key_upper](old_value, parsed_value)

    def register_callback(self, key: str, callback: Callable[[Any, Any], None]) -> None:
        """
        Registers a callback function to be executed when a configuration variable's value changes

        :param key: the name of the configuration variable (case-insensitive)
        :param callback: callable that accepts two arguments: old value and new value
        :raises ConfigError: if the configuration key is not defined
        """

        # to ensure thread safety across different submodules
        with self._lock:

            key_upper = key.upper()
            schema = self._schema.get_schema()

            # check if the key exists in the schema
            if key_upper not in schema:
                raise ConfigError(f"Configuration key '{key}' is not defined.")

            # assign the callback to the _callbacks dictionary
            self._callbacks[key_upper] = callback

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a shallow copy of the current configuration as a dictionary

        :return: A dictionary mapping configuration keys to their current values.
        """

        return self._config.copy()

    @staticmethod
    def _get_env_type(parser_type: Any) -> str:
        """
        Returns the readable format of a cfg item dtype
        """

        if callable(parser_type) and not inspect.isclass(parser_type):
            env_type = str(parser_type.__annotations__.get("return"))
        else:
            env_type = str(parser_type)

        # truncate unnecessary string characters
        return env_type[8:-2]

    def print_config(self):
        """
        Prints all configuration variables, their current values, mutability status and descriptions
        in a formatted table
        """

        schema = self._schema.get_schema()
        table = []

        for key, config_item in schema.items():

            env_key = self.PREFIX + key.upper()
            value = self._config.get(key)
            mutable = "Yes" if config_item.mutable else "No"
            description = config_item.description
            dtype = self._get_env_type(config_item.parser)

            table.append([env_key, dtype, value, mutable, description])

        headers = [
            "Variable",
            "Type",
            "Value",
            "Mutable",
            "Description",
        ]

        print(tabulate(table, headers=headers, tablefmt="grid"))

    def __repr__(self):
        """
        Returns the string representation of the ConfigManager instance.

        :return: a string representation of the ConfigManager
        """

        return f"<ConfigManager {self._config}>"


def _should_init_jax_distributed() -> bool:
    import os

    def _env_int(name, default=0):
        try:
            return int(os.environ.get(name, default))
        except Exception:
            return default

    # explicit JAX distributed env
    if os.environ.get("JAX_COORDINATOR_ADDRESS"):
        return True
    if _env_int("JAX_PROCESS_COUNT", 1) > 1:
        return True

    # common launcher envs
    if _env_int("SLURM_NTASKS", 1) > 1:
        return True
    if _env_int("OMPI_COMM_WORLD_SIZE", 1) > 1:
        return True
    if _env_int("PMI_SIZE", 1) > 1:
        return True

    return False


cfg = ConfigManager()

# disable NK tips
os.environ["NETKET_NO_TIPS"] = "True"

# disable C++ backend warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
