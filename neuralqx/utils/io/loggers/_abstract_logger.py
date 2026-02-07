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

import abc
from typing import Any


class AbstractLogger(abc.ABC):
    """
    Abstract base class defining the logging interface used throughout neuraLQX.

    A logger is responsible for collecting, organizing, sanitizing, displaying, and persisting
    structured information produced during solver execution, including configuration parameters,
    runtime metadata, and numerical results.

    Concrete implementations may differ in storage backend, output format, user interface, or
    execution context (e.g. serial vs MPI), but must expose the same public behavioral contract
    defined by this interface.

    This abstraction intentionally does not prescribe how logs are stored or rendered, it only
    defines what operations must be supported.
    """

    @abc.abstractmethod
    def add_field(self, parent_key: str, new_field_key: str) -> None:
        """
        Add a new log field under an existing parent log category.

        This method extends the structured log schema at runtime by inserting a new field key
        beneath a given parent category. The newly added field is initialised with an
        implementation-defined empty value (typically ``None``).

        :param parent_key: The name of the parent log category under which the new field should be
          created.
        :param new_field_key: The name of the field to add.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log(
        self,
        field_key: str | list[str],
        value: Any | list[Any],
    ) -> None:
        """
        Record one or more values in the log.

        This method assigns values to existing log fields. It must support both single-field logging
        and batched logging via lists of keys and values.

        When lists are provided, the implementation must ensure a one-to-one correspondence between
        field keys and values.

        :param field_key: The name(s) of the field(s) to log values to.
        :param value: The value(s) to assign to the corresponding field(s).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_log(self) -> dict[str, Any]:
        """
        Retrieve the full structured log.

        :returns: A dictionary representing the entire log structure, including all parent
          categories and their associated fields.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_value_of(self, key: str) -> Any:
        """
        Retrieve the value associated with a given log key.

        If the key corresponds to a parent category, the entire sub-dictionary for that category is
        returned. If the key corresponds to a leaf field, only the field value is returned.

        :param key:  The log field or parent category to retrieve.

        :returns: The requested value or sub-log.

        :raises KeyError: If the key does not exist anywhere in the log.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def write_log_to_file(self, path: str) -> None:
        """
        Persist the current log to disk.

        Implementations may choose the file format, naming scheme, and serialisation method
        (e.g. HTML, JSON, YAML), but must ensure that the full sanitized log state is written to the
        specified location.

        :param path: Directory or file path where the log output should be written.

        :raises IOError: If the log cannot be written to disk.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def display_log(self) -> None:
        """
        Render the current log for interactive inspection.

        This method is intended for human-readable output, such as console, TUI, or GUI rendering.
        The formatting and presentation are left to the concrete implementation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def erase_parent_content(self, parent: str) -> None:
        """
        Remove or reset all fields under a given parent log category.

        Fields that were part of the original log schema should be reset to an empty state, while
        fields added dynamically at runtime may be removed entirely.

        :param parent: The name of the parent log category to clear.

        :raises KeyError: If the specified parent category does not exist.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sanitize(self, log_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitise a log dictionary by removing invalid or empty entries.

        This method must recursively process the provided dictionary and remove entries that should
        not be exposed to users or persisted (e.g. fields with ``None`` values).

        :param log_dict: A log dictionary to sanitise.

        :returns: A cleaned version of the log dictionary suitable for display or persistence.
        """
        raise NotImplementedError
