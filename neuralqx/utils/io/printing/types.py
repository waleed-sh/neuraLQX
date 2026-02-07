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


import datetime
import inspect
from dataclasses import dataclass, field
from collections.abc import Callable

from ....utils.misc.auth import get_hash


@dataclass
class PrintEntry:
    """
    Represents a single print entry with relevant metadata
    """

    timestamp: str
    caller: str
    message: str
    uid: str = field(default_factory=get_hash)


class PrintHandler:
    """
    Handles logging of messages with metadata
    """

    def __init__(self, hash_function: Callable[[], str] = get_hash):
        """
        Initializes the PrintHandler


        :param hash_function: a function to generate unique identifiers
        """
        self.logs: list[PrintEntry] = []
        self.hash_function = hash_function

    def log(self, message: str) -> None:
        """
        Logs a message with timestamp, caller information, and a unique identifier


        :param message: the message to log
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caller_info = self._get_caller_info()
        uid = self.hash_function()

        print_entry = PrintEntry(
            timestamp=timestamp,
            caller=caller_info,
            message=message,
            uid=uid,
        )

        self.logs.append(print_entry)

    @staticmethod
    def _get_caller_info() -> str:
        """
        Retrieves the caller's filename, line number, and function name
        """

        stack = inspect.stack()
        if len(stack) > 2:
            caller = stack[3]
            return f"{caller.filename}:{caller.lineno} in {caller.function}"
        return "Unknown Caller"
