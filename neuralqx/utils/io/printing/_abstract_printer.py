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

from rich.console import Console

from .types import PrintHandler


class AbstractPrinter(abc.ABC):

    def __init__(self, logger: PrintHandler = None, console: Console = None):
        """
        An abstract base class for all neuraLQX printers. Subclasses can supply their own Rich
        console or one is created by default and saved as a class attribute.
        """
        self._logger = logger or PrintHandler()
        self._console = console or Console()

    @property
    def logger(self) -> PrintHandler:
        """Accessor for the printing handler."""
        return self._logger

    @property
    def console(self) -> Console:
        """Accessor for the Rich console."""
        return self._console

    @abc.abstractmethod
    def print(self, message: str) -> None:
        """
        Subclasses of this abstract class should implement in this method a manner in which the
        printer class displays the provided `message` in a formatted Rich panel and logs it.

        :param message: The message to display and log.
        """

    @abc.abstractmethod
    def display_logs(self) -> None:
        """
        Subclasses should implement a mechanism to display all logged messages in a formatted Rich
        panel.
        """
