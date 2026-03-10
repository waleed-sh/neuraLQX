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

from rich.panel import Panel
from rich.table import Table

from ._abstract_printer import AbstractPrinter
from .... import cfg
from ....utils import distributed as _dist


class NQXPrinter(AbstractPrinter):
    """
    Default neuraLQX printer implementation.

    This printer renders user-facing messages to a Rich-enabled console and records them via the
    underlying logger. Output is formatted as Rich panels containing the message, timestamp, and
    caller information for improved readability and traceability.

    The printer is MPI-aware: messages are only displayed on the global master rank (rank 0) unless
    running in serial mode. Worker ranks remain silent to avoid duplicated output.

    Printing behavior is controlled by the global configuration; messages are only emitted when the
    ``NQX_VERBOSE`` configuration flag is enabled.

    In addition to real-time message display, the printer can render all accumulated log entries in
    a structured Rich table for post hoc inspection.

    This class is intended to be the default console printer for neuraLQX applications.
    """

    def print(self, message: str) -> None:
        """
        Displays a message in a formatted Rich panel and logs it

        :param message: The message to display and log.
        """

        # distributed guard: only process-0 prints unless we run in serial
        if not _dist.is_global_master():
            # silent worker ranks
            return

        # if we are here, we are rank-0, so we print
        if cfg.get("VERBOSE"):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            panel = self._create_message_panel(message, timestamp)
            self.console.print(panel)
            self.logger.log(message)

    def _create_message_panel(self, message: str, timestamp: str) -> Panel:
        """
        Creates a Rich panel containing the message and timestamp

        :param message: the message to display
        :param timestamp: the timestamp of the message

        :return: a Rich panel with the formatted message
        """

        table = Table(show_header=False, box=None)
        table.add_row("Message:", message)
        table.add_row("Timestamp:", timestamp)

        caller_info = self._get_current_caller_info()

        panel = Panel(
            table,
            title="neuraLQX Printer",
            border_style="cyan",
            subtitle=f"Logged by {caller_info}",
            padding=(1, 2),
        )

        return panel

    @staticmethod
    def _get_current_caller_info() -> str:
        """
        Retrieves the current caller's class and function name

        :return: formatted caller information
        """

        stack = inspect.stack()
        if len(stack) > 1:
            # the actual called is 3 levels up
            frame = stack[3].frame
            cls = frame.f_locals.get("self", None)
            class_name = cls.__class__.__name__ if cls else "UnknownClass"
            function_name = frame.f_code.co_name
            return f"{class_name}.{function_name}()"
        return "UnknownCaller"

    def display_logs(self) -> None:
        """
        Displays all logged messages in a formatted Rich panel
        """

        # honour the same distributed rule as .print()
        if not _dist.is_global_master():
            return

        if not self.logger.logs:
            self.console.print(Panel("No logs available", border_style="red"))
            return

        log_table = self._create_log_table()
        log_panel = Panel(
            log_table,
            title="Logged Messages",
            border_style="bold green",
            padding=(1, 1),
        )
        self.console.print(log_panel)

    def _create_log_table(self) -> Table:
        """
        Creates a Rich table containing all log entries

        :return: a Rich table with all logs
        """

        table = Table(show_header=True, header_style="bold black", box=None)
        table.add_column("Timestamp", justify="center")
        table.add_column("Caller", justify="center")
        table.add_column("Message", justify="center")
        table.add_column("UID", justify="center")

        for log in self.logger.logs:
            table.add_row(log.timestamp, log.caller, log.message, log.uid)

        return table
