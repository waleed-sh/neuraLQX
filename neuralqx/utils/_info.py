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
Gather info about the system
"""

import os
from abc import ABC
from abc import abstractmethod

import platform

from typing import Any
from typing import Optional
from typing import Tuple
from typing import List
from typing import Union

import psutil
from datetime import datetime
from tabulate import tabulate

from types import SimpleNamespace

if not hasattr(psutil, "cpu_freq"):

    def _cpu_freq(percpu: bool = False):
        return None

    psutil.cpu_freq = _cpu_freq  # type: ignore[attr-defined]

# constants
STD_SPACE = 20
DIVIDER_LENGTH = 100
SWAP_DIVIDER = "=" * 20 + " SWAP " + "=" * 20


class Formatter:
    """
    A utility class for formatting outputs
    """

    @staticmethod
    def get_size(b: Union[int, float], suffix: str = "B") -> str:
        """
        Convert bytes to a human-readable format

        :param b: size in bytes
        :param suffix: suffix to append (default is 'B')
        :return: human-readable size string
        """

        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if b < factor:
                return f"{b:.2f}{unit}{suffix}"
            b /= factor
        return f"{b:.2f}P{suffix}"

    @staticmethod
    def printfmt(
        key: Any,
        value: Any = None,
        *,
        indent: int = 0,
        alignment: int = STD_SPACE,
        pre: str = "",
    ) -> str:
        """
        Format a key-value pair with optional indentation.

        :param key: the key to display
        :param value: the value associated with the key.
        :param indent: number of indentation levels
        :param alignment: alignment width for the key
        :param pre: prepend string
        :return: formatted string
        """
        INDENT_STR = "- "
        if indent > 0:
            indent_spaces = indent * 2
            alignment -= indent_spaces + len(INDENT_STR)
            pre = f"{pre}{' ' * indent_spaces}{INDENT_STR}"

        if value is not None:
            return f"{pre}{key:<{alignment}} : {value}"

        return f"{pre}{key:<{alignment}}"

    @staticmethod
    def create_divider(char: str = "=", length: int = DIVIDER_LENGTH) -> str:
        """
        Create a divider string

        :param char: the character to use for the divider
        :param length: the length of the divider
        :return: divider string
        """
        return char * length


class InfoSection(ABC):
    """
    Abstract base class for different system information sections.
    """

    @abstractmethod
    def gather_info(self) -> Tuple[str, str]:
        """
        Gather information for the section

        :return: a tuple containing the title and the formatted content
        """
        pass


class SystemInfoSection(InfoSection):
    """
    Gathers basic system information
    """

    def gather_info(self) -> Tuple[str, str]:
        """
        Gather basic system information

        :return: a tuple containing the title and the formatted content
        """
        uname = platform.uname()
        content = [
            Formatter.printfmt("System", uname.system),
            Formatter.printfmt("Node Name", uname.node),
            Formatter.printfmt("Release", uname.release),
            Formatter.printfmt("Version", uname.version),
            Formatter.printfmt("Machine", uname.machine),
            Formatter.printfmt("Processor", uname.processor),
        ]
        title = "System Information"
        return title, "\n".join(content)


class BootTimeSection(InfoSection):
    """
    Gathers system boot time information
    """

    def gather_info(self) -> Tuple[str, str]:
        """
        Gather boot time information.

        :return: a tuple containing the title and the formatted content
        """
        boot_time_timestamp = psutil.boot_time()
        boot_time = datetime.fromtimestamp(boot_time_timestamp)
        formatted_boot_time = boot_time.strftime("%Y/%m/%d %H:%M:%S")
        content = [Formatter.printfmt("Boot Time", formatted_boot_time)]
        title = "Boot Time"
        return title, "\n".join(content)


class CPUInfoSection(InfoSection):
    """
    Gathers CPU information
    """

    def gather_info(self) -> Tuple[str, str]:
        # cpu frequency (may be unavailable on some platforms/builds)
        try:
            cpufreq = psutil.cpu_freq()
        except Exception:
            cpufreq = None

        content = [
            Formatter.printfmt("Physical cores", psutil.cpu_count(logical=False)),
            Formatter.printfmt("Total cores", psutil.cpu_count(logical=True)),
        ]

        if cpufreq is not None and all(
            hasattr(cpufreq, k) for k in ("max", "min", "current")
        ):
            content.extend(
                [
                    Formatter.printfmt("Max Frequency", f"{cpufreq.max:.2f} MHz"),
                    Formatter.printfmt("Min Frequency", f"{cpufreq.min:.2f} MHz"),
                    Formatter.printfmt(
                        "Current Frequency", f"{cpufreq.current:.2f} MHz"
                    ),
                ]
            )
        else:
            content.append(Formatter.printfmt("CPU Frequency", "Unavailable"))

        content.append(Formatter.printfmt("CPU Usage Per Core"))

        cpu_percents = psutil.cpu_percent(percpu=True, interval=1)
        for i, percentage in enumerate(cpu_percents):
            content.append(Formatter.printfmt(f"Core {i}", f"{percentage}%", indent=1))

        total_cpu = psutil.cpu_percent()
        content.append(Formatter.printfmt("Total CPU Usage", f"{total_cpu}%"))

        return "CPU Info", "\n".join(content)


class MemoryInfoSection(InfoSection):
    """
    Gathers memory information
    """

    def gather_info(self) -> Tuple[str, str]:
        """
        Gather memory information

        :return: a tuple containing the title and the formatted content
        """
        svmem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        content = [
            Formatter.printfmt("Total", Formatter.get_size(svmem.total)),
            Formatter.printfmt("Available", Formatter.get_size(svmem.available)),
            Formatter.printfmt("Used", Formatter.get_size(svmem.used)),
            Formatter.printfmt("Percentage", f"{svmem.percent}%"),
            Formatter.create_divider("=") + " SWAP " + Formatter.create_divider("="),
            Formatter.printfmt("Total", Formatter.get_size(swap.total)),
            Formatter.printfmt("Free", Formatter.get_size(swap.free)),
            Formatter.printfmt("Used", Formatter.get_size(swap.used)),
            Formatter.printfmt("Percentage", f"{swap.percent}%"),
        ]
        title = "Memory Information"
        return title, "\n".join(content)


class DiskInfoSection(InfoSection):
    """
    Gathers disk information
    """

    def gather_info(self) -> Tuple[str, str]:
        """
        Gather disk information

        :return: a tuple containing the title and the formatted content
        """
        partitions = psutil.disk_partitions()
        content = [Formatter.printfmt("Partitions and Usage")]
        for partition in partitions:
            content.append(Formatter.printfmt(f"=== Device: {partition.device} ==="))
            content.append(
                Formatter.printfmt("Mountpoint", partition.mountpoint, indent=1)
            )
            content.append(
                Formatter.printfmt("File system type", partition.fstype, indent=1)
            )
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                content.append(
                    Formatter.printfmt(
                        "Total Size",
                        Formatter.get_size(partition_usage.total),
                        indent=1,
                    )
                )
                content.append(
                    Formatter.printfmt(
                        "Used", Formatter.get_size(partition_usage.used), indent=1
                    )
                )
                content.append(
                    Formatter.printfmt(
                        "Free", Formatter.get_size(partition_usage.free), indent=1
                    )
                )
                content.append(
                    Formatter.printfmt(
                        "Percentage", f"{partition_usage.percent}%", indent=1
                    )
                )
            except PermissionError:
                content.append(Formatter.printfmt("Permission Denied", indent=1))

        # disk io counters
        disk_io = psutil.disk_io_counters()
        content.append(
            Formatter.printfmt("Total Read", Formatter.get_size(disk_io.read_bytes))
        )
        content.append(
            Formatter.printfmt("Total Write", Formatter.get_size(disk_io.write_bytes))
        )
        title = "Disk Information"
        return title, "\n".join(content)


class DeviceInfoSection(InfoSection):
    """
    Gathers accelerator / device information using JAX
    """

    def gather_info(self) -> Tuple[str, str]:
        try:
            import jax
            from jax.lib import xla_bridge
        except Exception as e:
            return (
                "Device Information",
                Formatter.printfmt("JAX unavailable", str(e)),
            )

        backend = xla_bridge.get_backend().platform
        devices = jax.devices()

        content = [
            Formatter.printfmt("JAX backend", backend),
            Formatter.printfmt("Device count", len(devices)),
        ]

        for i, d in enumerate(devices):
            content.extend(
                [
                    Formatter.printfmt(f"Device {i}", "", indent=0),
                    Formatter.printfmt("Platform", d.platform, indent=1),
                    Formatter.printfmt("Kind", d.device_kind, indent=1),
                    Formatter.printfmt("ID", d.id, indent=1),
                ]
            )

        return "Device Information", "\n".join(content)


class NvidiaInfoSection(InfoSection):
    """
    Optional NVIDIA GPU details via nvidia-smi
    """

    def gather_info(self) -> Tuple[str, str]:
        import subprocess
        import shutil

        if shutil.which("nvidia-smi") is None:
            return (
                "NVIDIA GPU Details",
                Formatter.printfmt("nvidia-smi not found"),
            )

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception as e:
            return (
                "NVIDIA GPU Details",
                Formatter.printfmt("Failed to query nvidia-smi", str(e)),
            )

        rows = []
        for line in result.stdout.strip().splitlines():
            idx, name, mem_tot, mem_used, temp = map(str.strip, line.split(","))
            rows.append((idx, name, mem_tot, mem_used, temp))

        table = tabulate(
            rows,
            headers=["ID", "Name", "Total MB", "Used MB", "Temp °C"],
            tablefmt="grid",
        )

        return "NVIDIA GPU Details", table


class SystemInfoManager:
    """
    Manages the collection and display/export of system information
    """

    def __init__(
        self, sections: Optional[List[InfoSection]] = None, include_nvidia=False
    ):
        """
        Initialize the SystemInfoManager with specified sections

        :param sections: a list of InfoSection instances to gather information from.
        """
        if sections is None:
            self.sections = [
                SystemInfoSection(),
                BootTimeSection(),
                CPUInfoSection(),
                MemoryInfoSection(),
                DiskInfoSection(),
                DeviceInfoSection(),
            ]

            if include_nvidia:
                self.sections.append(NvidiaInfoSection())

        else:
            self.sections = sections

    def gather_all_info(self) -> List[Tuple[str, str]]:
        """
        Gather information from all configured sections

        :return: a list of tuples containing titles and their corresponding contents
        """
        gathered_info = []
        for section in self.sections:
            try:
                title, content = section.gather_info()
                gathered_info.append((title, content))
            except Exception as e:
                error_content = Formatter.printfmt("Error", str(e))
                gathered_info.append((section.__class__.__name__, error_content))
        return gathered_info

    def display_info(self) -> None:
        """
        Display all gathered system information in the console
        """
        gathered_info = self.gather_all_info()
        for title, content in gathered_info:
            divider = Formatter.create_divider()
            print(f"{divider}\n{title}\n{divider}\n{content}\n")

    def export_info(self, filepath: Optional[str] = None) -> None:
        """
        Export the gathered system information to a file or display it.

        :param filepath: the path to the file where information will be exported.
                         If None, information is printed to the console
        """
        gathered_info = self.gather_all_info()
        system_info = []
        for title, content in gathered_info:
            divider = Formatter.create_divider()
            system_info.append(f"{divider}\n{title}\n{divider}\n{content}\n")
        final_info = "\n".join(system_info)

        if filepath:
            try:
                # ensure the directory exists
                file_path = os.path.abspath(filepath)
                directory = os.path.dirname(file_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                # write the system information to the file
                with open(file_path, "w") as file:
                    file.write(final_info)
                print(f"System information exported successfully to {file_path}")
            except Exception as e:
                print(f"Failed to export system information: {e}")
        else:
            print(final_info)
