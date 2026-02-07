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


import importlib
from types import SimpleNamespace
import pytest
import shutil
import subprocess


@pytest.fixture
def imod():
    return importlib.import_module("neuralqx.utils._info")


@pytest.mark.parametrize(
    "b, expected_suffix",
    [
        (0, "0.00B"),
        (1, "1.00B"),
        (1023, "1023.00B"),
        (1024, "1.00KB"),
        (1024**2, "1.00MB"),
        (1024**3, "1.00GB"),
    ],
)
def test_formatter_get_size_scales_units(imod, b, expected_suffix):
    assert imod.Formatter.get_size(b).endswith(expected_suffix)


def test_formatter_get_size_custom_suffix(imod):
    assert imod.Formatter.get_size(1024, suffix="iB").endswith("1.00KiB")


def test_formatter_printfmt_without_value_no_indent(imod):
    s = imod.Formatter.printfmt("KeyOnly")
    assert "KeyOnly" in s

    assert ":" not in s


def test_formatter_printfmt_with_value_and_indent_adjusts_prefix(imod):
    s = imod.Formatter.printfmt("Core 0", "50%", indent=1, alignment=20)
    assert s.startswith("  - ")
    assert "Core 0" in s
    assert " : 50%" in s


def test_formatter_create_divider_custom(imod):
    d = imod.Formatter.create_divider(char="-", length=7)
    assert d == "-------"


def test_system_info_section_formats_platform_uname(imod, monkeypatch):
    fake_uname = SimpleNamespace(
        system="XOS",
        node="node-1",
        release="1.2.3",
        version="vX",
        machine="x86_64",
        processor="fakecpu",
    )
    monkeypatch.setattr(imod.platform, "uname", lambda: fake_uname, raising=True)

    title, content = imod.SystemInfoSection().gather_info()
    assert title == "System Information"
    for key, val in [
        ("System", "XOS"),
        ("Node Name", "node-1"),
        ("Release", "1.2.3"),
        ("Version", "vX"),
        ("Machine", "x86_64"),
        ("Processor", "fakecpu"),
    ]:
        assert key in content
        assert str(val) in content


def test_boot_time_section_uses_psutil_and_formats_time(imod, monkeypatch):
    monkeypatch.setattr(imod.psutil, "boot_time", lambda: 123, raising=True)

    class FakeDT:
        @classmethod
        def fromtimestamp(cls, ts):
            assert ts == 123
            return cls()

        def strftime(self, fmt):
            assert fmt == "%Y/%m/%d %H:%M:%S"
            return "2020/01/02 03:04:05"

    monkeypatch.setattr(imod, "datetime", FakeDT, raising=True)

    title, content = imod.BootTimeSection().gather_info()
    assert title == "Boot Time"
    assert "Boot Time" in content
    assert "2020/01/02 03:04:05" in content


def test_cpu_info_section_includes_per_core_and_total_usage(imod, monkeypatch):
    monkeypatch.setattr(
        imod.psutil,
        "cpu_freq",
        lambda: SimpleNamespace(max=5000.0, min=800.0, current=3200.0),
    )
    monkeypatch.setattr(imod.psutil, "cpu_count", lambda logical: 4 if logical else 2)

    calls = {"percpu": 0, "total": 0}

    def fake_cpu_percent(*, percpu=False, interval=None):
        if percpu:
            calls["percpu"] += 1

            assert interval == 1
            return [10.0, 20.0, 30.0, 40.0]
        calls["total"] += 1
        assert percpu is False
        return 25.5

    monkeypatch.setattr(imod.psutil, "cpu_percent", fake_cpu_percent, raising=True)

    title, content = imod.CPUInfoSection().gather_info()
    assert title == "CPU Info"
    assert "Physical cores" in content
    assert "Total cores" in content
    assert "Max Frequency" in content and "5000.00 MHz" in content
    assert "Min Frequency" in content and "800.00 MHz" in content
    assert "Current Frequency" in content and "3200.00 MHz" in content

    assert "CPU Usage Per Core" in content
    for i, pct in enumerate([10.0, 20.0, 30.0, 40.0]):
        assert f"Core {i}" in content
        assert f"{pct}%" in content

    assert "Total CPU Usage" in content
    assert "25.5%" in content
    assert calls["percpu"] == 1 and calls["total"] == 1


def test_memory_info_section_contains_swap_block_and_sizes(imod, monkeypatch):
    svmem = SimpleNamespace(
        total=1024**3, available=512 * 1024**2, used=512 * 1024**2, percent=50.0
    )
    swap = SimpleNamespace(
        total=1024**2, free=512 * 1024, used=512 * 1024, percent=50.0
    )

    monkeypatch.setattr(imod.psutil, "virtual_memory", lambda: svmem, raising=True)
    monkeypatch.setattr(imod.psutil, "swap_memory", lambda: swap, raising=True)

    title, content = imod.MemoryInfoSection().gather_info()
    assert title == "Memory Information"
    assert "Total" in content and "1.00GB" in content
    assert "Available" in content
    assert "Used" in content
    assert "Percentage" in content

    assert " SWAP " in content

    assert "1.00MB" in content


def test_disk_info_section_permission_denied_path_and_io_counters(imod, monkeypatch):
    partitions = [
        SimpleNamespace(device="/dev/sda1", mountpoint="/mnt/a", fstype="ext4"),
        SimpleNamespace(device="/dev/sda2", mountpoint="/mnt/b", fstype="ext4"),
    ]
    monkeypatch.setattr(
        imod.psutil, "disk_partitions", lambda: partitions, raising=True
    )

    def fake_disk_usage(mp):
        if mp == "/mnt/a":
            return SimpleNamespace(total=100, used=20, free=80, percent=20.0)
        raise PermissionError()

    monkeypatch.setattr(imod.psutil, "disk_usage", fake_disk_usage, raising=True)
    monkeypatch.setattr(
        imod.psutil,
        "disk_io_counters",
        lambda: SimpleNamespace(read_bytes=1234, write_bytes=5678),
        raising=True,
    )

    title, content = imod.DiskInfoSection().gather_info()
    assert title == "Disk Information"
    assert "Partitions and Usage" in content

    assert "=== Device: /dev/sda1 ===" in content
    assert "Mountpoint" in content and "/mnt/a" in content
    assert "File system type" in content and "ext4" in content
    assert "Total Size" in content
    assert "Percentage" in content and "20.0%" in content

    assert "=== Device: /dev/sda2 ===" in content
    assert "Permission Denied" in content

    assert "Total Read" in content
    assert "Total Write" in content


def test_nvidia_info_section_no_nvidia_smi(imod, monkeypatch):
    def fake_which(cmd):
        assert cmd == "nvidia-smi"
        return None

    monkeypatch.setattr(shutil, "which", fake_which, raising=True)

    title, content = imod.NvidiaInfoSection().gather_info()
    assert title == "NVIDIA GPU Details"
    assert "nvidia-smi not found" in content.lower()


def test_nvidia_info_section_with_gpus_uses_tabulate(imod, monkeypatch):
    def fake_which(cmd):
        assert cmd == "nvidia-smi"
        return "/usr/bin/nvidia-smi"

    monkeypatch.setattr(shutil, "which", fake_which, raising=True)

    class FakeCompleted:
        stdout = "0, GPU0, 1000, 900, 55\n"

    def fake_run(args, capture_output, text, check):
        assert args[:1] == ["nvidia-smi"]
        assert "--query-gpu=index,name,memory.total,memory.used,temperature.gpu" in args
        assert "--format=csv,noheader,nounits" in args
        assert capture_output is True
        assert text is True
        assert check is True
        return FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run, raising=True)

    def fake_tabulate(rows, headers, tablefmt):
        assert tablefmt == "grid"
        assert headers == ["ID", "Name", "Total MB", "Used MB", "Temp °C"]
        assert rows == [("0", "GPU0", "1000", "900", "55")]
        return "TABULATED_GPU_TABLE"

    monkeypatch.setattr(imod, "tabulate", fake_tabulate, raising=True)

    title, content = imod.NvidiaInfoSection().gather_info()
    assert title == "NVIDIA GPU Details"
    assert content == "TABULATED_GPU_TABLE"


def test_manager_gather_all_info_success_and_error(imod):
    class Ok(imod.InfoSection):
        def gather_info(self):
            return "OK", "fine"

    class Boom(imod.InfoSection):
        def gather_info(self):
            raise RuntimeError("nope")

    mgr = imod.SystemInfoManager(sections=[Ok(), Boom()])
    out = mgr.gather_all_info()

    assert out[0] == ("OK", "fine")

    assert out[1][0] == "Boom"

    assert "Error" in out[1][1]
    assert "nope" in out[1][1]


def test_manager_display_info_prints_sections(imod, capsys):
    class S1(imod.InfoSection):
        def gather_info(self):
            return "T1", "C1"

    class S2(imod.InfoSection):
        def gather_info(self):
            return "T2", "C2"

    mgr = imod.SystemInfoManager(sections=[S1(), S2()])
    mgr.display_info()

    captured = capsys.readouterr().out

    assert "T1" in captured and "C1" in captured

    assert "T2" in captured and "C2" in captured


def test_manager_export_info_prints_when_no_filepath(imod, capsys):
    class S(imod.InfoSection):
        def gather_info(self):
            return "T", "C"

    mgr = imod.SystemInfoManager(sections=[S()])
    mgr.export_info(filepath=None)

    out = capsys.readouterr().out
    assert "T" in out
    assert "C" in out


def test_manager_export_info_writes_file_and_makes_directory(imod, tmp_path, capsys):
    class S(imod.InfoSection):
        def gather_info(self):
            return "T", "C"

    mgr = imod.SystemInfoManager(sections=[S()])

    out_dir = tmp_path / "newdir"
    out_file = out_dir / "info.txt"
    assert not out_dir.exists()

    mgr.export_info(filepath=str(out_file))

    assert out_dir.exists()
    assert out_file.exists()
    data = out_file.read_text()
    assert "T" in data and "C" in data

    printed = capsys.readouterr().out
    assert "exported successfully" in printed.lower()


def test_manager_export_info_handles_write_failures_gracefully(
    imod, monkeypatch, capsys, tmp_path
):
    import builtins

    class S(imod.InfoSection):
        def gather_info(self):
            return "T", "C"

    mgr = imod.SystemInfoManager(sections=[S()])

    def boom_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(builtins, "open", boom_open, raising=True)

    mgr.export_info(filepath=str(tmp_path / "x" / "info.txt"))

    out = capsys.readouterr().out
    assert "Failed to export system information" in out
    assert "disk full" in out
