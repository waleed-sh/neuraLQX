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

import os
import threading
import time
from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore


@dataclass
class MetricsSample:
    t_wall_s: float
    process: Dict[str, Any]
    system: Dict[str, Any]
    gpus: List[Dict[str, Any]]


class MetricsSampler:
    """
    Low-overhead telemetry sampler.

    - No nvidia-smi calls (which are expensive and can perturb runs)
    - Uses psutil if available, NVML via pynvml if available
    - Sampling runs in a daemon thread and stores samples in memory,
      exported at the end. Keep sample rate low (default 1 Hz).
    """

    def __init__(self, sample_period_s: float = 1.0) -> None:
        self.sample_period_s = max(0.1, float(sample_period_s))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[MetricsSample] = []
        self._t0 = time.time()

        self._proc = None
        if psutil is not None:
            try:
                self._proc = psutil.Process(os.getpid())
                # prime cpu_percent
                self._proc.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
            except Exception:
                self._proc = None

        self._nvml_ok = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_ok = True
            except Exception:
                self._nvml_ok = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="nqx-metrics", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def samples(self) -> List[Dict[str, Any]]:
        out = []
        for s in self._samples:
            out.append(
                {
                    "t_wall_s": s.t_wall_s,
                    "process": s.process,
                    "system": s.system,
                    "gpus": s.gpus,
                }
            )
        return out

    def _sample_process(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self._proc is None:
            return d
        try:
            mem = self._proc.memory_info()
            io = (
                self._proc.io_counters() if hasattr(self._proc, "io_counters") else None
            )
            d.update(
                {
                    "cpu_percent": self._proc.cpu_percent(interval=None),
                    "rss": getattr(mem, "rss", None),
                    "vms": getattr(mem, "vms", None),
                    "num_threads": self._proc.num_threads(),
                }
            )
            if io is not None:
                d.update(
                    {
                        "read_bytes": getattr(io, "read_bytes", None),
                        "write_bytes": getattr(io, "write_bytes", None),
                    }
                )
        except Exception:
            return d
        return d

    def _sample_system(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if psutil is None:
            return d
        try:
            vm = psutil.virtual_memory()
            d.update(
                {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "mem_total": getattr(vm, "total", None),
                    "mem_used": getattr(vm, "used", None),
                    "mem_available": getattr(vm, "available", None),
                }
            )
        except Exception:
            return d
        return d

    def _sample_gpus(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self._nvml_ok or pynvml is None:
            return out
        try:
            n = pynvml.nvmlDeviceGetCount()
        except Exception:
            return out
        for i in range(n):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                pwr = None
                try:
                    pwr = pynvml.nvmlDeviceGetPowerUsage(h)  # milliwatts
                except Exception:
                    pwr = None
                temp = None
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        h, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temp = None
                clocks_sm = None
                clocks_mem = None
                try:
                    clocks_sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                    clocks_mem = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    pass
                name = None
                try:
                    name = pynvml.nvmlDeviceGetName(h).decode("utf-8", "ignore")
                except Exception:
                    name = None
                out.append(
                    {
                        "index": i,
                        "name": name,
                        "util_gpu": getattr(util, "gpu", None),
                        "util_mem": getattr(util, "memory", None),
                        "mem_used": getattr(mem, "used", None),
                        "mem_total": getattr(mem, "total", None),
                        "power_mw": pwr,
                        "temp_c": temp,
                        "clock_sm_mhz": clocks_sm,
                        "clock_mem_mhz": clocks_mem,
                    }
                )
            except Exception:
                continue
        return out

    def _run(self) -> None:
        while not self._stop.is_set():
            t = time.time() - self._t0
            sample = MetricsSample(
                t_wall_s=t,
                process=self._sample_process(),
                system=self._sample_system(),
                gpus=self._sample_gpus(),
            )
            self._samples.append(sample)
            self._stop.wait(self.sample_period_s)


_global_sampler: Optional[MetricsSampler] = None


def start_metrics(sample_period_s: float = 1.0) -> None:
    global _global_sampler
    if _global_sampler is None:
        _global_sampler = MetricsSampler(sample_period_s=sample_period_s)
        _global_sampler.start()


def stop_metrics() -> List[Dict[str, Any]]:
    global _global_sampler
    if _global_sampler is None:
        return []
    try:
        _global_sampler.stop()
        return _global_sampler.samples()
    finally:
        _global_sampler = None


def get_metrics_samples() -> List[Dict[str, Any]]:
    if _global_sampler is None:
        return []
    return _global_sampler.samples()
