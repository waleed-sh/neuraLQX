#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from neuralqx.profile import _metrics


def test_metrics_sampler_handles_missing_deps(monkeypatch):
    monkeypatch.setattr(_metrics, "psutil", None, raising=False)
    monkeypatch.setattr(_metrics, "pynvml", None, raising=False)

    s = _metrics.MetricsSampler(sample_period_s=0.01)
    assert s._sample_process() == {}
    assert s._sample_system() == {}
    assert s._sample_gpus() == []


def test_metrics_sampler_samples_process_and_system(monkeypatch):
    class FakeVMem:
        total = 1000
        used = 600
        available = 400

    class FakeMemInfo:
        rss = 111
        vms = 222

    class FakeIO:
        read_bytes = 10
        write_bytes = 20

    class FakeProc:
        def cpu_percent(self, interval=None):
            return 12.5

        def memory_info(self):
            return FakeMemInfo()

        def io_counters(self):
            return FakeIO()

        def num_threads(self):
            return 7

    class FakePsutil:
        @staticmethod
        def Process(pid):
            return FakeProc()

        @staticmethod
        def cpu_percent(interval=None):
            return 33.3

        @staticmethod
        def virtual_memory():
            return FakeVMem()

    monkeypatch.setattr(_metrics, "psutil", FakePsutil, raising=False)
    monkeypatch.setattr(_metrics, "pynvml", None, raising=False)

    s = _metrics.MetricsSampler(sample_period_s=0.01)
    p = s._sample_process()
    assert p["cpu_percent"] == 12.5
    assert p["rss"] == 111
    assert p["vms"] == 222
    assert p["num_threads"] == 7
    assert p["read_bytes"] == 10
    assert p["write_bytes"] == 20

    sys = s._sample_system()
    assert sys["cpu_percent"] == 33.3
    assert sys["mem_total"] == 1000
    assert sys["mem_used"] == 600
    assert sys["mem_available"] == 400


def test_metrics_sampler_samples_gpus_via_nvml(monkeypatch):
    class FakeUtil:
        gpu = 90
        memory = 12

    class FakeMem:
        used = 123
        total = 456

    class FakeNvml:
        NVML_TEMPERATURE_GPU = 0
        NVML_CLOCK_SM = 1
        NVML_CLOCK_MEM = 2

        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlDeviceGetCount():
            return 1

        @staticmethod
        def nvmlDeviceGetHandleByIndex(i):
            return object()

        @staticmethod
        def nvmlDeviceGetUtilizationRates(h):
            return FakeUtil()

        @staticmethod
        def nvmlDeviceGetMemoryInfo(h):
            return FakeMem()

        @staticmethod
        def nvmlDeviceGetPowerUsage(h):
            return 25000

        @staticmethod
        def nvmlDeviceGetTemperature(h, kind):
            return 65

        @staticmethod
        def nvmlDeviceGetClockInfo(h, kind):
            return 1000 if kind == FakeNvml.NVML_CLOCK_SM else 2000

        @staticmethod
        def nvmlDeviceGetName(h):
            return b"FakeGPU"

    monkeypatch.setattr(_metrics, "pynvml", FakeNvml, raising=False)
    monkeypatch.setattr(_metrics, "psutil", None, raising=False)

    s = _metrics.MetricsSampler(sample_period_s=0.01)
    g = s._sample_gpus()
    assert len(g) == 1
    assert g[0]["index"] == 0
    assert g[0]["name"] == "FakeGPU"
    assert g[0]["util_gpu"] == 90
    assert g[0]["mem_total"] == 456
    assert g[0]["power_mw"] == 25000
    assert g[0]["temp_c"] == 65
    assert g[0]["clock_sm_mhz"] == 1000
    assert g[0]["clock_mem_mhz"] == 2000


def test_global_start_stop_metrics(monkeypatch):
    class FakeSampler:
        def __init__(self, sample_period_s=1.0):
            self._stopped = False

        def start(self):
            pass

        def stop(self):
            self._stopped = True

        def samples(self):
            return [{"t_wall_s": 0.0, "process": {}, "system": {}, "gpus": []}]

    monkeypatch.setattr(_metrics, "MetricsSampler", FakeSampler, raising=True)
    _metrics.stop_metrics()
    _metrics.start_metrics(sample_period_s=0.1)
    out = _metrics.get_metrics_samples()
    assert out and out[0]["t_wall_s"] == 0.0
    _metrics.stop_metrics()
    assert _metrics.get_metrics_samples() == []
