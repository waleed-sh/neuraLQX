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


"""Profiler option declarations, imported without loading any collectors."""

from __future__ import annotations

import math


def register_profile_options(cfg) -> None:
    """Register runtime capture controls without importing collectors or devices.

    Args:
        cfg: Package configuration manager receiving profile_* declarations.

    Options are excluded from compilation fingerprints. Environment defaults
    use the corresponding NEURALQX_PROFILE_* names.
    """
    from .config import ConfigMutability
    from .config import ConfigValidationError
    from .config import non_empty_string_validator
    from .config import non_negative_int_validator
    from .config import parse_csv_tuple
    from .config import positive_int_validator

    def positive_float(value):
        if not math.isfinite(value) or value <= 0:
            raise ConfigValidationError("Expected a positive finite number.")

    def host_level(value):
        if value not in (1, 2, 3):
            raise ConfigValidationError(
                "Host tracer level must be 1, 2, or 3 to retain clock anchors."
            )

    def python_level(value):
        if value is not None and value not in (0, 1):
            raise ConfigValidationError("Python tracer level must be None, 0, or 1.")

    common = {
        "role": "profiling",
        "mutability": ConfigMutability.RUNTIME,
        "include_in_fingerprint": False,
    }
    declarations = (
        (
            "string",
            "output_dir",
            ".neuralqx/profiles",
            "Root directory for profiling runs.",
            non_empty_string_validator,
        ),
        (
            "optional_string",
            "run_id",
            None,
            "Unique shared run identifier. Required for multiple processes.",
            None,
        ),
        (
            "optional_string",
            "coordinator_address",
            None,
            "Rank-zero host:port for TCP trace collection.",
            None,
        ),
        (
            "float",
            "timeout_s",
            60.0,
            "Deadline for each distributed coordination or transfer phase.",
            positive_float,
        ),
        (
            "int",
            "clock_samples",
            5,
            "Round trips used to estimate each rank's clock offset.",
            positive_int_validator,
        ),
        (
            "int",
            "max_events",
            100_000,
            "Maximum buffered application and telemetry events per rank.",
            positive_int_validator,
        ),
        (
            "int",
            "max_lanes",
            1024,
            "Maximum application thread and async task lanes per rank.",
            positive_int_validator,
        ),
        (
            "int",
            "jax_host_tracer_level",
            1,
            "Native host detail from 1 to 3. Clock anchors require at least 1.",
            host_level,
        ),
        (
            "optional_int",
            "jax_python_tracer_level",
            None,
            "Python call tracing. None selects 1 for full and 0 otherwise.",
            python_level,
        ),
        ("bool", "nvtx", False, "Emit ranges for external NVIDIA Nsight tools.", None),
        (
            "float",
            "telemetry_interval_s",
            1.0,
            "Seconds between background GPU management samples.",
            positive_float,
        ),
        (
            "float",
            "nvidia_smi_timeout_s",
            2.0,
            "Timeout for each read-only nvidia-smi query.",
            positive_float,
        ),
        (
            "int",
            "max_trace_bytes",
            4_294_967_296,
            "Maximum compressed rank trace accepted by TCP collection.",
            positive_int_validator,
        ),
        (
            "int",
            "wait_steps",
            0,
            "Unprofiled steps per scheduled capture cycle.",
            non_negative_int_validator,
        ),
        (
            "int",
            "warmup_steps",
            0,
            "Discarded native-tracer warmup steps per cycle.",
            non_negative_int_validator,
        ),
        (
            "int",
            "active_steps",
            0,
            "Recorded steps per cycle. Zero records the whole context.",
            non_negative_int_validator,
        ),
        (
            "int",
            "repeat",
            1,
            "Capture cycles. Zero repeats until context exit.",
            non_negative_int_validator,
        ),
        (
            "int",
            "skip_first",
            0,
            "Unprofiled steps before the first capture cycle.",
            non_negative_int_validator,
        ),
    )
    for kind, suffix, default, doc, validator in declarations:
        kwargs = dict(
            common,
            default=default,
            doc=doc,
            env_default=f"NEURALQX_PROFILE_{suffix.upper()}",
        )
        if validator is not None:
            kwargs["validator"] = validator
        getattr(cfg, f"define_{kind}")(f"profile_{suffix}", **kwargs)
    for suffix, default, values, doc in (
        (
            "transport",
            "shared",
            ("shared", "tcp", "offline"),
            "Shared storage, TCP collection, or offline rank artifacts.",
        ),
        (
            "telemetry",
            "off",
            ("off", "auto", "nvml", "nvidia-smi"),
            "GPU counters. Auto tries NVML, then nvidia-smi, and reports unavailability.",
        ),
        (
            "telemetry_scope",
            "host",
            ("host", "rank"),
            "Sample once per hostname or in every rank.",
        ),
        (
            "synchronize",
            "boundary",
            ("none", "boundary"),
            "Wait for explicitly supplied results only at capture boundaries, or never.",
        ),
    ):
        cfg.define_enum(
            f"profile_{suffix}",
            default=default,
            values=values,
            doc=doc,
            env_default=f"NEURALQX_PROFILE_{suffix.upper()}",
            **common,
        )
    cfg.define_option(
        "profile_telemetry_devices",
        default=(),
        value_type=tuple,
        parser=parse_csv_tuple,
        doc="Management GPU UUIDs or indices. Empty selects all management-visible GPUs on a sampled host.",
        env_default="NEURALQX_PROFILE_TELEMETRY_DEVICES",
        **common,
    )
