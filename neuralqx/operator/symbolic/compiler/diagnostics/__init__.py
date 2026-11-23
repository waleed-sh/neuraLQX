# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
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


"""Compiler diagnostics models, rule runners, and formatting helpers."""

from neuralqx.operator.symbolic.compiler.diagnostics.collector import (
    filter_diagnostics_by_minimum_severity,
)
from neuralqx.operator.symbolic.compiler.diagnostics.collector import (
    run_default_diagnostics,
)
from neuralqx.operator.symbolic.compiler.diagnostics.formatting import format_diagnostic
from neuralqx.operator.symbolic.compiler.diagnostics.formatting import (
    format_diagnostics_block,
)
from neuralqx.operator.symbolic.compiler.diagnostics.models import DiagnosticSeverity
from neuralqx.operator.symbolic.compiler.diagnostics.models import DSLDiagnostic
from neuralqx.operator.symbolic.compiler.diagnostics.models import (
    count_diagnostics_by_severity,
)
from neuralqx.operator.symbolic.compiler.diagnostics.models import (
    diagnostic_severity_at_least,
)
from neuralqx.operator.symbolic.compiler.diagnostics.models import (
    diagnostic_severity_rank,
)
from neuralqx.operator.symbolic.compiler.diagnostics.models import (
    normalize_diagnostic_severity,
)

__all__ = [
    "DSLDiagnostic",
    "DiagnosticSeverity",
    "count_diagnostics_by_severity",
    "diagnostic_severity_at_least",
    "diagnostic_severity_rank",
    "filter_diagnostics_by_minimum_severity",
    "format_diagnostic",
    "format_diagnostics_block",
    "normalize_diagnostic_severity",
    "run_default_diagnostics",
]
