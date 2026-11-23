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


"""Rule registry for DSL diagnostics."""

from __future__ import annotations

from neuralqx.operator.symbolic.compiler.diagnostics.rules.base import (
    AbstractDiagnosticRule,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules.base import (
    DiagnosticRuleContext,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules.connectivity_rules import (
    GeneratedConnectivityValidityRule,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules.structural_rules import (
    ConstantFalsePredicateRule,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules.structural_rules import (
    DuplicateEmissionRule,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules.structural_rules import (
    MaxConnHintLowerBoundRule,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules.structural_rules import (
    MissingBranchTagRule,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules.symbol_rules import (
    PotentialZeroDivisionRule,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules.symbol_rules import (
    StaticIndexBoundsRule,
)
from neuralqx.operator.symbolic.compiler.diagnostics.rules.symbol_rules import (
    UnresolvedFreeSymbolsRule,
)


def default_diagnostic_rules() -> tuple[AbstractDiagnosticRule, ...]:
    """Builds the default ordered DSL diagnostics rule set.

    Returns:
        Ordered tuple of diagnostics rule instances.
    """
    return (
        UnresolvedFreeSymbolsRule(),
        StaticIndexBoundsRule(),
        PotentialZeroDivisionRule(),
        ConstantFalsePredicateRule(),
        DuplicateEmissionRule(),
        MaxConnHintLowerBoundRule(),
        MissingBranchTagRule(),
        GeneratedConnectivityValidityRule(),
    )


__all__ = [
    "AbstractDiagnosticRule",
    "ConstantFalsePredicateRule",
    "DiagnosticRuleContext",
    "DuplicateEmissionRule",
    "GeneratedConnectivityValidityRule",
    "MaxConnHintLowerBoundRule",
    "MissingBranchTagRule",
    "PotentialZeroDivisionRule",
    "StaticIndexBoundsRule",
    "UnresolvedFreeSymbolsRule",
    "default_diagnostic_rules",
]
