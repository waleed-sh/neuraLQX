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
This module defines explicit version constraints for mandatory dependencies whose correctness
guarantees (numerical accuracy, parallel safety, or API stability) are relied upon by neuraLQX.
"""

from dataclasses import dataclass
from typing import Sequence, Union
import importlib

from .version import InvalidVersion
from .version import Version
from .version import VersionInput
from .version import parse_version
from .version import get_module_version_string


class DependencyViolation(RuntimeError):
    """
    Raised when a dependency violates an enforced compatibility policy.
    """

    pass


@dataclass(frozen=True)
class DependencyPolicy:
    """
    Declarative specification of a dependency compatibility constraint.
    """

    name: str
    minimum: Union[VersionInput, None] = None
    maximum: Union[VersionInput, None] = None
    rationale: str = ""

    def __post_init__(self) -> None:
        min_version = _coerce_policy_bound(self.minimum, field_name="minimum")
        max_version = _coerce_policy_bound(self.maximum, field_name="maximum")

        if (
            min_version is not None
            and max_version is not None
            and min_version > max_version
        ):
            raise ValueError(
                f"DependencyPolicy('{self.name}') has an invalid range: "
                f"minimum {min_version} is greater than maximum {max_version}."
            )

        object.__setattr__(self, "minimum", min_version)
        object.__setattr__(self, "maximum", max_version)


def enforce_policy(policy: DependencyPolicy) -> None:
    """
    Enforce a single dependency policy.

    :param policy: The dependency constraint to enforce.

    :raises DependencyViolation: If the dependency is missing or violates the version policy.
    """
    try:
        importlib.import_module(policy.name)
    except ModuleNotFoundError as exc:
        raise DependencyViolation(_format_missing_dependency(policy)) from exc

    installed_raw = get_module_version_string(policy.name)
    if installed_raw == "unknown":
        installed = Version("0.0.0")
    else:
        try:
            installed = parse_version(installed_raw)
        except (TypeError, ValueError, InvalidVersion):
            raise DependencyViolation(
                _format_unparseable_version_error(policy, installed_raw)
            ) from None

    if policy.minimum and installed < policy.minimum:
        raise DependencyViolation(
            _format_version_error(
                policy,
                installed,
                relation="below the supported minimum",
            )
        )

    if policy.maximum and installed > policy.maximum:
        raise DependencyViolation(
            _format_version_error(
                policy,
                installed,
                relation="above the supported maximum",
            )
        )


def enforce_policies(policies: Sequence[DependencyPolicy]) -> None:
    """
    Enforce a sequence of dependency policies. Evaluation stops at the first violation.
    """
    for policy in policies:
        enforce_policy(policy)


def _format_missing_dependency(policy: DependencyPolicy) -> str:
    return (
        f"Dependency check failed: required package not found.\n\n"
        f"Package name:\n"
        f"  {policy.name}\n\n"
        f"Why this dependency is required:\n"
        f"  {policy.rationale or 'This package is required for correct execution of neuraLQX.'}\n\n"
        f"Impact:\n"
        f"  neuraLQX cannot guarantee correctness or stability without this dependency.\n\n"
        f"Suggested action:\n"
        f"  Install the missing package and retry."
    )


def _format_version_error(
    policy: DependencyPolicy,
    installed: Version,
    *,
    relation: str,
) -> str:
    installed_tuple = installed.as_tuple(width=max(3, len(installed.release)))
    installed_tuple_str = ".".join(map(str, installed_tuple))
    installed_raw_str = get_module_version_string(policy.name)

    constraints = []
    if policy.minimum:
        constraints.append(f">= {policy.minimum}")
    if policy.maximum:
        constraints.append(f"<= {policy.maximum}")

    return (
        f"Dependency version incompatibility detected.\n\n"
        f"Package:\n"
        f"  {policy.name}\n\n"
        f"Installed version:\n"
        f"  {installed_raw_str} (parsed as {installed_tuple_str})\n\n"
        f"Required version policy:\n"
        f"  {' and '.join(constraints)}\n\n"
        f"Problem:\n"
        f"  The installed version is {relation}.\n\n"
        f"Why this matters:\n"
        f"  {policy.rationale or 'This version mismatch may lead to incorrect numerical results, runtime failures, or unsupported execution paths.'}\n\n"
        f"Suggested action:\n"
        f"  Adjust your environment to satisfy the required version policy and retry."
    )


def _format_unparseable_version_error(
    policy: DependencyPolicy,
    installed_raw: str,
) -> str:
    constraints = []
    if policy.minimum:
        constraints.append(f">= {policy.minimum}")
    if policy.maximum:
        constraints.append(f"<= {policy.maximum}")

    return (
        f"Dependency version incompatibility detected.\n\n"
        f"Package:\n"
        f"  {policy.name}\n\n"
        f"Installed version:\n"
        f"  {installed_raw} (unparseable)\n\n"
        f"Required version policy:\n"
        f"  {' and '.join(constraints)}\n\n"
        f"Problem:\n"
        f"  The installed version string could not be interpreted as a comparable version.\n\n"
        f"Why this matters:\n"
        f"  {policy.rationale or 'Version-dependent safety checks cannot be enforced when the dependency version cannot be parsed.'}\n\n"
        f"Suggested action:\n"
        f"  Install a standard release of this dependency (PEP 440 compliant) and retry."
    )


def _coerce_policy_bound(
    value: Union[VersionInput, None],
    *,
    field_name: str,
) -> Union[Version, None]:
    if value is None:
        return None

    try:
        parsed = parse_version(value)
    except (TypeError, ValueError, InvalidVersion):
        raise ValueError(
            f"DependencyPolicy.{field_name} for '{value}' is not a valid version. "
            f"Please provide a PEP 440 compliant version string or tuple."
        ) from None

    return parsed


_DEFAULT_POLICIES = (
    DependencyPolicy(
        name="flax",
        minimum="0.6.5",
        rationale=(
            "Versions prior to 0.5 did not properly support complex valued layers. As Flax is "
            "NetKet's default neural-network library, using older version of Flax may not support "
            "complex valued wavefunctions. This can lead to the inability to correctly solve "
            "constraints which are negative."
        ),
    ),
    DependencyPolicy(
        name="netket",
        minimum="3.19.0",
        rationale=(
            "neuraLQX requires a NetKet version above 3.19.0 for JAX sharding support."
        ),
    ),
    DependencyPolicy(
        name="jax",
        minimum="0.5.0",
        rationale=(
            "JAX versions below 0.5.0 are currently not supported. This is because some modified "
            "code requires certain NetKet and JAX versions for just-in-time compilation of "
            "some functions."
        ),
    ),
)


def enforce_default_dependencies() -> None:
    """
    Enforce all mandatory neuraLQX dependency policies.
    """
    enforce_policies(_DEFAULT_POLICIES)


enforce_default_dependencies()
