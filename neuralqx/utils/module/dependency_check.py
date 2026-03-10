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
from typing import Tuple, Sequence, Union
import importlib

from .version_check import get_module_version, get_module_version_string

Version = Tuple[int, int, int]


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
    minimum: Union[Version, None] = None
    maximum: Union[Version, None] = None
    rationale: str = ""


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

    installed = get_module_version(policy.name)

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
    installed_tuple_str = ".".join(map(str, installed))
    installed_raw_str = get_module_version_string(policy.name)

    constraints = []
    if policy.minimum:
        constraints.append(f">= {'.'.join(map(str, policy.minimum))}")
    if policy.maximum:
        constraints.append(f"<= {'.'.join(map(str, policy.maximum))}")

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


_DEFAULT_POLICIES = (
    DependencyPolicy(
        name="flax",
        minimum=(0, 6, 5),
        rationale=(
            "Versions prior to 0.5 did not properly support complex valued layers. As Flax is "
            "NetKet's default neural-network library, using older version of Flax may not support "
            "complex valued wavefunctions. This can lead to the inability to correctly solve "
            "constraints which are negative."
        ),
    ),
    DependencyPolicy(
        name="netket",
        minimum=(3, 19, 0),
        rationale=(
            "neuraLQX requires a NetKet version above 3.19.0 for JAX sharding support."
        ),
    ),
    DependencyPolicy(
        name="jax",
        minimum=(0, 5, 0),
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
