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


"""Package exception hierarchy and compatibility with Python exception categories.

All exceptions raised deliberately by neuraLQX inherit from neuralqxError.
The built-in-compatible classes also retain their Python category, arguments,
and operating-system fields. Third-party failures keep their original type
unless a package boundary explicitly translates them. StopRun is successful
callback control flow and is deliberately outside the error hierarchy.
"""

from __future__ import annotations

import sys
import warnings
from textwrap import dedent

_ERROR_SEPARATOR = "=" * 113


def _documentation_base() -> str:
    module = sys.modules.get("neuralqx.config")
    manager = getattr(module, "config", None)
    if manager is not None:
        return manager.get_static("Errors Directory")
    return "https://neuralqx.readthedocs.io/en/latest/guides/errors.html"


def new_error_content(message: str, *, error_name: str | None = None) -> str:
    """Attach documentation links to a struct diagnostic or legacy warning.

    Args:
        message: Human-readable failure or warning description.
        error_name: Exception class used to select a documentation anchor.

    Returns:
        Dedented text with the configured documentation location. A default
        location is used while configuration itself is being initialized.
    """
    base = _documentation_base()
    specific = f"{base.rstrip('/')}#{error_name.lower()}" if error_name else None
    specific_block = f"\n\nThis specific error:\n\t{specific}" if specific else ""
    return (
        f"{dedent(message)}"
        f"\n"
        f"\n{_ERROR_SEPARATOR}"
        f"\n"
        f"You can find a list of all neuraLQX errors and warnings including their "
        f"\ndetailed explanations at:"
        f"\n\t {base}{specific_block}"
        f"\n{_ERROR_SEPARATOR}"
        f"\n"
    )


def _nqx_formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = _nqx_formatwarning


class neuralqxError(Exception):
    """Base for package failures, preserving ordinary exception argument semantics.

    Catch this class to handle neuraLQX-generated failures. Validation subclasses
    also inherit from their corresponding built-in exception, so existing
    ``except ValueError`` and similar handlers continue to work. Formatting is
    left to the caller except for the legacy struct diagnostics.
    """


class neuralqxWarning(Warning):
    """Legacy package warning that emits itself when explicitly constructed.

    Args:
        msg: Warning text before the documentation footer is added.
        stack_level: Stack depth passed to warnings.warn.
    """

    msg: str
    """Rendered warning text, including its documentation links."""

    def __init__(self, msg: str, stack_level: int = 2):
        """Format and emit one warning at the requested caller depth."""
        self.msg = new_error_content(msg, error_name=self.__class__.__name__)
        super().__init__(self.msg)
        warnings.warn(self, stacklevel=stack_level)

    def __str__(self) -> str:
        """Return the formatted warning text."""
        return self.msg


class NeuralqxValueError(neuralqxError, ValueError):
    """A package argument has an invalid value."""


class NeuralqxTypeError(neuralqxError, TypeError):
    """A package argument has an unsupported type."""


class NeuralqxRuntimeError(neuralqxError, RuntimeError):
    """An operation cannot complete in the current runtime state."""


class NeuralqxNotImplementedError(NeuralqxRuntimeError, NotImplementedError):
    """A requested package operation is not implemented."""


class NeuralqxIndexError(neuralqxError, IndexError):
    """An index lies outside a package-defined range."""


class NeuralqxKeyError(neuralqxError, KeyError):
    """A requested key is absent from a package mapping or registry."""


class NeuralqxAttributeError(neuralqxError, AttributeError):
    """A package attribute is absent or cannot be assigned."""


class NeuralqxArithmeticError(neuralqxError, ArithmeticError):
    """A numerical invariant failed during a package calculation."""


class NeuralqxOverflowError(NeuralqxArithmeticError, OverflowError):
    """A numerical result exceeds its package representation."""


class NeuralqxMemoryError(neuralqxError, MemoryError):
    """A requested allocation exceeds an explicit package guard."""


class NeuralqxOSError(neuralqxError, OSError):
    """A package operation failed at a filesystem or transport boundary."""


class NeuralqxFileExistsError(NeuralqxOSError, FileExistsError):
    """An output already exists and replacement was not requested."""


class NeuralqxFileNotFoundError(NeuralqxOSError, FileNotFoundError):
    """An expected package input or artifact is missing."""


class NeuralqxIsADirectoryError(NeuralqxOSError, IsADirectoryError):
    """An operation requiring a file received a directory."""


class NeuralqxPermissionError(NeuralqxOSError, PermissionError):
    """A package filesystem operation was denied."""


class NeuralqxTimeoutError(NeuralqxOSError, TimeoutError):
    """A package operation exceeded its configured deadline."""


class StructError(neuralqxError):
    """Base for struct and pytree failures, retaining their documentation footer."""

    def __init__(self, msg: str):
        """Format a struct diagnostic with its error documentation anchor.

        Args:
            msg: Plain diagnostic text.
        """
        super().__init__(new_error_content(msg, error_name=self.__class__.__name__))


class FrozenStructError(StructError, NeuralqxAttributeError):
    """Mutation was attempted on a frozen Struct instance."""


class SerializationError(StructError):
    """Struct or registered-pytree input or output failed."""


class ValidationError(StructError, NeuralqxTypeError):
    """A struct declaration or runtime value is invalid."""


class ConfigError(NeuralqxRuntimeError):
    """Base class for all neuraLQX configuration errors."""


class UnknownOptionError(ConfigError):
    """Raised when accessing or mutating an unknown configuration option."""


class ConfigValidationError(ConfigError):
    """Raised when a configuration value fails parsing or validation."""


class ProfileError(NeuralqxRuntimeError):
    """A requested capture, collection, or export could not be completed."""


class ProfileStateError(ProfileError):
    """An operation conflicts with the lifetime of a profiling session."""


class ProfileTimeoutError(ProfileError, NeuralqxTimeoutError):
    """A profiling peer did not respond before the configured deadline."""


class DependencyViolation(NeuralqxRuntimeError):
    """
    Raised when a dependency violates an enforced compatibility policy.
    """

    pass


class SU2HilbertError(NeuralqxValueError):
    """Base class for SU(2) Hilbert-space construction and state errors.

    All SU(2)-specific public exceptions inherit from this type so callers can
    catch configuration, layout, recoupling, and exact-enumeration failures
    without also catching unrelated :class:`ValueError` instances from lower
    level numerical code. The class is intentionally broad and carries no
    additional fields.
    """


class InvalidSU2CutoffError(SU2HilbertError):
    """Invalid maximum doubled-spin cutoff for an SU(2) Hilbert space.

    SU(2) labels are represented as non-negative doubled spins. The cutoff must
    therefore be an integer ``two_j_max >= 0``. This error is raised for missing
    cutoffs, booleans, non-integer values, negative values, or conflicting
    ``two_j_max`` and compatibility ``cutoff`` arguments.
    """

    def __init__(self, value: object) -> None:
        """Initializes the cutoff error message.

        Args:
            value: Invalid cutoff value supplied by the caller.

        The resulting message explains the non-negative integer requirement for
        doubled-spin cutoffs and echoes the offending value.
        """
        super().__init__(
            f"SU(2) Hilbert spaces require an integer two_j_max >= 0. Received {value!r}."
        )


class SU2UnconstrainedNotImplementedError(SU2HilbertError):
    """The spin-network constructor cannot select the magnetic-index basis.

    The SU2Hilbert alias intentionally represents the gauge-invariant
    spin-network basis. The unconstrained magnetic-index link basis has
    different state variables, validation rules, sampling behavior, and
    proposal kernels, so construction fails explicitly instead of silently
    producing the wrong representation. This avoids exposing misleading state
    layout or proposal semantics.
    """

    def __init__(self) -> None:
        """Initializes the unsupported-basis error message.

        The constructor takes no arguments because the error always describes
        the separate unconstrained SU(2) representation and points callers
        to its explicit constructor.
        """
        super().__init__(
            "The new SU(2) implementation represents the gauge-invariant "
            "spin-network basis. The unconstrained magnetic-index link basis "
            "is a different Hilbert space. Use SU2PeterWeylHilbert explicitly."
        )


class UnsupportedRecouplingSchemeError(SU2HilbertError):
    """Requested SU(2) vertex recoupling scheme is unavailable.

    Spin-network vertices currently use the ``"left_comb"`` recoupling scheme.
    This error is raised when a caller asks for another scheme so the boundary
    of the implemented API is clear at construction time and downstream layout
    assumptions stay explicit for every SU(2) Hilbert-space method.
    """

    def __init__(self, recoupling: object) -> None:
        """Initializes the recoupling-scheme error message.

        Args:
            recoupling: Unsupported recoupling scheme requested by the caller.

        The resulting message names the currently supported scheme and echoes
        the requested value for configuration diagnostics.
        """
        super().__init__(
            f"SU(2) currently supports recoupling='left_comb' only. Received {recoupling!r}."
        )


class InvalidSU2StateLayoutError(SU2HilbertError):
    """Flat SU(2) spin-network state has the wrong trailing shape.

    SU(2) helpers expect the final axis of each state array to match the
    spin-network layout size: edge doubled-spin labels followed by vertex
    channel labels. This error reports arrays whose trailing shape cannot be
    interpreted as such a flat state.
    """

    def __init__(self, expected_size: int, got: object) -> None:
        """Initializes the state-layout error message.

        Args:
            expected_size: Required trailing state-axis length.
            got: Observed trailing shape or shape-like object.

        The resulting message is intended for user-facing validation failures
        where a flat state cannot be interpreted by the SU(2) layout.
        """
        super().__init__(
            "Invalid SU(2) spin-network state shape: expected final axis "
            f"of length {expected_size}, got {got!r}."
        )


class SU2ChannelConstructionFailureError(SU2HilbertError):
    """Admissible recoupling channels cannot be built for fixed edge labels.

    Some proposed edge-spin sectors cannot satisfy the local SU(2)
    admissibility rules at a vertex. This error is raised by helpers that are
    asked to produce a completed physical state and cannot find compatible
    recoupling channels for the supplied edge labels.
    """

    def __init__(self, vertex: object, edge_spins: object) -> None:
        """Initializes the channel-construction error message.

        Args:
            vertex: Vertex identifier associated with the failure.
            edge_spins: Edge doubled-spin labels that could not be completed.

        The resulting message preserves the local edge sector so callers can
        report or debug the inadmissible spin configuration.
        """
        super().__init__(
            "Could not construct SU(2)-admissible recoupling channels at "
            f"vertex {vertex!r} for incident doubled-spins {edge_spins!r}."
        )


class SU2ExactEnumerationTooLargeError(SU2HilbertError):
    """Exact SU(2) basis enumeration exceeded an explicit guard.

    Exact enumeration is intended for small validation systems, deterministic
    tests, and cases where ranking or unranking is required. The guard prevents
    accidental materialization of very large spin-network bases that would be
    expensive in memory and time. Increase the explicit limit only when the
    basis size is known to be acceptable.
    """

    def __init__(self, max_states: int) -> None:
        """Initializes the exact-enumeration guard error message.

        Args:
            max_states: Explicit enumeration limit that was exceeded.

        The resulting message explains that exact enumeration is intentionally
        guarded and reports the active limit.
        """
        super().__init__(
            "Exact SU(2) enumeration is restricted to small validation "
            f"systems. The valid basis exceeded max_states={max_states}."
        )


class U1GaugeFixingError(NeuralqxValueError):
    """Base error for malformed U(1) gauge-fixing input.

    This exception covers invalid edge references, duplicate or inconsistent
    relation data, and topology metadata that cannot define a constructive
    gauge-fixing slice. It subclasses :class:`ValueError` because these
    failures indicate invalid user configuration rather than numerical
    execution failure. Catch this base class when all U(1) gauge-fixing
    configuration problems should be handled in the same way.
    """


class U1CyclicGaugeFixingError(U1GaugeFixingError):
    """Gauge-fixing relations cannot be ordered constructively.

    A constructive U(1) gauge fixing requires every slave edge to be
    reconstructible from free edges or from earlier slave edges. This exception
    is raised when the dependency graph contains a cycle, meaning no
    topological reconstruction order exists. The caller must change at least
    one relation to break the cycle.
    """


class CheckpointError(SerializationError):
    """An incomplete, incompatible, or malformed .nqx checkpoint was encountered."""

    message: str
    """Plain diagnostic text suitable for checkpoint metadata or collective errors."""

    def __init__(self, message: str):
        """Keep the plain reason alongside the standard package error formatting.

        Args:
            message: Diagnostic text suitable for collective error propagation.
        """
        self.message = message
        super().__init__(message)


class ReconstructionError(CheckpointError):
    """A saved component needs an explicitly supplied template or registered codec."""


class UnavailableOperatorError(NeuralqxNotImplementedError):
    """An intentionally absent model operation, distinct from a failed construction.

    The total constraint may omit an absent Euclidean or Lorentzian component.
    It must still propagate errors raised inside an implemented component,
    including ordinary NotImplementedError exceptions from an unavailable
    backend. This exception identifies only the model's declared absence.
    """

    name: str
    """Property whose default implementation is unavailable."""

    def __init__(self, model: str, name: str, detail: str = "") -> None:
        """Record the absent operation and name the concrete requesting model."""
        self.name = name
        message = f"{model} does not implement a default for {name}."
        super().__init__(f"{message} {detail}" if detail else message)


class StopRun(Exception):
    """Successful callback termination, deliberately outside neuralqxError."""


class HolomorphicUndeclaredWarning(neuralqxWarning, UserWarning):
    """Warning emitted when complex holomorphicity must be declared explicitly."""

    def __init__(self, message: str):
        """Store text for a single explicit warnings.warn call."""
        Warning.__init__(self, message)

    def __str__(self) -> str:
        """Return the text supplied by the differentiation-mode resolver."""
        return Warning.__str__(self)


__all__ = [
    "CheckpointError",
    "ConfigError",
    "ConfigValidationError",
    "DependencyViolation",
    "FrozenStructError",
    "HolomorphicUndeclaredWarning",
    "InvalidSU2CutoffError",
    "InvalidSU2StateLayoutError",
    "NeuralqxArithmeticError",
    "NeuralqxAttributeError",
    "NeuralqxFileExistsError",
    "NeuralqxFileNotFoundError",
    "NeuralqxIndexError",
    "NeuralqxIsADirectoryError",
    "NeuralqxKeyError",
    "NeuralqxMemoryError",
    "NeuralqxNotImplementedError",
    "NeuralqxOSError",
    "NeuralqxOverflowError",
    "NeuralqxPermissionError",
    "NeuralqxRuntimeError",
    "NeuralqxTimeoutError",
    "NeuralqxTypeError",
    "NeuralqxValueError",
    "ProfileError",
    "ProfileStateError",
    "ProfileTimeoutError",
    "ReconstructionError",
    "SU2ChannelConstructionFailureError",
    "SU2ExactEnumerationTooLargeError",
    "SU2HilbertError",
    "SU2UnconstrainedNotImplementedError",
    "SerializationError",
    "StopRun",
    "StructError",
    "U1CyclicGaugeFixingError",
    "U1GaugeFixingError",
    "UnavailableOperatorError",
    "UnknownOptionError",
    "UnsupportedRecouplingSchemeError",
    "ValidationError",
    "neuralqxError",
    "neuralqxWarning",
    "new_error_content",
]
