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
Custom neuraLQX errors implementation
"""

import warnings
from textwrap import dedent
from typing import Union

from neuralqx import cfg


def _error_anchor_url(error_name: str) -> str:
    base = cfg.get_static("Errors Directory")
    return f"{base.rstrip('/')}" + f"#{error_name.lower()}"


def new_error_content(message: str, *, error_name: str | None = None):
    base = cfg.get_static("Errors Directory")
    specific = _error_anchor_url(error_name) if error_name else None

    specific_block = "\n\nThis specific error:\n\t" + specific if specific else ""

    return (
        f"{dedent(message)}"
        f"\n"
        f"\n================================================================================================================="
        f"\n"
        f"You can find a list of all neuraLQX errors and warnings including their "
        f"\ndetailed explanations at:"
        f"\n\t {base}{specific_block}"
        f"\n================================================================================================================="
        f"\n"
    )


def _nqx_formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = _nqx_formatwarning


class neuralqxError(Exception):
    def __init__(self, msg: str):
        super().__init__(new_error_content(msg, error_name=self.__class__.__name__))


class neuralqxWarning(Warning):
    def __init__(self, msg: str, stack_level: int = 2):

        self.msg = new_error_content(msg, error_name=self.__class__.__name__)
        super().__init__(self.msg)
        warnings.warn(self, stacklevel=stack_level)

    def __str__(self):
        return self.msg


class GraphUnavailableWarning(neuralqxWarning):
    def __init__(self, stack_level: int = 2):
        super().__init__(
            "\n"
            "Unable to export graph."
            "\n"
            "You have set the parameter `plot = False` when creating the graph.",
            stack_level=stack_level,
        )


class DeniedExperimentalModuleImportError(neuralqxError):

    def __init__(self):
        super().__init__(
            "\n"
            "Attempted to import an experimental module."
            "\n"
            "To enable the use of the experimental module, set the environment"
            " variable `NQX_EXPERIMENTAL` to `1` before importing neuralqx as follows:"
            "\n"
            "\n\t>>> import os"
            "\n\t>>> os.environ['NQX_EXPERIMENTAL'] = '1'"
            "\n\t>>> import neuralqx as nqx"
        )


class DeniedExperimentalFeatureError(neuralqxError):

    def __init__(self, name: str):
        super().__init__(
            "\n"
            f"The use of the class '{name}' is currently experimental which means "
            f"that it may produce wrong or unexpected behaviour sometimes and it may"
            f" change or be removed in future releases.\n\n"
            "\n"
            f"To enable the use of `{name}`, set the environment"
            " variable `NQX_EXPERIMENTAL` to `1` before importing neuralqx as follows:"
            "\n"
            "\n\t>>> import os"
            "\n\t>>> os.environ['NQX_EXPERIMENTAL'] = '1'"
            "\n\t>>> import neuralqx as nqx"
        )


class IncorrectMonitoringValueError(neuralqxError):

    def __init__(
        self,
        name: str,
        avail: list,
    ):
        super().__init__(
            "\n"
            f"Attempted to choose the loss metric `{name}` for live monitoring but it "
            f"does not exist in the list of allowed values to be monitored. Please select one "
            f"of the following loss metrics to monitor:"
            f"\n\t{avail}"
        )


class AcceptanceUnavailableError(neuralqxError):

    def __init__(self, name: str):
        super().__init__(
            "\n"
            f"The `{name}` callback cannot be used with a sampler of type `ExactSampler` "
            f"since this sampler uses exact inference methods to which acceptance "
            f"does not apply."
        )


class IncorrectEdgeFormatError(neuralqxError):

    def __init__(self):
        super().__init__(
            "\n"
            "You entered the edges in an unrecognised format."
            "\nFor planar graphs, edges should be in a format of: "
            "\n\t[[a, b], [c, d], ...]"
            "\nwhere a, b, c, d, ... are positive integers labeling the vertices."
            "\n\nFor non-planar graphs, edges should be in a format of:"
            "\n\t[[(a, b, c), (d, e, f)], [(g, h, i), (j, k, l)], ...]"
            "\nwhere now, vertices are labeled by triplets of the form (x, y, z), "
            "where x, y, z, ... are positive integers."
        )


class UnspecifiedGaugeFixingError(neuralqxError):

    def __init__(self):
        super().__init__(
            "\n"
            "Attempted creating a gauge invariant Hilbert space without specifying "
            "a gauge fixing array."
        )


class CyclicGaugeFixingError(neuralqxError):

    def __init__(self):
        super().__init__(
            "\n"
            "You have specified a gauge fixing which contains a cycle and therefore "
            "does not form a directed acyclic graph (e.g. edge A depends on B, B "
            "depends on C, and C depends on A)."
            "\n\nTopological sort cannot be applied. Please try to rewrite your gauge fixing "
            "such that it contains no cycles (forms a directed acyclic graph)."
            "\n\nNote: if you set `auto_constraint` to `True` when creating the Hilbert space "
            "and do not provide a gauge fixing array, we will attempt to generate one for you."
        )


class IncorrectGaugeFixingArrayError(neuralqxError):

    def __init__(
        self,
        reason: str,
    ):
        super().__init__(
            "\n"
            "You have entered a gauge fixing array which is not correct.\n"
            f"\nReason: {reason}"
            f"\n\nThe correct format should be:"
            "\n[[['(A, B)'], ['(C, D)', '(E, F)', ...]], [['(G, H)'], ['(I, J)', '(K, L)', "
            "...]], ...]"
            "\nto produce a constraint that imposes:"
            "\nstate[a] = state[c] + state[d] + ... and"
            "\nstate[e] = state[g] + state[h] + ... and"
            "\n..."
            "\nwhere 'a' is the dual representation of '(A, B)', ... "
        )


class InvalidFreeEdgeSelectionError(neuralqxError):

    def __init__(
        self,
        num_edges: int,
        num_free: int,
    ):
        super().__init__(
            "\n"
            f"You have chosen {num_edges} edges to be modified but your "
            f"graph has only {num_free} independent edges not fixed by other edges."
        )


class InvalidEdgeSelectionError(neuralqxError):

    def __init__(
        self,
        num_edges: int,
        size: int,
    ):
        super().__init__(
            "\n"
            f"You have chosen {num_edges} edges to be modified but your "
            f"space has only {size} edges."
        )


class CrossProductInHigherDimensionsError(neuralqxError):

    def __init__(self):
        super().__init__(
            "\n" "The cross product can only be taken if gauge_dimensions = 3."
        )


class HilbertSpaceGaugeGroupDimensionsMismatchError(neuralqxError):

    def __init__(self):
        super().__init__(
            "\n"
            "The gauge dimensions of your Hilbert space and your gauge group do not match."
        )


class MissingPenaltyFactorError(neuralqxError):

    def __init__(self):
        super().__init__(
            "\n"
            "The Penalty operator you constructed does not have a `factor` attribute in it. "
            "Please make sure that you add this attribute to your operator as it is needed in "
            "weighing how much the penalty term contributes to the total cost."
        )


class InvalidChargeError(neuralqxError):

    def __init__(
        self,
        charge: int,
        cutoff: int,
    ):
        super().__init__(
            "\n"
            f"You have chosen a charge {charge} but the maximal allowed charge in your "
            f"Hilbert spaec is {cutoff}."
        )


class ExpectationValueError(neuralqxError):

    def __init__(self, func: str):
        super().__init__("\n" f"Cannot call `{func}()` for an empty list of operators.")


class InvalidOperatorsSequenceError(neuralqxError):

    def __init__(self, func: str):
        super().__init__(
            "\n"
            f"To use `{func}()` on more than one operator, the operators must be grouped "
            f"together in a python list."
        )


class InvalidOperatorsInSequenceError(neuralqxError):

    def __init__(self, func: str):
        super().__init__(
            "\n"
            f"You have specified at least on operator in your list of operators to be of `Squared` "
            f"type. In that case, all elements in the list specified to `{func}()` must be either "
            f"`Squared` or `PenaltyCost` operators."
        )


class InvalidCutoffError(neuralqxError):

    def __init__(self, dtype_cutoff: type, dtype_step: type):
        super().__init__(
            "\n"
            "Currently, neuraLQX only supports integer integer valued cutoffs and steps (e.g. your"
            " set of allowed basis labels should all be integers). You have chosen either a step or"
            f" a cutoff of type {dtype_cutoff} and a step of type {dtype_step}. "
            f"Half integer (float) steps or cutoffs are still under "
            f"development and will be available in a later release."
        )


class InvalidMelsFunctionError(neuralqxError):

    def __init__(self):
        super().__init__(
            "\n"
            "If you wish to use the `FunctionalLocalOperator` class and utilise a function to "
            "alter the produced matrix elements, this function needs to be in the form of a lambda "
            "expression. For example, instead of defining "
            "\n\tmels_func = numpy.sqrt"
            "\nuse"
            "\n\tmels_func = lambda x: numpy.sqrt(x)"
        )


class IncompatibleJaxOperatorError(neuralqxError):

    def __init__(self, op_type: str):
        super().__init__(
            "\n"
            f"The operator type `{op_type}` cannot be converted to a Jax friendly local operator."
        )


class DuplicateEdgesError(neuralqxError):

    def __init__(self):
        super().__init__(
            "\n"
            "Currently, neuraLQX does not support multiple edges between two nodes. If you need "
            "to use this, create 'ghost' vertices for each of the multiple edges between the nodes."
            "\n"
            "This is a feature under development and will be available in a future release."
        )


class ExceededDtypeValuesError(neuralqxError):

    def __init__(self, dtype: str):
        super().__init__(
            "\n"
            "You have selected to create a Hilbert space with only positive quantum numbers but "
            "have chosen a cutoff limit high enough that it exceeded the limits of the dtype of "
            f"the space (dtype: {dtype}). Please select a lower cutoff."
        )


class OutOfRangeIndexError(neuralqxError):

    def __init__(
        self,
        index: int,
        max_index: int,
    ):
        super().__init__(
            "\n"
            f"The index `{index}` is out of range for the available mapping. The maximum index "
            f"available is {max_index}."
        )


class InvalidIndexError(neuralqxError):

    def __init__(self, edge: Union[list, tuple]):
        super().__init__(
            "\n"
            f"The requested edge `{edge}` does not correspond to any index in the mapping."
        )


class NonExistentNonPlanarEdgesError(neuralqxError):

    def __init__(self):
        super().__init__(
            "\n"
            "The current graph is a planar graph and has no representation for its edges in the "
            "non-planar representation."
        )


class OrientationValenceMismatchError(neuralqxError):

    def __init__(self, valence: int, orientation_len: int):
        super().__init__(
            "\n"
            f"You have requested a vertex of valence {valence} but have specified the orientation "
            f"for only {orientation_len} edges. Please make sure that the number of orientations "
            f"match the requested valence."
        )


class NonExistentNonPlanarVerticesError(neuralqxError):

    def __init__(self):
        super().__init__(
            "\n"
            "The current graph is a planar graph and has no representation for its vertices in the "
            "non-planar representation."
        )


class DistributedRuntimeUnavailableWarning(neuralqxWarning):
    def __init__(self, stack_level: int = 2):
        super().__init__(
            "\n"
            "Distributed execution was requested but the distributed runtime is not available. "
            "Falling back to serial mode."
            "\n"
            "Please make sure your JAX distributed launch is correctly configured and retry."
            "\n\n"
            "\t>>> python your_script.py"
            "\n\n"
            "to run in serial mode, or launch with your cluster/distributed runner to enable "
            "multi-process execution.",
            stack_level=stack_level,
        )


class DistributedStateImportInSerialModeError(neuralqxError):

    def __init__(self, n_nodes, ranks_per_node):
        super().__init__(
            "\n"
            f"Attempted to load a state saved in distributed mode (Total processes = {n_nodes}, "
            f"processes per host = {ranks_per_node}) in a serial environment. "
            f"\n\n"
            f"This is currently not allowed. Please configure your distributed runtime to match the "
            f"one the state was exported from and try again."
        )


class DistributedStateImportMismatchError(neuralqxError):

    def __init__(self, mismatch_type: str, saved_data, current_data):

        qualifier = (
            "Total processes" if mismatch_type == "nodes" else "Processes per host"
        )

        super().__init__(
            "\n"
            f"Attempted to load a state saved with a different distributed configuration."
            f"\n\n"
            f"The state was saved with {qualifier} = {saved_data} but the current "
            f"{qualifier} = {current_data}. "
            f"This is currently not allowed. Please configure your runtime to match the one the "
            f"state was exported from and try again."
        )


class NonHermitianInverseCostError(neuralqxError):
    def __init__(self):
        super().__init__(
            "\n"
            f"The operator provided is not Hermitian. Currently, this wrapper can only"
            f"\nbe used with Hermitian operators as we use the standard covariance"
            f"\nmethod to compute its expectation value."
        )


class InvalidSurfaceError(neuralqxError):
    def __init__(self, s_type):
        super().__init__(
            "\n"
            f"The surface should be a list or tuple of edges. You provided `{s_type}`."
        )


class AreaDifferenceEdgesError(neuralqxError):
    def __init__(self, num_edges):
        super().__init__(
            "\n"
            f"The area difference operator requires two edges. You provided `{num_edges}`."
        )


class AreaDifferenceSurfacesError(neuralqxError):
    def __init__(self, num_edges):
        super().__init__(
            "\n"
            f"The area difference operator between two surfaces requires two surfaces. "
            f"You provided `{num_edges}`."
        )


class IncompatibleNonGIOperatorError(neuralqxError):
    def __init__(self):
        super().__init__(
            "\n"
            f"The operator you requested requires that you work in non gauge invariant Hilbert "
            f"\nspaces, but you currently implemented a gauge invariant space."
        )


class AutoConstraintGaugeFixingConflictError(neuralqxError):
    def __init__(self):
        super().__init__(
            "\n"
            "You have set the parameter `auto_constraint` to `True` but have also provided a "
            "set of gauge fixings. "
            "\n"
            "Please choose only one (e.g. either set `auto_constraint` to `True` to automatically "
            "generate the gauge fixing or"
            "\n"
            "provide the gauge fixing yourself and set `auto_constraint` to `False`)."
            ""
        )


class IncompatibleNonPlanarGraphModel(neuralqxError):
    def __init__(self):
        super().__init__(
            "\n"
            "You have provided a non-planar graph for a model which is compatible only with planar "
            "graphs."
            "\n"
            "Please use a planar graph for this model."
        )


class IncompatiblePlanarGraphModel(neuralqxError):
    def __init__(self):
        super().__init__(
            "\n"
            "You have provided a planar graph for a model which is compatible only with non-planar "
            "graphs."
            "\n"
            "Please use a non-planar graph for this model."
        )


class IncompatibleHilbertSpaceError(neuralqxError):
    def __init__(self, available, required):
        super().__init__(
            "\n"
            f"The Hilbert space you have provided is of type `{available.__name__}`, "
            f"\n"
            f"but this class requires a Hilbert space of type `{required.__name__}`."
        )


class IncompatibleModdedOperatorWarning(neuralqxWarning):
    def __init__(self, stack_level: int = 2):
        super().__init__(
            "\n"
            "The modded operator is only available with the computational backend.\n"
            "The constructed LocalOperator does NOT respect modular arithmetic.",
            stack_level=stack_level,
        )


class RandomEmbeddingForPlanarGraphWarning(neuralqxWarning):
    def __init__(self, stack_level: int = 2):
        super().__init__(
            "\n"
            "Random embedding is not available for planar graphs."
            "\n"
            "The random embedding option has been ignored.",
            stack_level=stack_level,
        )


class LiveMonitoringUnavailableWarning(neuralqxWarning):
    def __init__(self, stack_level: int = 2):
        super().__init__(
            "\n" "Live monitoring is disabled when conducting distributed simulations.",
            stack_level=stack_level,
        )


class ComputationalModelConcretizationWarning(neuralqxWarning):
    def __init__(self, stack_level: int = 2):
        super().__init__(
            "\n"
            "This model only supports computational operators as a backend (or at least the main constraint is only"
            " implemented as a computational operator)."
            "\n"
            "The `computational=False` option has been ignored.",
            stack_level=stack_level,
        )


class SuboptimalOperatorForGPUWarning(neuralqxWarning):
    def __init__(self, stack_level: int = 2):
        super().__init__(
            "\n"
            "The operator you created is a subclass of ComputationalOperator. When using GPUs, this type is "
            "suboptimal. We recommend using the ComputationalJaxOperator type instead."
            "\n"
            "If this is an operator available in neuraLQX, you can find the JAX variant in the "
            "`neuralqx.operators.computational` module. If this is a custom operator, consider "
            "implementing the JAX variant for GPU use.",
            stack_level=stack_level,
        )


__all__ = [
    "GraphUnavailableWarning",
    "DeniedExperimentalFeatureError",
    "DeniedExperimentalModuleImportError",
    "IncompatibleNonGIOperatorError",
    "InvalidSurfaceError",
    "InvalidIndexError",
    "InvalidChargeError",
    "InvalidCutoffError",
    "InvalidMelsFunctionError",
    "IncompatibleJaxOperatorError",
    "InvalidOperatorsSequenceError",
    "InvalidOperatorsInSequenceError",
    "InvalidEdgeSelectionError",
    "InvalidFreeEdgeSelectionError",
    "IncorrectEdgeFormatError",
    "IncorrectMonitoringValueError",
    "IncorrectGaugeFixingArrayError",
    "NonHermitianInverseCostError",
    "AutoConstraintGaugeFixingConflictError",
    "DistributedStateImportMismatchError",
    "OutOfRangeIndexError",
    "CrossProductInHigherDimensionsError",
    "DistributedStateImportInSerialModeError",
    "DistributedRuntimeUnavailableWarning",
    "AreaDifferenceEdgesError",
    "DuplicateEdgesError",
    "OrientationValenceMismatchError",
    "AreaDifferenceSurfacesError",
    "ExpectationValueError",
    "MissingPenaltyFactorError",
    "NonExistentNonPlanarEdgesError",
    "NonExistentNonPlanarVerticesError",
    "ExceededDtypeValuesError",
    "CyclicGaugeFixingError",
    "AcceptanceUnavailableError",
    "UnspecifiedGaugeFixingError",
    "HilbertSpaceGaugeGroupDimensionsMismatchError",
    "IncompatibleNonPlanarGraphModel",
    "IncompatiblePlanarGraphModel",
    "IncompatibleHilbertSpaceError",
    "IncompatibleModdedOperatorWarning",
    "RandomEmbeddingForPlanarGraphWarning",
    "LiveMonitoringUnavailableWarning",
    "ComputationalModelConcretizationWarning",
    "SuboptimalOperatorForGPUWarning",
]
