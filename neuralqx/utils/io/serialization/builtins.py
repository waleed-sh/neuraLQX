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


"""Reconstruction recipes for graph, Hilbert and sampler configuration."""

import dataclasses

import jax

from .registry import _register_codec as register_codec


def _fields(value):
    if hasattr(type(value), "fields"):
        return {
            name: getattr(value, name)
            for name, field in type(value).fields().items()
            if field.init
        }
    return {
        field.name: getattr(value, field.name)
        for field in dataclasses.fields(value)
        if field.init
    }


def _sampler_config(value):
    config = _fields(value)
    config.pop("n_chains_per_device", None)
    for name in ("importance_sampling", "operator_blur"):
        if config.get(name) is None:
            config.pop(name, None)
    return config


def _graph_config(graph):
    return {
        "edges": graph.edges,
        "is_planar": graph.is_planar,
        "original_edges": graph._original_edges,
        "embedding_map": graph._embedding_map,
    }


def _restore_graph(config):
    from neuralqx.graph import Graph

    graph = Graph(config["edges"], is_planar=config["is_planar"])
    graph._original_edges = config["original_edges"]
    graph._embedding_map = config["embedding_map"]
    return graph


def install_codecs():
    """Register a fixed set of installed types, never types named by a checkpoint."""
    from neuralqx import graph
    from neuralqx.hilbert import HeterogeneousDiscreteHilbert
    from neuralqx.hilbert import HomogeneousDiscreteHilbert
    from neuralqx.hilbert import SU2GaugeInvariantHilbert
    from neuralqx.hilbert import U1GaugeInvariantHilbert
    from neuralqx.hilbert import U1Hilbert
    from neuralqx.hilbert.constraint import AndConstraint
    from neuralqx.hilbert.constraint import IdentityConstraint
    from neuralqx.hilbert.constraint import LinearConstraint
    from neuralqx.hilbert.constraint import NotConstraint
    from neuralqx.hilbert.constraint import OrConstraint
    from neuralqx.hilbert.local import ExplicitLocalSpace
    from neuralqx.hilbert.local import HeterogeneousLocalSpace
    from neuralqx.hilbert.local import LocalRange
    from neuralqx.hilbert.local import VectorRange
    from neuralqx.hilbert.su2.peter_weyl import SU2PeterWeylHilbert
    from neuralqx.hilbert.u1.constraints.relation import GaugeRelation
    from neuralqx.nn.utils.connections import ConnectionPolicy
    from neuralqx.operator import ProductOperator
    from neuralqx.operator import ScaledOperator
    from neuralqx.operator import Squared
    from neuralqx.operator import SumOfSquares
    from neuralqx.operator import SumOperator
    from neuralqx.operator.prebuilt.su2 import CompilationBudget
    from neuralqx.operator.prebuilt.su2 import EuclideanVertexConstraint
    from neuralqx.sampler import MetropolisSampler
    from neuralqx.sampler.importance import ImportanceSampling
    from neuralqx.sampler.importance import OperatorBlur
    from neuralqx.sampler.rules import ExchangeRule
    from neuralqx.sampler.rules import LocalRule
    from neuralqx.sampler.rules import NeighborRule
    from neuralqx.sampler.rules import SU2ExchangeRule
    from neuralqx.sampler.rules import SU2MetropolisRule
    from neuralqx.sampler.rules import U1GaugeRule
    from neuralqx.utils.typing.graph import Edge
    from neuralqx.utils.typing.graph import Vertex

    classes = (
        ImportanceSampling,
        OperatorBlur,
        CompilationBudget,
        EuclideanVertexConstraint,
        SumOperator,
        ScaledOperator,
        ProductOperator,
        Squared,
        SumOfSquares,
        HomogeneousDiscreteHilbert,
        HeterogeneousDiscreteHilbert,
        U1Hilbert,
        U1GaugeInvariantHilbert,
        SU2GaugeInvariantHilbert,
        SU2PeterWeylHilbert,
        GaugeRelation,
        IdentityConstraint,
        LinearConstraint,
        AndConstraint,
        OrConstraint,
        NotConstraint,
        LocalRange,
        ExplicitLocalSpace,
        VectorRange,
        HeterogeneousLocalSpace,
        ConnectionPolicy,
        LocalRule,
        U1GaugeRule,
        SU2MetropolisRule,
        SU2ExchangeRule,
        ExchangeRule,
        NeighborRule,
        Vertex,
        Edge,
    )
    for cls in classes:
        register_codec(
            cls,
            name=f"{cls.__module__}.{cls.__qualname__}",
            encode=_fields,
            decode=lambda config, cls=cls: cls(**config),
        )
    register_codec(
        MetropolisSampler,
        name="neuralqx.MetropolisSampler",
        encode=_sampler_config,
        decode=lambda config: MetropolisSampler(**config),
    )
    # Only installed public operator classes are eligible for reconstruction.
    # Custom operators can register their own versioned recipe.
    from neuralqx.operator import AbstractOperator
    from neuralqx.operator.prebuilt import bojowald_swiderski
    from neuralqx.operator.prebuilt import coloring
    from neuralqx.operator.prebuilt import qr
    from neuralqx.operator.prebuilt import su2
    from neuralqx.operator.prebuilt import wcl

    from .registry import _BY_TYPE

    for module in (wcl, su2, qr, bojowald_swiderski, coloring):
        for name in module.__all__:
            cls = getattr(module, name)
            if (
                isinstance(cls, type)
                and issubclass(cls, AbstractOperator)
                and cls not in _BY_TYPE
            ):
                register_codec(
                    cls,
                    name=f"{cls.__module__}.{cls.__qualname__}",
                    encode=_fields,
                    decode=lambda config, cls=cls: cls(**config),
                )

    graph_codec = register_codec(
        graph.Graph,
        name="neuralqx.graph.Graph",
        encode=_graph_config,
        decode=_restore_graph,
    )
    for name in graph.__all__:
        cls = getattr(graph, name)
        if isinstance(cls, type) and issubclass(cls, graph.Graph):
            _BY_TYPE[cls] = graph_codec
    register_codec(
        jax.lax.Precision,
        name="jax.lax.Precision",
        encode=lambda value: value.name,
        decode=lambda name: jax.lax.Precision[name],
    )
    from .models import install_model_codecs

    install_model_codecs()
