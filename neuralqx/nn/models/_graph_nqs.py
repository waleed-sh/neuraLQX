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

from dataclasses import field

from typing import Any
from typing import Dict
from typing import List

import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn


class LocalGraphNQS(nn.Module):
    r"""
    A graph-local, lightweight neural quantum state designed for gauge-theoretic lattice models
    (e.g. U(1)^3 weak-coupling LQG) defined on a fixed graph.

    This ansatz is built to be a minimal yet expressive replacement for an RBM-like architecture,
    with the following design goals:

    - Graph-awareness:
      The model knows which edges are incident on which vertices. Instead of connecting all degrees
      of freedom to all hidden units (as in a dense RBM), we:
        * embed edges locally,
        * aggregate those embeddings per vertex,
        * then combine vertex features into a single global feature vector.

      This aligns more naturally with local operators such as vertex Hamilton constraints and volume
      operators, which only couple a small set of adjacent edges.

    - Local expressivity:
      Each edge's gauge charges (e.g. 3 integers for U(1)^3) are passed through a small MLP -> "edge
      embedding". For each vertex, the embeddings of the edges that touch it are summed -> "vertex
      embedding", which is then fed through another small MLP. This gives the network enough
      capacity to model non-linear, gauge-local correlations without being overkill.

    - Global summary:
      Vertex embeddings are summed over all vertices to obtain a global feature vector. This is then
      processed by a final MLP and read out into:
        * a scalar log-amplitude log|ψ(σ)| (real),
        * and optionally a scalar phase arg ψ(σ) (real),
      which we pack into a complex log ψ(σ).

      This "sum over vertices" plays a similar role to summing hidden units in an RBM, but in a
      structured, graph-informed way.

    - Efficiency/small footprint:
      The network is intentionally low-width (tens of hidden units). The cost is dominated by the
      local Hamiltonian/constraint evaluation in VMC, not by the neural net. This keeps per
      iteration runtime reasonable even when the operator (e.g. Thiemann H_v) is expensive.

    - Avoiding collapse to trivial all σ = 0:
      The architecture itself does not hard-code any preference for σ = 0. However, the physics of
      the constraint can favour small flux sectors. To give some control, we provide an optional
      `zero_repulsion` term which adds a soft amplitude penalty that grows with ‖σ‖². This can be
      used to gently discourage trivial/near-trivial configurations while preserving low-energy
      structure.

    Graph/connectivity information
    ------------------------------

    We do not store a full graph object inside this module. Instead, we only require a minimal
    mapping that encodes which edges are incident on which vertices.

    The `vertex_edges` argument is a Python list of length V:

        vertex_edges[v] : List[int]
            A list of edge indices (0 ≤ e < n_edges) that touch vertex v.

    From this, `_vertex_const()` builds static JAX arrays:

        - `vertex_edges` : (V, max_deg), padded with -1 where a vertex has fewer than max_deg
          incident edges.
        - `vertex_mask`  : (V, max_deg), with 1.0 at valid edge slots and 0.0 at padding.

    The generation of this list is done externally (see `make_local_graph_nqs_kwargs()`), so that
    this module stays agnostic of the particular graph library used.

    Operationally, the network behaves as follows:
    Given σ (B, n_edges * gauge_dim):

    1. Edge tower (local per-edge MLP):
       - Reshape to (B, n_edges, gauge_dim).
       - Flatten to (B * n_edges, gauge_dim).
       - Apply `n_edge_layers` × Dense + SiLU.
       - Reshape back to (B, n_edges, edge_hidden).

    2. Vertex aggregation:
       - For each vertex v, gather all incident edge embeddings from step 1 using `vertex_edges`.
       - Multiply by `vertex_mask` to zero-out padding.
       - Sum over the incident edges -> (B, V, edge_hidden).

    3. Vertex tower (shared vertex MLP):
       - Flatten to (B * V, edge_hidden).
       - Apply `n_vertex_layers` × Dense + SiLU, producing (B * V, vertex_hidden).
       - Reshape back to (B, V, vertex_hidden).

    4. Global aggregation + head:
       - Sum over vertices -> (B, vertex_hidden).
       - Apply `n_global_layers` × Dense + SiLU -> (B, global_hidden).
       - Apply a linear amplitude head -> (B,).
       - Optionally add a soft "zero repulsion" term depending on ‖σ‖².
       - Optionally apply a linear phase head -> (B,).
       - Combine into complex log ψ(σ) if `use_phase=True`.

    Notes
    -----

    - This ansatz is deliberately "shallow": it has no deep message-passing or attention. It trades
      some expressive power for simplicity and runtime speed.

    - All graph-specific structure is injected via `vertex_edges`, so you can reuse this module
      across different graphs and models by just changing that list and the Hilbert size.
    """

    #
    #
    #   static metadata: graph/Hilbert

    n_edges: int
    """Number of edges in the underlying graph (without gauge copies)."""

    gauge_dim: int
    """Number of gauge copies per edge (e.g. 3 for U(1)^3)."""

    n_vertices: int
    """Number of vertices (stored for information, shapes are derived from vertex_edges)."""

    vertex_edges: List[List[int]] = field(repr=False, metadata={"pytree_node": False})
    """For each vertex v, a list of edge indices incident on v. This is a pure Python structure kept
     outside the Flax PyTree."""

    #
    #
    #   hyperparameters

    edge_hidden: int = 16
    """Hidden width for edge-level embeddings."""

    vertex_hidden: int = 32
    """Hidden width for vertex-level embeddings."""

    global_hidden: int = 32
    """Hidden width for global-level embeddings."""

    n_edge_layers: int = 2
    """Number of Dense+SiLU layers in the edge MLP."""

    n_vertex_layers: int = 2
    """Number of Dense+SiLU layers in the vertex MLP."""

    n_global_layers: int = 2
    """Number of Dense+SiLU layers in the global MLP."""

    use_phase: bool = True
    """Whether to include a phase head and output complex log ψ."""

    #
    #
    #   dtypes

    param_dtype: Any = jnp.float64
    """Parameter and intermediate activation dtype."""

    output_dtype: Any = jnp.complex128
    """Output dtype (real for amplitude-only, complex for full log ψ)."""

    #
    #
    #   regulariser

    zero_repulsion: float = 0.0
    """When > 0, log|ψ| gets an additive term ∝ log(1 + ‖σ‖²)."""

    def __hash__(self):
        return id(self)

    #
    #
    #   SETUP: construct submodules (MLPs and heads)

    def setup(self):
        """
        Flax setup method.

        Here we construct the three MLP "towers":
          - edge_layers: local per-edge embedding,
          - vertex_layers: local per-vertex embedding,
          - global_layers: global aggregation embedding,

        and the linear heads:
          - amp_head: maps global features to log|ψ|,
          - phase_head: maps global features to arg ψ (if enabled)
        """

        #
        #
        #   EDGE TOWER: local per-edge MLP
        #
        #   Goal:
        #     Given the gauge_dim charges on each edge, we map them into a learned edge embedding of
        #     dimension edge_hidden, sharing weights across all edges

        self.edge_layers = [
            # Each Dense projects the current hidden representation to the same dimension
            # edge_hidden, using param_dtype for numerical stability and consistency with the rest
            # of the model
            nn.Dense(self.edge_hidden, dtype=self.param_dtype)
            for _ in range(self.n_edge_layers)
        ]

        #
        #
        #   VERTEX TOWER: shared vertex MLP
        #
        #   Goal:
        #     After aggregating incident edges at each vertex, we process the resulting per-vertex
        #     vector with a shared MLP. This lets the network learn non-linear vertex-local features
        #     (e.g. how combinations of incident edge fluxes contribute to local constraint
        #     satisfaction)

        self.vertex_layers = [
            nn.Dense(self.vertex_hidden, dtype=self.param_dtype)
            for _ in range(self.n_vertex_layers)
        ]

        #
        #
        #   GLOBAL TOWER: global MLP over summed vertex features
        #
        #   Goal:
        #     Summarise the entire graph configuration into a global vector and then use a small MLP
        #     to capture non-linear global correlations before the final readout

        self.global_layers = [
            nn.Dense(self.global_hidden, dtype=self.param_dtype)
            for _ in range(self.n_global_layers)
        ]

        #
        #
        #   HEADS: amplitude and (optional) phase
        #
        #   Goal:
        #     Map the global embedding into:
        #       - a scalar log-amplitude,
        #       - optionally a scalar phase.
        #     We use `use_bias=False` to keep the heads simple and let the preceding MLPs handle
        #     shifts

        self.amp_head = nn.Dense(
            1,  # scalar output
            use_bias=False,  # no extra bias term
            dtype=self.param_dtype,
        )

        # Only construct a phase head if phase learning is enabled
        if self.use_phase:
            self.phase_head = nn.Dense(
                1,  # scalar output
                use_bias=False,  # no extra bias term
                dtype=self.param_dtype,
            )

    #
    #
    #   VERTEX CONSTANTS: cache vertex-edge indices and masks in JAX arrays

    def _vertex_const(self):
        """
        Build static (vertex_edges, vertex_mask) arrays from the Python list-of-lists
        `self.vertex_edges`.

        `vertex_edges` is a Python object (list of lists) and cannot be used directly inside JAX
        jitted code. We therefore convert it into JAX arrays once and cache them in a Flax
        "variable" collection (`cache`), so subsequent forward passes can reuse them cheaply.

        This method derives the effective vertex count `n_v` from `len(self.vertex_edges)` rather
        than trusting the `n_vertices` field. This avoids shape mismatches if n_vertices and the
        list-of-lists get out of sync.

        Returns
        -------
        dict with keys:
            - "vertex_edges": jnp.ndarray, shape (V, max_deg)
                Edge indices incident at each vertex, padded with -1 in unused entries.
            - "vertex_mask": jnp.ndarray, shape (V, max_deg)
                1.0 where an edge index is valid, 0.0 where padded.

        These arrays are later used to gather and sum edge embeddings per vertex.
        """

        # Grab the Python list-of-lists of incident edges
        ve_lists: List[List[int]] = self.vertex_edges

        # Compute the effective number of vertices from the list length
        n_v = len(ve_lists)

        # If there are no vertices (degenerate case), return empty arrays with consistent dtypes and
        # shapes
        if n_v == 0:
            idx = jnp.zeros((0, 0), dtype=jnp.int32)
            mask = jnp.zeros((0, 0), dtype=jnp.float32)
            return {"vertex_edges": idx, "vertex_mask": mask}

        # Determine the maximum vertex degree (max number of edges incident on any vertex)
        # We will pad all vertices to this length
        max_deg = max(len(lst) for lst in ve_lists)

        # Initialise the vertex -> edge index array with -1 padding
        # Shape: (V, max_deg)
        idx = jnp.full((n_v, max_deg), -1, dtype=jnp.int32)

        # Initialise the vertex mask with zeros, valid positions will be set to 1.0 below
        mask = jnp.zeros((n_v, max_deg), dtype=jnp.float32)

        # Loop over vertices and fill in the edge indices and masks
        for v, lst in enumerate(ve_lists):

            # Number of incident edges at this vertex
            L = len(lst)
            if L == 0:
                # No edges attached to this vertex, leave the row as padding
                continue

            # Convert the Python list of edge indices to a JAX array
            arr = jnp.array(lst, dtype=jnp.int32)

            # Write edge indices into the first L slots of row v
            idx = idx.at[v, :L].set(arr)

            # Mark those L slots as valid in the mask
            mask = mask.at[v, :L].set(1.0)

        # Return both index and mask arrays for use in the forward pass
        return {"vertex_edges": idx, "vertex_mask": mask}

    #
    #
    #   FORWARD PASS: compute log ψ(σ) from integer-encoded configuration

    @nn.compact
    def __call__(self, sigma: jnp.ndarray) -> jnp.ndarray:

        # Extract batch size and total number of degrees of freedom from the input array shape
        B, Ntot = sigma.shape

        # Sanity check: verify that the Hilbert size matches the expected graph structure
        # n_edges * gauge_dim
        assert (
            Ntot == self.n_edges * self.gauge_dim
        ), "sigma shape mismatch: expected n_edges * gauge_dim entries."

        #
        #
        #   LOAD/BUILD CACHED VERTEX CONSTANTS
        #
        #   Goal:
        #     Convert the Python `vertex_edges` list-of-lists into JAX arrays once, then reuse them
        #     on every forward pass. We store them in a Flax variable collection named "cache"

        # cache vertex-edge index structures
        const = self.variable(
            "cache",  # collection name
            "vertex_const",  # variable name within collection
            self._vertex_const,  # initialiser function
        )

        #
        #
        #   EDGE EMBEDDINGS
        #
        #   Goal:
        #     Map raw integer charges on each edge (for all gauge copies) to a small, dense
        #     embedding vector. This replaces the "visible to hidden" mapping of an RBM with a tiny
        #     local MLP

        # Reshape flat sigma into (B, n_edges, gauge_dim) to separate edges and gauge copies
        sig3 = sigma.reshape(B, self.n_edges, self.gauge_dim).astype(jnp.float32)

        # Flatten the (edge, batch) axes into one to feed into the edge MLP: we now have one row per
        # edge in the batch
        # Shape is (B*E, gauge_dim)
        x_edge = sig3.reshape(B * self.n_edges, self.gauge_dim)

        # Start from the raw edge charges as the initial hidden state
        h = x_edge

        # Apply each Dense layer in the edge MLP, followed by a SiLU non-linearity
        for layer in self.edge_layers:

            # Linear projection to edge_hidden features
            h = layer(h)

            # Non-linear activation (smooth and non-saturating near 0)
            h = jnn.silu(h)

        # Reshape back to a (B, n_edges, edge_hidden) tensor, where each edge now has a learned
        # embedding
        # Shape is (B, E, Fe)
        edge_embed = h.reshape(B, self.n_edges, self.edge_hidden)

        #
        #
        #   VERTEX AGGREGATION
        #
        #   Goal:
        #     For each vertex v, gather the embeddings of all incident edges and combine them into a
        #     single "vertex feature" vector by summing. This is where the graph structure enters
        #     the model

        # Extract the JAX arrays encoding vertex -> edge indices and masks
        # Shapes are (V, max_deg)
        v_edges = const.value["vertex_edges"]
        v_mask = const.value["vertex_mask"]

        # Use jnp.take to gather edge embeddings for each vertex:
        #  - axis=1 indicates we index over the edge dimension E
        #  - v_edges[v, :] provides the indices of edges for vertex v
        #
        # Resulting shape: (B, V, max_deg, Fe)
        gathered = jnp.take(edge_embed, v_edges, axis=1)

        # Multiply by the vertex mask (broadcasted over batch and feature dimension) to zero-out
        # padded entries where v_edges == -1
        gathered = gathered * v_mask[None, :, :, None]

        # Sum over the max_deg dimension to aggregate contributions from all incident edges at each
        # vertex.
        # Shape is (B, V, Fe)
        v_agg = jnp.sum(gathered, axis=2)

        #
        #
        #   VERTEX MLP (shared across vertices)
        #
        #   Goal:
        #     Process each vertex feature vector with the same MLP to learn non-linear vertex-local
        #     features, while sharing parameters across all vertices in the graph

        # Unpack the shape for clarity
        Bv, V, Fe = v_agg.shape

        # Sanity check: the feature dimension after aggregation must match the configured
        # edge_hidden
        assert Fe == self.edge_hidden, "edge_hidden mismatch in vertex step."

        # Flatten (batch, vertex) into a single axis to apply the MLP independently to each vertex
        v_flat = v_agg.reshape(Bv * V, Fe)

        # Feed through the shared vertex MLP, layer by layer
        h_v = v_flat
        for layer in self.vertex_layers:

            # Linear projection to vertex_hidden features
            h_v = layer(h_v)

            # SiLU activation for non-linearity
            h_v = jnn.silu(h_v)

        # Reshape back to (B, V, vertex_hidden): one embedding per vertex
        v_embed = h_v.reshape(Bv, V, self.vertex_hidden)

        #
        #
        #   GLOBAL AGGREGATION + MLP
        #
        #   Goal:
        #     Combine local vertex embeddings into a global feature vector, then apply a final MLP
        #     and readouts to produce log ψ(σ)

        # Sum vertex embeddings over all vertices, yielding a global summary vector per
        # configuration
        g = jnp.sum(v_embed, axis=1)

        # Process the global summary through the global MLP
        h_g = g
        for layer in self.global_layers:

            # Linear projection to global_hidden features
            h_g = layer(h_g)

            # SiLU activation
            h_g = jnn.silu(h_g)

        #
        #
        #   AMPLITUDE HEAD: map global features to log-amplitude

        # Apply the linear amplitude head to produce a scalar per sample
        # Output shape is (B, 1), we squeeze the last dimension
        amp = self.amp_head(h_g)[..., 0]

        #
        #
        #   OPTIONAL ZERO REPULSION: softly discourage σ = 0
        #
        #   Goal:
        #     Add a smooth term that grows with ‖σ‖² to nudge the sampler away from the trivial
        #     configuration, without imposing a hard constraint or discontinuity

        if self.zero_repulsion != 0.0:

            # Convert sigma to float for norm computation
            centered = sigma.astype(jnp.float32)

            # Compute squared ℓ2 norm per configuration: ‖σ‖²
            norm2 = jnp.sum(centered * centered, axis=-1)

            # Use log(1 + ‖σ‖²) as a smooth, slowly-growing penalty
            repulsion = jnp.log1p(norm2)

            # Add the penalty with the configured strength
            amp = amp + self.zero_repulsion * repulsion

        # Cast the final log-amplitude to the requested output dtype
        logabs = amp.astype(self.output_dtype)

        #
        #
        #   PHASE HEAD (OPTIONAL): map global features to phase

        if self.use_phase:

            # Compute scalar phase per sample via a linear head
            phase = self.phase_head(h_g)[..., 0]

            # Cast to the same dtype as logabs
            phase = phase.astype(self.output_dtype)

            # Assemble complex log ψ = log|ψ| + i * phase
            out = logabs + 1j * phase
        else:
            # Amplitude-only model: return real log|ψ|
            out = logabs

        # Return log ψ(σ) with shape (B,)
        return out

    #
    #
    #   REGULARIZATION REPORT: placeholder for future L2/etc.

    def regularization_terms(self, params: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """
        Optional hook returning a dictionary of regularization terms derived from the network
        parameters.

        For now, this is a placeholder that returns an empty dict. It can be extended to, e.g.,
        include L2 penalties on weights or other custom regularisers, keyed by name.
        """

        # Initialise an empty dictionary for regularization contributions
        reg: Dict[str, jnp.ndarray] = {}

        # No regularization by default
        return reg


#
#
#   HELPER: build kwargs for LocalGraphNQS from graph and Hamiltonian


def make_local_graph_nqs_kwargs(graph, H):
    """
    Construct keyword arguments for `LocalGraphNQS` from a graph object and a Hilbert  object `H`.

    This helper extracts just enough structural information from the `graph` and `H` objects to
    initialise a `LocalGraphNQS` instance without the network itself depending on the full graph API

    In particular, it builds the `vertex_edges` list-of-lists, which encodes which edges are
    incident on which vertices

    We assume that the graph handler provides:

      - `graph.handler.graph_edges_data["graph"]["connectivities"]`:
        a mapping (str(vertex) -> dict) describing connectivity info per vertex. We only use its
        keys to enumerate vertex ids here

      - `graph.handler.list_of_node_connectivity[v]`:
        a dict with keys "outgoing" and "incoming", each a list of oriented edges (u, w, key) that
        start or end at node v

      - `graph.edge_to_index(edge)`:
        a function mapping an oriented edge key (u, v, key) to a unique integer index in
        [0, n_edges).

    The Hilbert object `H` is assumed to provide:

      - `H.gauge_dimensions`:
        the gauge_dim (e.g. 3 for U(1)^3).

      - `graph.n_edges`:
        the total number of edges at a single gauge-copy level.

    Returns
    -------
    kwargs : dict
        Dictionary of keyword arguments compatible with `LocalGraphNQS`,
        including:
          - n_edges
          - gauge_dim
          - n_vertices
          - vertex_edges
          - and small default widths/depths and dtypes.
    """

    # Extract the connectivity dictionary from the graph handler
    conn = graph.handler.graph_edges_data["graph"]["connectivities"]

    # Sort vertex ids numerically, conn keys are assumed to be strings
    vertex_ids = sorted(int(k) for k in conn.keys())

    # Prepare the list-of-lists mapping each vertex to its incident edges
    vertex_edges: List[List[int]] = []

    # Loop over all vertices in sorted order
    for v in vertex_ids:

        # Gather all oriented edges that are either outgoing from v or incoming to v. This gives us
        # all edges incident on vertex v
        edges_v = (
            graph.handler.list_of_node_connectivity[v]["outgoing"]
            + graph.handler.list_of_node_connectivity[v]["incoming"]
        )

        # Map each oriented edge key to its integer edge index in [0, n_edges)
        # Use a set to deduplicate, then sort to get a stable ordering
        idxs = sorted({int(graph.edge_to_index(e)) for e in edges_v})

        # Append the resulting list of indices to the vertex_edges list
        vertex_edges.append(idxs)

    # Finally, assemble a kwargs dictionary for LocalGraphNQS. These are intentionally small
    # defaults, widths and depths can be tuned from the caller if needed
    return dict(
        # Graph/Hilbert structure
        n_edges=graph.n_edges,
        gauge_dim=H.gauge_dimensions,
        n_vertices=len(vertex_edges),
        vertex_edges=vertex_edges,
        # Small network defaults, these can be increased if the problem requires more expressivity
        edge_hidden=16,
        vertex_hidden=32,
        global_hidden=32,
        n_edge_layers=2,
        n_vertex_layers=2,
        n_global_layers=2,
        # By default, learn a phase as well as an amplitude
        use_phase=True,
        # Use double precision for parameters, complex128 for outputs
        param_dtype=jnp.float64,
        output_dtype=jnp.complex128,
        # Start with zero_repulsion = 0.0 (no explicit bias against σ = 0)
        # This can be set > 0 to discourage trivial configurations if needed
        zero_repulsion=0.0,
    )
