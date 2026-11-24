"""One VMC calculation used throughout the distributed simulation guides."""

# Import neuraLQX before Optax so configured distributed startup runs first.
# ruff: noqa: I001

import argparse

import jax
import neuralqx as nqx
import optax
from neuralqx.operator import SumOperator
from neuralqx.operator.prebuilt.wcl import NumberOperator


def build_calculation():
    """Construct the same physical problem and global batch on every process."""
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    edges = [(vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
    graph = nqx.graph.Graph(edges, is_planar=True)
    hilbert = nqx.hilbert.U1Hilbert(graph, cutoff=2)
    charges = tuple(NumberOperator(hilbert, edge=i) for i in range(4))
    objective = SumOperator(tuple(charge @ charge for charge in charges))

    sampler = nqx.sampler.MetropolisLocal(hilbert, n_chains=64)
    model = nqx.nn.init_tensor_parallel(
        lambda: nqx.nn.MLP(
            hilbert.size, hidden_features=(32, 32), complex_output=False, rngs=17
        )
    )
    state = nqx.vqs.VariationalState(
        sampler,
        model,
        n_samples=1024,
        n_discard_per_chain=16,
        sampler_seed=23,
        chunk_size=128,
    )
    driver = nqx.driver.VMC(objective, optax.sgd(learning_rate=0.01), variational_state=state)
    return state, objective, driver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    print(
        f"Process {jax.process_index()}/{jax.process_count()}: "
        f"{jax.local_device_count()} local, {jax.device_count()} global devices",
        flush=True,
    )
    state, objective, driver = build_calculation()
    if nqx.config.network_sharding_enabled:
        from neuralqx.jax.tensor_parallel import tensor_mesh

        if jax.process_index() == 0:
            print("Tensor mesh:", dict(tensor_mesh().shape))
    for path, value in jax.tree_util.tree_flatten_with_path(state.parameters)[0]:
        if jax.process_index() == 0:
            print(f"Weight {jax.tree_util.keystr(path)}: {value.shape}, {value.sharding}")
    driver.run(args.steps)

    # Every process participates in the expectation and any communication.
    result = state.expect(objective)
    mean = jax.device_get(result.mean)
    if jax.process_index() == 0:
        print(f"Mean squared-charge objective: {mean}")
        print(f"Global samples: {state.samples.shape}")
        print(f"Sample sharding: {state.samples.sharding}")

    # Only local shards are inspected here, so every process can print its own.
    for shard in state.samples.addressable_shards:
        print(
            f"Process {jax.process_index()}, device {shard.device.id}: "
            f"index={shard.index}, local shape={shard.data.shape}",
            flush=True,
        )


if __name__ == "__main__":
    main()
