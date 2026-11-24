"""Run the distributed guides explicitly on logical CPU devices.

This maintainer check is independent of MkDocs and never runs during a build.
Each page and simulation policy runs in a fresh process with its own topology.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs/distributed/examples"
PAGES = {
    "docs/distributed/jax-parallelism.md": (4, "samples", 1),
    "docs/distributed/sharding.md": (8, "all", 2),
    "docs/distributed/cpu.md": (4, "samples,tensor", 2),
    "docs/distributed/multi-gpu.md": (8, "all", 2),
    "docs/distributed/multi-node.md": (8, "all", 2),
    "docs/distributed/configuration.md": (4, "samples,tensor", 2),
    "docs/distributed/troubleshooting.md": (4, "samples,tensor", 2),
    "docs/distributed/tensor-parallelism/models.md": (4, "samples,tensor", 2),
    "docs/distributed/tensor-parallelism/custom-models.md": (4, "samples,tensor", 2),
    "docs/developers/networks/tensor-rules.md": (2, "tensor", 2),
}
POLICIES = (
    (1, "samples", 1),
    (4, "samples", 1),
    (4, "samples,operators", 1),
    (4, "tensor", 4),
    (4, "samples,tensor", 2),
    (4, "operators,tensor", 2),
    (8, "all", 2),
)


def environment(count, policy, tensor_size):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NEURALQX_", "JAX_", "XLA_"))
    }
    env.update(
        JAX_PLATFORMS="cpu",
        JAX_NUM_CPU_DEVICES=str(count),
        NEURALQX_SHARDING=policy,
        NEURALQX_MESH_TENSOR_AXIS_SIZE=str(tensor_size),
        PYTHONDONTWRITEBYTECODE="1",
        MPLBACKEND="Agg",
        PYTHONPATH=os.pathsep.join((str(ROOT), str(EXAMPLES))),
    )
    return env


def page_worker(page):
    sys.path[:0] = [str(ROOT), str(EXAMPLES)]
    blocks = re.findall(r"^```python\n(.*?)^```", page.read_text(), re.M | re.S)
    namespace = {"__name__": "documentation_example", "__file__": str(page)}
    for number, block in enumerate(blocks, 1):
        print(f"BLOCK {number}/{len(blocks)}", flush=True)
        exec(compile(block, f"{page}:block-{number}", "exec"), namespace)
    print(f"PASS: {len(blocks)} Python blocks", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        page_worker(args.worker.resolve())
        return 0
    log_dir = args.log_dir or Path(tempfile.mkdtemp(prefix="neuralqx-distributed-docs-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            path.replace("/", "_"),
            [sys.executable, "-B", __file__, "--worker", str(ROOT / path)],
            layout,
        )
        for path, layout in PAGES.items()
    ]
    jobs.extend(
        (
            f"simulation_{count}_{policy.replace(',', '_')}",
            [sys.executable, "-B", str(EXAMPLES / "simulation.py"), "--steps", "2"],
            (count, policy, tensor_size),
        )
        for count, policy, tensor_size in POLICIES
    )
    failures = 0
    for name, command, layout in jobs:
        log = log_dir / f"{name}.log"
        with log.open("w") as output:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment(*layout),
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
        if result.returncode:
            failures += 1
            print(f"FAIL {name}", flush=True)
            print("\n".join(log.read_text().splitlines()[-20:]), flush=True)
        else:
            print(f"PASS {name}", flush=True)
    print(f"{len(jobs) - failures}/{len(jobs)} checks passed. Logs: {log_dir}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
