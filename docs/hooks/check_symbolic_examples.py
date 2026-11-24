"""Execute the symbolic guides' Python blocks, in order and one process per page.

This is an explicit maintainer check. MkDocs never imports or runs this file.
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
GUIDES = (
    ROOT / "docs/concepts/operators/symbolic",
    ROOT / "docs/developers/symbolic",
)


def run_page(page: Path) -> None:
    sys.path.insert(0, str(ROOT))
    source = page.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", source, re.MULTILINE | re.DOTALL)
    namespace = {"__name__": "__main__", "__file__": str(page)}
    for index, block in enumerate(blocks, 1):
        print(f"BLOCK {index}/{len(blocks)}", flush=True)
        exec(compile(block, f"{page}:block-{index}", "exec"), namespace)
    print(f"PASS: {len(blocks)} Python blocks", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages", nargs="*", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        run_page(args.worker.resolve())
        return 0

    pages = args.pages or sorted(page for guide in GUIDES for page in guide.rglob("*.md"))
    log_dir = args.log_dir or Path(tempfile.mkdtemp(prefix="neuralqx-symbolic-docs-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    failures = []
    for page in pages:
        page = page.resolve()
        relative = page.relative_to(ROOT)
        log = log_dir / (str(relative).replace(os.sep, "_") + ".log")
        with log.open("w") as output:
            result = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(page)],
                cwd=ROOT,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            failures.append(relative)
            print(f"FAIL {relative}\n{log}", flush=True)
            print("\n".join(log.read_text().splitlines()[-18:]), flush=True)
        else:
            print(f"PASS {relative}", flush=True)
    print(f"{len(pages) - len(failures)}/{len(pages)} pages passed. Logs: {log_dir}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
