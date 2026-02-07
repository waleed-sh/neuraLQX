#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import os
import sys
import subprocess
from pathlib import Path

import pytest


def _pytest_cov_is_active(env: dict) -> bool:
    """
    True when tests are being run under pytest-cov (typical CI: pytest --cov ...).
    We detect this via env vars pytest-cov sets.
    """
    # pytest-cov sets these when active
    if "COV_CORE_SOURCE" in env or "COV_CORE_CONFIG" in env:
        return True
    # some setups only export this
    if "PYTEST_ADDOPTS" in env and "--cov" in env["PYTEST_ADDOPTS"]:
        return True
    return False


@pytest.fixture
def run_vmc_case():
    def _run(case: str, output_path: str | os.PathLike, *, timeout: int = 600) -> None:
        this_dir = Path(__file__).resolve().parent
        test_file = this_dir / "test_models_run_vmc.py"
        repo_root = this_dir.parents[1]

        env = os.environ.copy()

        # hard isolation knobs to avoid hangs
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("JAX_PLATFORMS", "cpu")
        env.setdefault("JAX_PLATFORM_NAME", "cpu")
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        env.setdefault("PYTHONUNBUFFERED", "1")

        # ensure subprocess imports local repo
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

        # IMPORTANT: do NOT use COVERAGE_PROCESS_START or sitecustomize here.
        # That’s what tends to leak and hang other tests.
        # Instead, explicitly run coverage only for THIS subprocess when pytest-cov is active.
        if _pytest_cov_is_active(env):
            cmd = [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                # respect the coverage config if present, harmless if absent
                "--rcfile",
                str(repo_root / ".coveragerc"),
                str(test_file),
                "--case",
                str(case),
                "--output",
                str(output_path),
            ]
        else:
            cmd = [
                sys.executable,
                str(test_file),
                "--case",
                str(case),
                "--output",
                str(output_path),
            ]

        res = subprocess.run(
            cmd,
            env=env,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if res.returncode != 0:
            raise AssertionError(
                f"Isolated VMC subprocess failed for case={case!r}\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"--- STDOUT ---\n{res.stdout}\n\n"
                f"--- STDERR ---\n{res.stderr}\n"
            )

    return _run
