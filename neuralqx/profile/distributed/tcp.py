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


"""Boundary clock exchange and streamed rank artifacts over a private TCP service."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from neuralqx.utils.errors import NeuralqxValueError

from ..errors import ProfileError
from ..trace.io import atomic_file
from ..trace.io import write_json
from .clock import ClockSample
from .common import LocalCoordinator
from .common import rank_directory
from .common import validate_peer
from .common import wait_until


class _Collector:
    """Rank-zero upload state shared by HTTP handler threads."""

    def __init__(self, coordinator):
        self.coordinator = coordinator
        """Rank-zero coordinator that owns this HTTP collection state."""

        self.lock = threading.Lock()
        """Lock protecting registrations and upload commit state."""

        self.peers = {0: coordinator.hello}
        """Registered rank identities indexed by process rank."""

        self.manifests: dict[int, Path] = {}
        """Completed manifest paths indexed by rank."""

        self.uploading: set[int] = set()
        """Ranks with an active trace upload."""

        self.uploaded: set[int] = set()
        """Ranks whose complete trace upload has committed."""

    def directory(self, rank: int) -> Path:
        """Return the fixed collection directory for a registered worker rank."""
        return rank_directory(self.coordinator.root / "collected", rank)


class _Server(ThreadingHTTPServer):
    """HTTP server whose shutdown waits for active artifact handlers."""

    collector: _Collector
    """Shared registration and artifact state for request handlers."""

    daemon_threads = False
    """False keeps active artifact handlers alive until server_close joins them."""


class _Handler(BaseHTTPRequestHandler):
    """Bounded control requests and streaming uploads for one profiling run."""

    server: _Server
    """Owning server, including its run-specific collection state."""

    protocol_version = "HTTP/1.1"
    """HTTP protocol used for length-delimited request and response bodies."""

    def log_message(self, *args) -> None:
        """Suppress HTTP access logging so capture does not write to application stderr."""
        pass

    def setup(self) -> None:
        """Initialize the connection and apply the configured coordination timeout."""
        super().setup()
        self.connection.settimeout(self.server.collector.coordinator.options.timeout_s)

    def _reply(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 1024 * 1024:
            raise ProfileError("Invalid profiler control message size.")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ProfileError("Expected a JSON control object.")
        return value

    def do_GET(self) -> None:
        """Dispatch a registration-status or clock query."""
        self._dispatch()

    def do_POST(self) -> None:
        """Dispatch a registration, streamed trace, or committed manifest upload."""
        self._dispatch()

    def _dispatch(self) -> None:
        state = self.server.collector
        coordinator = state.coordinator
        try:
            if self.headers.get("X-Neuralqx-Run") != coordinator.hello["run_id"]:
                self._reply({"error": "Profiling run identifier mismatch."}, 403)
                return
            if self.path == "/clock" and self.command == "GET":
                t1 = time.perf_counter_ns()
                self._reply({"t1": t1, "t2": time.perf_counter_ns()})
            elif self.path == "/peers" and self.command == "GET":
                with state.lock:
                    peers = list(state.peers.values())
                self._reply(
                    {"peers": peers if len(peers) == coordinator.rank.count else []}
                )
            elif self.path == "/register" and self.command == "POST":
                peer = self._json()
                validate_peer(peer, coordinator.hello["run_id"], coordinator.rank.count)
                with state.lock:
                    if peer["index"] in state.peers:
                        raise ProfileError("A profiling rank registered twice.")
                    state.peers[peer["index"]] = peer
                self._reply({"registered": True})
            elif self.path.startswith("/trace/") and self.command == "POST":
                rank = self._rank(state)
                self._receive_trace(state, rank)
                self._reply({"received": True})
            elif self.path.startswith("/manifest/") and self.command == "POST":
                rank = self._rank(state)
                manifest = self._json()
                validate_peer(
                    manifest, coordinator.hello["run_id"], coordinator.rank.count
                )
                if manifest["index"] != rank:
                    raise ProfileError("Manifest rank does not match upload rank.")
                with state.lock:
                    if rank not in state.uploaded or rank in state.manifests:
                        raise ProfileError(
                            "Trace upload must finish before its manifest is committed."
                        )
                    path = state.directory(rank) / "manifest.json"
                    write_json(path, manifest)
                    state.manifests[rank] = path
                self._reply({"committed": True})
            else:
                self._reply({"error": "Unknown profiler endpoint."}, 404)
        except (ValueError, OSError, ProfileError) as error:
            try:
                self._reply({"error": str(error)}, 400)
            except OSError:
                pass

    def _rank(self, state) -> int:
        rank = int(self.path.rsplit("/", 1)[1])
        with state.lock:
            if rank == 0 or rank not in state.peers:
                raise ProfileError(
                    "Artifact upload requires a registered nonzero rank."
                )
        return rank

    def _receive_trace(self, state, rank: int) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        options = state.coordinator.options
        if not 0 < length <= options.max_trace_bytes:
            raise ProfileError(
                "Rank trace exceeds profile_max_trace_bytes or has invalid length."
            )
        with state.lock:
            if rank in state.uploading or rank in state.uploaded:
                raise ProfileError("Duplicate rank trace upload.")
            state.uploading.add(rank)
        try:
            deadline = time.monotonic() + options.timeout_s
            with atomic_file(
                state.directory(rank) / "trace.json.gz", binary=True
            ) as output:
                remaining = length
                while remaining:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        raise ProfileError("Rank trace upload exceeded its deadline.")
                    self.connection.settimeout(timeout)
                    data = self.rfile.read(min(1024 * 1024, remaining))
                    if not data:
                        raise ProfileError(
                            "Rank trace upload ended before Content-Length bytes arrived."
                        )
                    output.write(data)
                    remaining -= len(data)
            with state.lock:
                state.uploaded.add(rank)
        finally:
            with state.lock:
                state.uploading.discard(rank)


class TcpCoordinator(LocalCoordinator):
    """Collect node-local rank traces through a rank-zero HTTP service.

    Use on a trusted job network. The run identifier prevents accidental
    cross-run traffic but is not authentication. Uploads are bounded and
    streamed. No numerical collectives or per-annotation messages are used.
    """

    def __init__(self, *args):
        """Resolve a coordinator address using the LocalCoordinator constructor arguments."""
        super().__init__(*args)
        address = self.options.coordinator_address
        try:
            parsed = urlsplit(f"//{address}")
            host, port = parsed.hostname, parsed.port
        except ValueError as error:
            raise NeuralqxValueError(
                "coordinator_address must be host:port."
            ) from error
        if not host or port is None or not 0 < port < 65536 or parsed.path:
            raise NeuralqxValueError(
                "TCP collection requires a reachable coordinator_address in host:port form."
            )
        self.host = host
        """Reachable hostname or IP address of rank zero."""

        self.port = port
        """TCP port used by the run-specific collection service."""

        self.server = None
        """Rank-zero HTTP server, or None on workers and after shutdown."""

        self.thread = None
        """Rank-zero listener thread, absent on workers."""

        self.collector = None
        """Shared upload state on rank zero, absent on workers."""

    def _request(self, path: str, payload: dict | None = None) -> dict:
        connection = http.client.HTTPConnection(
            self.host, self.port, timeout=self.options.timeout_s
        )
        body = (
            None if payload is None else json.dumps(payload, allow_nan=False).encode()
        )
        try:
            connection.request(
                "GET" if payload is None else "POST",
                path,
                body,
                headers={
                    "X-Neuralqx-Run": self.hello["run_id"],
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            result = json.loads(response.read(1024 * 1024 + 1))
            if response.status != 200:
                raise ProfileError(
                    result.get(
                        "error", f"Profiler collector returned HTTP {response.status}."
                    )
                )
            return result
        finally:
            connection.close()

    def prepare(self) -> None:
        """Start or join the rank-zero service and measure the entry clock offset."""
        if self.rank.index == 0:
            server_type = type(
                "ProfilerHTTPServer",
                (_Server,),
                {
                    "address_family": (
                        socket.AF_INET6 if ":" in self.host else socket.AF_INET
                    )
                },
            )
            self.collector = _Collector(self)
            self.server = server_type((self.host, self.port), _Handler)
            self.server.collector = self.collector
            self.thread = threading.Thread(
                target=self.server.serve_forever,
                kwargs={"poll_interval": 0.05},
                name="neuralqx-profile-collector",
                daemon=True,
            )
            self.thread.start()
        else:

            def register():
                try:
                    return self._request("/register", self.hello)
                except (ConnectionError, OSError):
                    return None

            wait_until(
                register, self.options.timeout_s, "the TCP profiling coordinator"
            )
        self.peers = wait_until(
            lambda: self._request("/peers")["peers"],
            self.options.timeout_s,
            "all TCP profiling rank registrations",
        )
        self.start_clock = self._calibrate()

    def _calibrate(self) -> ClockSample:
        if self.rank.index == 0:
            return ClockSample(time.perf_counter_ns(), 0, 0)
        samples = []
        for _ in range(self.options.clock_samples):
            t0 = time.perf_counter_ns()
            payload = self._request("/clock")
            t3 = time.perf_counter_ns()
            samples.append(ClockSample.round_trip(t0, payload["t1"], payload["t2"], t3))
        return min(samples, key=lambda sample: sample.uncertainty_ns)

    def _upload(self, path: Path) -> None:
        size = path.stat().st_size
        if size > self.options.max_trace_bytes:
            raise ProfileError("Rank trace exceeds profile_max_trace_bytes.")
        deadline = time.monotonic() + self.options.timeout_s
        connection = http.client.HTTPConnection(
            self.host, self.port, timeout=self.options.timeout_s
        )
        try:
            connection.putrequest("POST", f"/trace/{self.rank.index}")
            connection.putheader("X-Neuralqx-Run", self.hello["run_id"])
            connection.putheader("Content-Length", str(size))
            connection.endheaders()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ProfileError("Rank trace upload exceeded its deadline.")
                    connection.sock.settimeout(remaining)
                    connection.send(chunk)
            response = connection.getresponse()
            payload = json.loads(response.read(1024 * 1024))
            if response.status != 200:
                raise ProfileError(payload.get("error", "Rank trace upload failed."))
        finally:
            connection.close()

    def collect(self, manifest: dict):
        """Calibrate the exit clock and transfer completed rank artifacts.

        Args:
            manifest: Completed local trace metadata, updated with clock samples.

        Returns:
            Rank-ordered committed manifests on rank zero. None on workers, which
            retain their own trace and manifest after uploading.

        Raises:
            ProfileError: An artifact is rejected or exceeds the transfer limit.
            ProfileTimeoutError: Required rank uploads do not complete in time.
        """
        end = self._calibrate()
        manifest["clock"] = {
            "method": "tcp_round_trip",
            "start": self.start_clock.to_dict(),
            "end": end.to_dict(),
        }
        local = rank_directory(self.root, self.rank.index)
        path = local / "manifest.json"
        write_json(path, manifest)
        if self.rank.index:
            self._upload(local / "trace.json.gz")
            self._request(f"/manifest/{self.rank.index}", manifest)
            return None
        collector = self.collector
        assert collector is not None
        with collector.lock:
            collector.manifests[0] = path

        def completed():
            with collector.lock:
                paths = dict(collector.manifests)
            return (
                [paths[index] for index in range(self.rank.count)]
                if len(paths) == self.rank.count
                else None
            )

        return wait_until(
            completed, self.options.timeout_s, "completed TCP rank trace uploads"
        )

    def close(self) -> None:
        """Stop the rank-zero listener and wait for active request handlers to finish."""
        if self.server is not None:
            assert self.thread is not None
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=self.options.timeout_s)
            self.server = None
