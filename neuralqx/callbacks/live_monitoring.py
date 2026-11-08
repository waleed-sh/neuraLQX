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

"""Local, asynchronous monitoring for variational driver runs."""

from __future__ import annotations

import csv
import errno
import io
import json
import math
import queue
import secrets
import threading
import time
import warnings
from collections import deque
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import parse_qs
from urllib.parse import urlsplit

import numpy as np

from .base import Callback
from .control import CallbackControl

_NOT_SCALAR = object()


def _number(value, *, unsupported=None):
    if value is None or callable(value):
        return unsupported
    if getattr(value, "shape", ()) != ():
        return unsupported
    try:
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in "biufc":
            return unsupported
        result = float(array.real)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, RuntimeError):
        return unsupported


def _attribute(value, *names):
    for name in names:
        item = (
            value.get(name)
            if isinstance(value, Mapping)
            else getattr(value, name, None)
        )
        if item is not None and not callable(item):
            return item
    return None


def _scalar_stat(value):
    if isinstance(value, tuple) and len(value) == 2:
        mean, error = value
    else:
        mean = _attribute(value, "mean", "Mean")
        error = _attribute(value, "error_of_mean", "Sigma", "sigma", "error")
        if mean is None:
            mean = value
    # Conversion runs only on the consumer thread, never in a training hook.
    mean = _number(mean, unsupported=_NOT_SCALAR)
    return None if mean is _NOT_SCALAR else [mean, _number(error)]


class LiveMonitor:
    """Serve a local dashboard and consume bounded scalar updates in the background.

    The browser plots the objective and selected metrics with their standard
    errors. History contains at most ``max_points`` updates. A full pending
    queue drops its oldest update so a slow display cannot block optimization.
    All assets are served locally and no web-framework dependency is needed.

    ``finish()`` freezes the run while keeping its dashboard available.
    ``shutdown()`` releases the port and stops the background threads. Use the
    monitor as a context manager when it should close at the end of a block.
    """

    def __init__(
        self,
        host="127.0.0.1",
        port=8050,
        *,
        objective_label="Constraint",
        x_label="Iteration",
        y_label="Expectation",
        update_interval_ms=500,
        max_points=10_000,
        queue_size=32,
    ):
        if not 0 <= int(port) <= 65535:
            raise ValueError("port must be between 0 and 65535.")
        if int(max_points) < 2 or int(queue_size) < 1 or int(update_interval_ms) < 50:
            raise ValueError(
                "Require max_points >= 2, queue_size >= 1, and update_interval_ms >= 50."
            )
        self.host = str(host)
        self.objective_label = str(objective_label)
        self.x_label = str(x_label)
        self.y_label = str(y_label)
        self.update_interval_ms = int(update_interval_ms)
        self.max_points = int(max_points)
        self._queue = queue.Queue(maxsize=int(queue_size))
        self._rows = deque(maxlen=self.max_points)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._abort = threading.Event()
        self._started = time.monotonic()
        self._ended = None
        self._status = "running"
        self._finish_status = None
        self._message = ""
        self._sequence = 0
        self._latest_step = None
        self._metrics = {}
        self._dropped = 0
        self._token = secrets.token_urlsafe(24)
        self._assets = {
            path: (
                files("neuralqx.callbacks")
                .joinpath("_live_monitor", filename)
                .read_bytes(),
                mime,
            )
            for path, filename, mime in (
                ("/", "index.html", "text/html; charset=utf-8"),
                ("/monitor.css", "monitor.css", "text/css; charset=utf-8"),
                ("/monitor.js", "monitor.js", "text/javascript; charset=utf-8"),
            )
        }
        handler = self._handler()
        for candidate in range(int(port), min(int(port) + 51, 65536)):
            try:
                self._server = ThreadingHTTPServer((self.host, candidate), handler)
                break
            except OSError as error:
                if error.errno != errno.EADDRINUSE or candidate == min(
                    int(port) + 50, 65535
                ):
                    raise
        self.port = self._server.server_address[1]
        self._consumer = threading.Thread(
            target=self._consume, name="nqx-monitor-data", daemon=True
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="nqx-monitor-http",
            daemon=True,
        )
        self._consumer.start()
        self._thread.start()

    @property
    def url(self):
        """Local dashboard address, using the port actually bound by the server."""
        host = "127.0.0.1" if self.host in {"", "0.0.0.0"} else self.host
        return f"http://{host}:{self.port}"

    @property
    def is_running(self):
        """Whether the HTTP server is available, including after a run finishes."""
        return not self._closed.is_set() and self._thread.is_alive()

    @property
    def abort_requested(self):
        """Whether the browser requested a graceful stop at a step boundary."""
        return self._abort.is_set()

    def log(
        self, step, objective=None, observables=None, *, log_data=None, diagnostics=None
    ):
        """Enqueue an objective and optional scalar metrics without synchronizing JAX.

        Values may be statistics objects, scalars, or ``(mean, error)`` pairs.
        ``observables`` retains the upstream keyword for additional metrics.
        Legacy ``log(step, log_data)`` and ``log_data=...`` calls accept the
        objective under its display label or ``"Loss"``.
        """
        if self._closed.is_set() or self._finish_status is not None:
            return
        if isinstance(objective, Mapping) and observables is None and log_data is None:
            log_data, objective = objective, None
        if log_data is not None:
            objective = log_data.get(self.objective_label, log_data.get("Loss"))
            observables = {
                key: value
                for key, value in log_data.items()
                if key not in {self.objective_label, "Loss", "wallclock"}
            }
        payload = (
            int(step),
            time.monotonic() - self._started,
            objective,
            dict(observables or {}),
            dict(diagnostics or {}),
        )
        self._latest_step = int(step)
        self._enqueue(payload)

    def _enqueue(self, payload):
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                self._dropped += 1

    def _consume(self):
        while not self._closed.is_set():
            if self._finish_status is not None and self._queue.empty():
                with self._lock:
                    self._status = self._finish_status
                    self._ended = time.monotonic()
                return
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                step, elapsed, objective, extras, diagnostics = item
                metrics = {}
                stat = _scalar_stat(objective)
                if stat is not None:
                    metrics["__objective__"] = stat
                for name, value in extras.items():
                    stat = _scalar_stat(value)
                    if stat is not None and str(name) != "__objective__":
                        metrics[str(name)] = stat
                diagnostic_values = {
                    str(key): _number(value) for key, value in diagnostics.items()
                }
                with self._lock:
                    self._sequence += 1
                    self._metrics.update(dict.fromkeys(metrics))
                    self._rows.append(
                        {
                            "seq": self._sequence,
                            "step": step,
                            "elapsed": elapsed,
                            "metrics": metrics,
                            "diagnostics": diagnostic_values,
                        }
                    )
            except Exception as error:
                # A malformed optional metric must not interrupt training or
                # strand the consumer queue. Surface the failure in the UI.
                with self._lock:
                    self._message = (
                        f"Skipped a monitoring update: {type(error).__name__}"
                    )
            finally:
                self._queue.task_done()

    def finish(self, status="completed"):
        """Flush pending updates in order and leave the final dashboard available."""
        if status not in {"completed", "stopped", "failed"}:
            raise ValueError("Unknown monitor completion status.")
        if self._status == "running" and not self._closed.is_set():
            self._finish_status = status

    def snapshot(self, after=0):
        """Return JSON-ready updates newer than a cursor and current run metadata."""
        with self._lock:
            first = self._rows[0]["seq"] if self._rows else self._sequence + 1
            reset = after < first - 1 or after > self._sequence
            rows = []
            for row in reversed(self._rows):
                if not reset and row["seq"] <= after:
                    break
                rows.append(row)
            rows.reverse()
            return {
                "rows": rows,
                "cursor": self._sequence,
                "reset": reset,
                "status": (
                    "stopping"
                    if self.abort_requested and self._status == "running"
                    else self._status
                ),
                "step": self._latest_step,
                "elapsed": (self._ended or time.monotonic()) - self._started,
                "metrics": list(self._metrics),
                "objective_label": self.objective_label,
                "x_label": self.x_label,
                "y_label": self.y_label,
                "max_points": self.max_points,
                "interval": self.update_interval_ms,
                "dropped": self._dropped,
                "message": self._message,
                "token": self._token,
            }

    def _csv(self):
        snapshot = self.snapshot(-1)
        names = snapshot["metrics"]
        output = io.StringIO()
        writer = csv.writer(output)
        labels = [
            self.objective_label if name == "__objective__" else name for name in names
        ]
        writer.writerow(
            [
                "step",
                "elapsed",
                *[column for name in labels for column in (name, name + " error")],
            ]
        )
        for row in snapshot["rows"]:
            writer.writerow(
                [
                    row["step"],
                    row["elapsed"],
                    *[
                        value
                        for name in names
                        for value in row["metrics"].get(name, (None, None))
                    ],
                ]
            )
        return output.getvalue().encode()

    def _handler(self):
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, data, mime="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'",
                )
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                request = urlsplit(self.path)
                if request.path in monitor._assets:
                    data, mime = monitor._assets[request.path]
                    self._send(200, data, mime)
                elif request.path == "/api/updates":
                    try:
                        after = int(parse_qs(request.query).get("after", [0])[0])
                    except ValueError:
                        self._send(400, b"{}")
                        return
                    data = json.dumps(
                        monitor.snapshot(after), allow_nan=False, separators=(",", ":")
                    ).encode()
                    self._send(200, data)
                elif request.path == "/history.csv":
                    self._send(200, monitor._csv(), "text/csv; charset=utf-8")
                else:
                    self._send(404, b"{}")

            def do_POST(self):
                origin = self.headers.get("Origin")
                if (
                    self.path != "/api/stop"
                    or not secrets.compare_digest(
                        self.headers.get("X-Monitor-Token", ""), monitor._token
                    )
                    or (origin and urlsplit(origin).netloc != self.headers.get("Host"))
                    or self.headers.get("Sec-Fetch-Site") == "cross-site"
                ):
                    self._send(403, b"{}")
                    return
                if monitor._status == "running":
                    monitor._abort.set()
                self._send(200, b'{"ok":true}')

        return Handler

    def shutdown(self):
        """Close the server and release its port. Repeated calls are harmless."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1)
        self._consumer.join(timeout=1)

    close = shutdown

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


class LiveMonitoringCallback(Callback):
    """Stream driver metrics to a local dashboard without blocking on device values.

    Attach with ``driver.run(..., callbacks=LiveMonitoringCallback())``. The
    server starts on ``run_started`` and prints its URL. ``every`` and
    ``min_interval`` limit collection by step and elapsed time respectively.
    The first and final completed steps are always submitted. No driver
    history is enabled and no additional operator expectations are evaluated.

    The final dashboard remains available until ``shutdown()`` or process
    exit. ``close_on_finish=True`` instead closes it with the run. On multiple
    JAX processes, install the callback on every process so stop requests can
    be broadcast at the same step boundary. Only process zero serves the UI.
    """

    requires_log_data = True
    stores_history = False

    def __init__(
        self,
        host="127.0.0.1",
        port=8050,
        *,
        objective_label=None,
        every=1,
        min_interval=0.25,
        update_interval_ms=500,
        max_points=10_000,
        queue_size=32,
        close_on_finish=False,
    ):
        if int(every) < 1 or not math.isfinite(float(min_interval)) or min_interval < 0:
            raise ValueError(
                "every must be positive and min_interval finite and nonnegative."
            )
        self.every = int(every)
        self.min_interval = float(min_interval)
        self.close_on_finish = bool(close_on_finish)
        self._settings = dict(
            host=host,
            port=port,
            update_interval_ms=update_interval_ms,
            max_points=max_points,
            queue_size=queue_size,
        )
        self._objective_label = objective_label
        self.monitor = None
        self._distributed = False
        self._last_sent = None
        self._last_time = -math.inf
        self._last_payload = None
        self._failed = False

    def run_started(self, context):
        import jax

        self.shutdown()
        self._distributed = jax.process_count() > 1
        self._last_sent = None
        self._last_time = -math.inf
        self._last_payload = None
        self._failed = False
        if jax.process_index() == 0:
            try:
                self.monitor = LiveMonitor(
                    **self._settings,
                    objective_label=self._objective_label or context.loss_name,
                )
            except OSError as error:
                warnings.warn(
                    f"Live monitoring could not start: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            print(f"neuraLQX live monitor: {self.monitor.url}", flush=True)

    def step_started(self, context):
        del context
        stop = self.monitor is not None and self.monitor.abort_requested
        if self._distributed:
            from jax.experimental.multihost_utils import broadcast_one_to_all

            stop = bool(broadcast_one_to_all(np.asarray(stop)))
        if stop:
            return CallbackControl.stop_run("Stop requested from the live monitor.")
        return None

    def step_finished(self, context):
        if self.monitor is None:
            return
        # Keep immutable scalar references. Device-to-host conversion happens
        # in the consumer thread, after the training hook has returned.
        extras = {
            key: value
            for key, value in context.log_data.items()
            if key not in {context.loss_name, "Loss", "wallclock", "acceptance"}
        }
        diagnostics = {
            "acceptance": context.log_data.get("acceptance"),
            "variance": _attribute(context.loss, "variance", "Variance"),
            "r_hat": _attribute(context.loss, "r_hat", "R_hat"),
        }
        self._last_payload = (context.step, context.loss, extras, diagnostics)
        now = time.monotonic()
        if self._last_sent is None or (
            context.step % self.every == 0
            and now - self._last_time >= self.min_interval
        ):
            self._publish_last()
            self._last_time = now

    def _publish_last(self):
        if self.monitor is not None and self._last_payload is not None:
            step, loss, extras, diagnostics = self._last_payload
            self.monitor.log(
                step, objective=loss, observables=extras, diagnostics=diagnostics
            )
            self._last_sent = self._last_payload

    def run_failed(self, context):
        del context
        self._failed = True

    def run_finished(self, context):
        del context
        if self.monitor is None:
            return
        if self._last_payload is not None and self._last_payload is not self._last_sent:
            self._publish_last()
        status = (
            "failed"
            if self._failed
            else "stopped" if self.monitor.abort_requested else "completed"
        )
        self.monitor.finish(status)
        self._last_payload = None
        if self.close_on_finish:
            self.shutdown()

    def shutdown(self):
        """Release the dashboard's port and background threads."""
        if self.monitor is not None:
            self.monitor.shutdown()
            self.monitor = None

    close = shutdown

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


__all__ = ["LiveMonitor", "LiveMonitoringCallback"]
