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
Live monitoring utilities for neuraLQX optimisation runs.

This module implements a lightweight Dash/Plotly web UI for streaming scalar metrics
during training (objective/loss plus optional observables). A training-side callback
enqueues compact scalar payloads into a thread-safe queue, while a background server
thread consumes the queue and renders live plots and a "latest values" stats bar.

The UI runs only on the global MPI master process. It also exposes an "Abort simulation"
button; when pressed, a server-side flag is set and the training callback can request
termination by returning False to the driver.
"""

from __future__ import annotations

import errno
import queue
import socket
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from collections.abc import Iterable

import numpy as np
import requests

import plotly.graph_objs as go

import dash
from dash import Dash
from dash import dcc
from dash import html
from dash.dependencies import Input
from dash.dependencies import Output
from dash.dependencies import State

from flask import request

from neuralqx.utils import distributed as _dist

#
#
#   defaults for port finding and refresh rate

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8050
_MAX_PORT = 8100
_UPDATE_INTERVAL_MS = 100


@dataclass(frozen=True)
class _ScalarStat:
    """
    Container for a scalar mean value and its associated uncertainty.

    :param mean: Scalar mean/expectation value.
    :param sigma: Scalar standard deviation / error estimate associated with `mean`.
    """

    mean: float
    """Scalar mean/expectation value."""

    sigma: float
    """Scalar standard deviation / error estimate associated with `mean`."""


def _as_scalar_stat(obj: Any) -> _ScalarStat | None:
    """
    Best-effort extraction of a scalar (mean, sigma) pair from common "Stats-like" objects.

    Accepted inputs include:
    - Objects exposing `Mean`/`Sigma` or `mean`/`sigma` attributes.
    - Plain scalar numerics (treated as mean with sigma = 0).
    - 0-d numpy arrays (treated as scalar mean with sigma = 0).

    If extraction fails, returns None instead of raising.

    :param obj: Candidate object to interpret as a scalar statistic.
    :return: `_ScalarStat(mean, sigma)` if extraction succeeds, otherwise None.
    """

    if obj is None:
        return None

    mean = getattr(obj, "Mean", None)
    if mean is None:
        mean = getattr(obj, "mean", None)
    sigma = getattr(obj, "Sigma", None)
    if sigma is None:
        sigma = getattr(obj, "sigma", None)

    if mean is not None:
        try:
            m = float(np.real(mean))
        except Exception:
            try:
                m = float(np.real(np.asarray(mean)).item())
            except Exception:
                return None
        if sigma is None:
            s = 0.0
        else:
            try:
                s = float(np.real(sigma))
            except Exception:
                try:
                    s = float(np.real(np.asarray(sigma)).item())
                except Exception:
                    s = 0.0
        return _ScalarStat(m, s)

    # numeric scalar
    if isinstance(obj, (int, float, np.number)):
        return _ScalarStat(float(np.real(obj)), 0.0)

    # 0-d array
    try:
        arr = np.asarray(obj)
        if arr.shape == ():
            return _ScalarStat(float(np.real(arr).item()), 0.0)
    except Exception:
        pass

    return None


def _fmt(x: float) -> str:
    """
    Format a float for compact display in the UI.

    Values with very small magnitude (|x| < 1e-4, excluding 0) are displayed in scientific
    notation, otherwise, fixed-point formatting is used.

    :param x: Value to format.
    :return: Human-readable string representation of `x`.
    """

    ax = abs(x)
    if ax != 0.0 and ax < 1e-4:
        return f"{x:.3e}"
    return f"{x:.4f}"


def _fmt_pm(stat: _ScalarStat) -> str:
    """
    Format a scalar statistic as "mean ± sigma" for display.

    :param stat: Scalar statistic containing a mean and uncertainty.
    :return: Formatted string of the form "<mean> ± <sigma>".
    """

    return f"{_fmt(stat.mean)} ± {_fmt(stat.sigma)}"


class LiveMonitor:
    """
    Dash server plus in-memory time-series store for live training visualisation.

    The training side pushes only small scalar payloads into an internal thread-safe queue.
    A background server thread periodically drains the queue, appends points to per-metric
    time series, and updates a Plotly figure plus a "latest values" stats bar.

    :param host: Interface to bind the Dash server to.
    :param port: Preferred starting port for the Dash server. The monitor searches for an
        available port in [port, _MAX_PORT].
    :param objective_label: Display name for the objective/loss trace (always shown).
    :param x_label: Label for the x-axis (typically iteration/step).
    :param y_label: Label for the y-axis (typically objective expectation).
    :param update_interval_ms: UI polling interval in milliseconds.
     If no port is available, the monitor disables itself (not running).
    """

    # a muted palette, objective uses its own fixed color
    # deep teal
    _OBJECTIVE_COLOR = "#0B3A3B"
    _PALETTE = [
        "#007AFF",  # blue
        "#34C759",  # green
        "#FF9500",  # orange
        "#AF52DE",  # purple
        "#FF3B30",  # red
        "#5AC8FA",  # light blue
        "#5856D6",  # indigo
        "#FF2D55",  # pink
    ]

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        *,
        objective_label: str = "Constraint",
        x_label: str = "Iteration",
        y_label: str = "<C>",
        update_interval_ms: int = _UPDATE_INTERVAL_MS,
    ):
        """
        Create and start the live monitoring server (on the MPI master only).

        Initialises internal queue and time-series storage, builds the Dash layout and callbacks,
        finds an available port, and starts a daemon server thread. If no port is available in
        the configured range, monitoring is disabled (the instance will report `is_running=False`).

        :param host: Interface to bind the server to.
        :param port: Preferred starting port for port probing.
        :param objective_label: Display label for the objective/loss trace.
        :param x_label: X-axis label.
        :param y_label: Y-axis label.
        :param update_interval_ms: Polling interval for UI updates in milliseconds.
        """

        self._lock = threading.Lock()
        self._updates_active = False

        self.host = host
        self.port = None  # type: ignore[assignment]
        self.objective_label = str(objective_label)
        self.x_label = str(x_label)
        self.y_label = str(y_label)

        # fast thread-safe queue for incoming scalar updates
        try:
            self._queue: queue.SimpleQueue[tuple[int, _ScalarStat | None, dict[str, _ScalarStat]]] = queue.SimpleQueue()  # type: ignore[attr-defined]
        except Exception:
            self._queue = queue.Queue()  # type: ignore[assignment]

        # time series: name -> {steps, mean, sigma}
        self._series = defaultdict(lambda: {"steps": [], "mean": [], "sigma": []})
        self._latest: dict[str, _ScalarStat] = {}
        self._last_step: int | None = None
        self._abort_requested = False

        # observable toggles (objective is always shown)
        self._selected_observables: list[str] = []

        # stable colors per observable
        self._color_map: dict[str, str] = {}
        self._palette_idx = 0

        # render bookkeeping
        self._data_version = 0
        self._last_rendered_version = -1
        self._last_rendered_selected: tuple[str, ...] = ()

        # Build Dash app.
        self.app = Dash(
            __name__,
            # avoid external CSS to keep this file drop-in
        )

        # port selection
        available_port = self.find_available_port(port, _MAX_PORT)
        if available_port is None:
            print(
                f"Failed to find an available port between {port} and {_MAX_PORT}.\n"
                "Live Monitoring is disabled. Free a port in that range or pick a different port."
            )
            return

        self.port = available_port
        self._updates_active = True

        # layout
        self.app.layout = self._build_layout(update_interval_ms)

        # callbacks
        self.app.callback(
            [
                Output("live-graph", "figure"),
                Output("stats-bar", "children"),
                Output("obs-checklist", "options"),
                Output("obs-checklist", "value"),
                Output("subtitle-line", "children"),
                Output("interval-component", "disabled"),
            ],
            [
                Input("interval-component", "n_intervals"),
                Input("obs-checklist", "value"),
            ],
            [State("obs-checklist", "options")],
        )(self._update_ui)

        # abort button callback (sets a flag read by the training callback)
        self.app.callback(
            [
                Output("abort-btn", "disabled"),
                Output("abort-btn", "children"),
                Output("abort-btn", "style"),
                Output("right-panel", "style"),
                Output("right-panel-title", "style"),
                Output("right-panel-help", "style"),
                Output("obs-checklist", "style"),
                Output("obs-checklist", "labelStyle"),
            ],
            Input("abort-btn", "n_clicks"),
            [
                State("abort-btn", "style"),
                State("right-panel", "style"),
                State("right-panel-title", "style"),
                State("right-panel-help", "style"),
                State("obs-checklist", "style"),
                State("obs-checklist", "labelStyle"),
            ],
            prevent_initial_call=True,
        )(self._on_abort_clicked)

        # shutdown route (a la POST)
        @self.app.server.route("/shutdown", methods=["POST"])
        def _shutdown_route():  # pragma: no cover
            """
            Handle a POST request to shut down the live monitoring server.

            Disables periodic UI updates and attempts to invoke the Werkzeug server shutdown hook
            when available.

            :return: Plaintext response indicating shutdown has been initiated.
            """

            self._stop_updates()
            func = request.environ.get("werkzeug.server.shutdown")
            if func is not None:
                func()
            return "neuraLQX Live Monitoring is shutting down..."

        # start server
        self._thread = threading.Thread(
            target=self._run_server,
            args=(host, available_port),
            daemon=True,
        )
        self._thread.start()
        print(f"neuraLQX Live Monitor is running on http://{host}:{available_port}")

    #
    #
    #   public API

    def log(
        self,
        step: int,
        objective: tuple[float, float] | None = None,
        observables: dict[str, tuple[float, float]] | None = None,
        # back-compat: allow old signature log(step, log_data_dict)
        log_data: dict[str, Any] | None = None,
    ) -> None:
        """
        Enqueue a compact scalar payload for the UI.

        Preferred usage:
            log(step, objective=(mean, sigma), observables={"E": (m, s), ...})

        Backwards compatible usage:
            log(step, log_data={"Loss": stats_obj, "Obs": stats_obj, ...})

        When using `log_data`, entries are parsed via `_as_scalar_stat`, only parsable scalar
        values are forwarded. If monitoring is not running, this method is a no-op.

        :param step: Integer optimisation step associated with the payload.
        :param objective: Optional `(mean, sigma)` pair for the objective/loss.
        :param observables: Optional mapping `name -> (mean, sigma)` for additional traces.
        :param log_data: Optional legacy mapping `name -> Stats-like object` to be parsed.
        """

        if not self.is_running:
            return

        if log_data is not None and (objective is None and observables is None):
            # parse dict values into scalar stats
            obj = None
            obs: dict[str, _ScalarStat] = {}
            for k, v in log_data.items():
                st = _as_scalar_stat(v)
                if st is None:
                    continue
                if k == "Loss":
                    obj = st
                else:
                    obs[str(k)] = st
            self._queue.put((int(step), obj, obs))
            return

        obj_stat = None
        if objective is not None:
            try:
                obj_stat = _ScalarStat(
                    float(np.real(objective[0])), float(np.real(objective[1]))
                )
            except Exception:
                obj_stat = None

        obs_stats: dict[str, _ScalarStat] = {}
        if observables:
            for name, (m, s) in observables.items():
                try:
                    obs_stats[str(name)] = _ScalarStat(
                        float(np.real(m)), float(np.real(s))
                    )
                except Exception:
                    continue

        self._queue.put((int(step), obj_stat, obs_stats))

    @property
    def is_running(self) -> bool:
        """
        Whether the monitor is actively running and accepting updates.

        :return: True if a port was successfully bound and UI updates are active, otherwise False.
        """

        return bool(self.port) and self._updates_active

    def shutdown(self) -> None:
        """
        Stop UI updates and attempt to shut down the Dash/Flask server.

        This is best-effort and intentionally non-blocking. It disables updates immediately,
        then tries a short-timeout HTTP POST to the internal shutdown route, and finally attempts
        to join the server thread with a short timeout.
        """

        if not self.port:
            return

        self._stop_updates()

        # best-effort HTTP shutdown (fast timeout, never block training teardown)
        try:
            requests.post(
                f"http://{self.host}:{self.port}/shutdown",
                timeout=0.25,
            )
        except Exception:
            pass

        # best-effort join
        try:
            self._thread.join(timeout=0.5)
        except Exception:
            pass

    #
    #
    #   dash internals

    def _stop_updates(self) -> None:
        """
        Disable periodic UI updates in a thread-safe manner.
        """

        with self._lock:
            self._updates_active = False

    def _run_server(self, host: str, port: int) -> None:
        """
        Run the Dash server in the current thread.

        This is intended to be used as the target of a daemon `threading.Thread`.

        :param host: Interface to bind the server to.
        :param port: Port to bind the server to.
        """

        self.app.run(
            host=host,
            port=port,
            debug=False,
            use_reloader=False,
        )

    @staticmethod
    def find_available_port(start_port: int, max_port: int) -> int | None:
        """
        Find the first available TCP port in a given inclusive range.

        Ports are tested by attempting to bind a local socket. If binding fails, the search
        continues until `max_port` is exceeded.

        :param start_port: First port to probe.
        :param max_port: Last port to probe (inclusive).
        :return: The first available port, or None if no port is available.
        """

        port = int(start_port)
        while port <= max_port:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind((_DEFAULT_HOST, port))
                    return port
                except OSError as e:
                    if e.errno == errno.EADDRINUSE:
                        port += 1
                    else:
                        port += 1
        return None

    def _build_layout(self, update_interval_ms: int):
        """
        Construct the Dash layout for the live monitoring UI.

        Builds a single-page layout containing:
        - A Plotly graph for time-series traces with error bars.
        - A stats bar showing the latest values for the objective and selected observables.
        - A checklist to toggle observable traces.
        - An abort button to request termination of the training run.

        :param update_interval_ms: Polling interval for the UI update callback in milliseconds.
        :return: A Dash layout (HTML Div tree) suitable for `app.layout`.
        """

        font_stack = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

        page_style = {
            "height": "100vh",
            "width": "100vw",
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "center",
            "backgroundColor": "#F5F5F7",
            "fontFamily": font_stack,
        }

        card_style = {
            "width": "92vw",
            "maxWidth": "1400px",
            "backgroundColor": "white",
            "borderRadius": "18px",
            "boxShadow": "0 16px 40px rgba(0,0,0,0.10)",
            "padding": "18px 18px 14px 18px",
        }

        header_style = {
            "fontSize": "20px",
            "fontWeight": "600",
            "letterSpacing": "0.2px",
            "textAlign": "left",
            "marginBottom": "4px",
            "color": "#111111",
        }

        subtitle_style = {
            "fontSize": "12px",
            "color": "#6E6E73",
            "marginBottom": "14px",
        }

        body_style = {
            "display": "flex",
            "gap": "16px",
            "alignItems": "stretch",
        }

        left_col_style = {
            "flex": "1 1 auto",
            "minWidth": "720px",
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px",
        }

        right_col_style = {
            "width": "290px",
            "flex": "0 0 290px",
            "borderLeft": "1px solid #EFEFF4",
            "paddingLeft": "16px",
            "display": "flex",
            "flexDirection": "column",
        }

        panel_title_style = {
            "fontSize": "14px",
            "fontWeight": "600",
            "color": "#1C1C1E",
            "marginBottom": "10px",
        }

        panel_help_style = {
            "fontSize": "12px",
            "color": "#6E6E73",
            "marginBottom": "10px",
            "lineHeight": "16px",
        }

        checklist_style = {
            "display": "flex",
            "flexDirection": "column",
            "gap": "8px",
            "padding": "10px 10px 12px 10px",
            "backgroundColor": "#F8F8FA",
            "border": "1px solid #EFEFF4",
            "borderRadius": "14px",
            "maxHeight": "520px",
            "overflowY": "auto",
        }

        stats_style = {
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "8px",
            "padding": "10px 10px",
            "borderRadius": "14px",
            "backgroundColor": "#F8F8FA",
            "border": "1px solid #EFEFF4",
            "minHeight": "44px",
            "alignItems": "center",
        }

        graph_config = {
            "displaylogo": False,
            "responsive": True,
        }

        abort_btn_style = {
            "width": "100%",
            "marginTop": "12px",
            "padding": "10px 12px",
            "borderRadius": "12px",
            "border": "1px solid rgba(0,0,0,0.08)",
            "backgroundColor": "#7A1116",
            "color": "white",
            "fontSize": "13px",
            "fontWeight": "600",
            "cursor": "pointer",
            "outline": "none",
        }

        abort_btn_wrapper_style = {
            "marginTop": "auto",
            "paddingTop": "12px",
            "paddingBottom": "2px",
        }

        return html.Div(
            style=page_style,
            children=[
                html.Div(
                    style=card_style,
                    children=[
                        html.Div("neuraLQX Live Monitoring", style=header_style),
                        html.Div(id="subtitle-line", style=subtitle_style),
                        html.Div(
                            style=body_style,
                            children=[
                                html.Div(
                                    style=left_col_style,
                                    children=[
                                        dcc.Graph(
                                            id="live-graph",
                                            figure=self._empty_figure(),
                                            config=graph_config,
                                            style={"height": "560px"},
                                        ),
                                        html.Div(
                                            id="stats-bar",
                                            style=stats_style,
                                            children=[
                                                self._pill(
                                                    self.objective_label,
                                                    "—",
                                                    color=self._OBJECTIVE_COLOR,
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="right-panel",
                                    style=right_col_style,
                                    children=[
                                        html.Div(
                                            "Observables",
                                            id="right-panel-title",
                                            style=panel_title_style,
                                        ),
                                        html.Div(
                                            "Toggle observables to plot them and show their current expectation below.",
                                            id="right-panel-help",
                                            style=panel_help_style,
                                        ),
                                        dcc.Checklist(
                                            id="obs-checklist",
                                            options=[],
                                            value=[],
                                            labelStyle={
                                                "display": "flex",
                                                "alignItems": "center",
                                                "gap": "10px",
                                                "fontSize": "13px",
                                                "color": "#1C1C1E",
                                                "padding": "6px 6px",
                                                "borderRadius": "10px",
                                            },
                                            inputStyle={
                                                "width": "16px",
                                                "height": "16px",
                                            },
                                            style=checklist_style,
                                        ),
                                        html.Div(
                                            style=abort_btn_wrapper_style,
                                            children=[
                                                html.Button(
                                                    "Abort simulation",
                                                    id="abort-btn",
                                                    n_clicks=0,
                                                    style=abort_btn_style,
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Interval(
                            id="interval-component",
                            interval=int(update_interval_ms),
                            n_intervals=0,
                        ),
                    ],
                )
            ],
        )

    @property
    def abort_requested(self) -> bool:
        """
        Whether the UI has requested an abort of the running simulation.

        This flag is set when the user presses the abort button. It is intended to be polled by
        the training-side callback, which may stop optimisation by returning False to the driver.

        :return: True if an abort has been requested, otherwise False.
        """

        return bool(self._abort_requested)

    def _on_abort_clicked(
        self,
        n_clicks: int | None,
        btn_style: dict[str, Any] | None,
        panel_style: dict[str, Any] | None,
        title_style: dict[str, Any] | None,
        help_style: dict[str, Any] | None,
        checklist_style_in: dict[str, Any] | None,
        checklist_label_style_in: dict[str, Any] | None,
    ):
        """
        Handle an abort button click by setting the abort flag and updating UI styling.

        When clicked, this:
        - Sets an internal abort flag (thread-safe) for the training callback to poll.
        - Disables the abort button and changes its appearance.
        - Restyles the right-hand panel and checklist to visually indicate an aborted state.

        :param n_clicks: Dash click count for the abort button.
        :param btn_style: Current button style dict (may be None).
        :param panel_style: Current right-panel style dict (may be None).
        :param title_style: Current right-panel title style dict (may be None).
        :param help_style: Current right-panel help style dict (may be None).
        :param checklist_style_in: Current checklist container style dict (may be None).
        :param checklist_label_style_in: Current checklist label style dict (may be None).
        :return: tuple of Dash outputs matching the callback signature (updated disabled state,
            button text, and style dictionaries).
        """

        if not n_clicks:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        # flip server-side flag polled by the NetKet callback
        with self._lock:
            self._abort_requested = True

        deep_red = "#7A1116"

        # fainter version for checklist box
        faint_red = "rgba(122,17,22,0.35)"

        # button gets disabled + "aborted" look
        new_btn_style = dict(btn_style or {})
        new_btn_style.update(
            {
                "backgroundColor": "rgba(255,255,255,0.10)",
                "border": "1px solid rgba(255,255,255,0.22)",
                "cursor": "default",
            }
        )

        # right panel gets a deep red background
        new_panel_style = dict(panel_style or {})
        new_panel_style.update(
            {
                "backgroundColor": deep_red,
                "borderLeft": "1px solid rgba(0,0,0,0.08)",
                "borderRadius": "14px",
                "padding": "16px",
            }
        )

        # title/help becomes white text
        new_title_style = dict(title_style or {})
        new_title_style.update({"color": "rgba(255,255,255,0.95)"})

        new_help_style = dict(help_style or {})
        new_help_style.update({"color": "rgba(255,255,255,0.80)"})

        # checklist container box is now faint red + subtle border
        new_checklist_style = dict(checklist_style_in or {})
        new_checklist_style.update(
            {
                "backgroundColor": faint_red,
                "border": "1px solid rgba(255,255,255,0.18)",
            }
        )

        # checklist labels turn white
        new_checklist_label_style = dict(checklist_label_style_in or {})
        new_checklist_label_style.update({"color": "rgba(255,255,255,0.92)"})

        return (
            True,
            "Simulation aborted",
            new_btn_style,
            new_panel_style,
            new_title_style,
            new_help_style,
            new_checklist_style,
            new_checklist_label_style,
        )

    def _empty_figure(self) -> dict[str, Any]:
        """
        Create an empty Plotly figure with the monitor's configured axis labels and styling.

        :return: A Plotly figure dict with no traces and a preconfigured layout.
        """

        layout = go.Layout(
            template="plotly_white",
            margin=dict(l=70, r=20, t=20, b=55),
            xaxis=dict(
                title=self.x_label,
                gridcolor="#EFEFF4",
                zeroline=False,
                showline=False,
            ),
            yaxis=dict(
                title=self.y_label,
                gridcolor="#EFEFF4",
                zeroline=False,
                showline=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                font=dict(size=12),
            ),
            hovermode="closest",
            uirevision="live",  # preserve zoom while updating
        )
        return {"data": [], "layout": layout}

    def _pill(self, name: str, value: str, *, color: str) -> html.Div:
        """
        Create a small "pill" UI element for the stats bar.

        The pill contains a colored dot, a metric name, and a formatted value string.

        :param name: Label to display (metric name).
        :param value: Value string to display (already formatted).
        :param color: Color used for the indicator dot (and associated identity).
        :return: A Dash HTML Div representing the pill.
        """

        return html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "8px",
                "padding": "6px 10px",
                "borderRadius": "999px",
                "backgroundColor": "white",
                "border": "1px solid #EFEFF4",
            },
            children=[
                html.Span(
                    "●",
                    style={
                        "color": color,
                        "fontSize": "12px",
                        "lineHeight": "12px",
                    },
                ),
                html.Span(
                    name,
                    style={
                        "fontSize": "12px",
                        "fontWeight": "600",
                        "color": "#1C1C1E",
                    },
                ),
                html.Span(
                    value,
                    style={
                        "fontSize": "12px",
                        "color": "#3A3A3C",
                        "whiteSpace": "nowrap",
                    },
                ),
            ],
        )

    def _color_for(self, name: str) -> str:
        """
        Get or assign a stable color for an observable name.

        Colors are assigned from an internal palette on first use and then cached so that
        each observable keeps the same color across updates.

        :param name: Observable name.
        :return: Hex color string associated with `name`.
        """

        if name in self._color_map:
            return self._color_map[name]
        color = self._PALETTE[self._palette_idx % len(self._PALETTE)]
        self._palette_idx += 1
        self._color_map[name] = color
        return color

    def _consume_queue(self) -> bool:
        """
        Drain queued scalar updates into the in-memory time-series store.

        Consumes all currently available queue entries, updates the per-metric series arrays,
        updates the "latest value" cache, and bumps an internal data version counter if any
        points were added.

        :return: True if at least one datapoint was appended, otherwise False.
        """

        if not self.is_running:
            return False

        added = False
        while True:
            try:
                step, obj, obs = self._queue.get_nowait()
            except queue.Empty:
                break

            self._last_step = step

            if obj is not None:
                s = self._series[self.objective_label]
                s["steps"].append(step)
                s["mean"].append(obj.mean)
                s["sigma"].append(obj.sigma)
                self._latest[self.objective_label] = obj
                added = True

            for name, stat in obs.items():
                if not name or name == self.objective_label:
                    continue
                s = self._series[name]
                s["steps"].append(step)
                s["mean"].append(stat.mean)
                s["sigma"].append(stat.sigma)
                self._latest[name] = stat
                added = True

        if added:
            self._data_version += 1
        return added

    def _build_traces(self, selected_observables: Iterable[str]) -> list[go.Scatter]:
        """
        Build Plotly traces for the objective and selected observables.

        The objective trace is always included when data exists. Each selected observable is
        rendered as a line+marker trace with error bars using its stable assigned color.

        :param selected_observables: Iterable of observable names to include.
        :return: list of Plotly `go.Scatter` traces.
        """

        traces: list[go.Scatter] = []

        # objective always
        obj = self._series.get(self.objective_label)
        if obj and obj["mean"]:
            traces.append(
                go.Scatter(
                    x=obj["steps"],
                    y=obj["mean"],
                    mode="lines+markers",
                    name=self.objective_label,
                    marker=dict(size=6, color=self._OBJECTIVE_COLOR),
                    line=dict(color=self._OBJECTIVE_COLOR, width=2),
                    error_y=dict(
                        type="data",
                        array=obj["sigma"],
                        visible=True,
                        thickness=1.2,
                        width=4,
                        color=self._OBJECTIVE_COLOR,
                    ),
                )
            )

        for name in selected_observables:
            if name == self.objective_label:
                continue
            s = self._series.get(name)
            if not s or not s["mean"]:
                continue
            c = self._color_for(name)
            traces.append(
                go.Scatter(
                    x=s["steps"],
                    y=s["mean"],
                    mode="lines+markers",
                    name=name,
                    marker=dict(size=5, color=c),
                    line=dict(color=c, width=2),
                    error_y=dict(
                        type="data",
                        array=s["sigma"],
                        visible=True,
                        thickness=1.0,
                        width=3,
                        color=c,
                    ),
                )
            )

        return traces

    def _build_stats_bar(self, selected_observables: Iterable[str]):
        """
        Build the "latest values" stats bar content.

        Always includes the objective pill, and includes pills for each selected observable
        showing the most recently received value (formatted as mean ± sigma).

        :param selected_observables: Iterable of observable names to include.
        :return: list of Dash HTML elements for the stats bar.
        """

        pills: list[html.Div] = []

        # Always show objective.
        obj_stat = self._latest.get(self.objective_label)
        pills.append(
            self._pill(
                self.objective_label,
                _fmt_pm(obj_stat) if obj_stat is not None else "—",
                color=self._OBJECTIVE_COLOR,
            )
        )

        # show selected observables
        for name in selected_observables:
            stat = self._latest.get(name)
            pills.append(
                self._pill(
                    name,
                    _fmt_pm(stat) if stat is not None else "—",
                    color=self._color_for(name),
                )
            )

        return pills

    def _update_ui(
        self,
        n: int,
        selected: list[str] | None,
        _options_state,  # pylint: disable=unused-argument
    ):
        """
        Dash callback that updates the figure, stats bar, and observable checklist.

        This callback:
        - Stops updating if shutdown has begun.
        - Drains queued updates into the time-series store.
        - Computes checklist options from known observable names.
        - Preserves user selection while filtering out unknown values.
        - Avoids re-rendering when neither data nor selection changed.

        :param n: Interval tick count (unused except to trigger updates).
        :param selected: Current checklist selection (observable names).
        :param _options_state: Current checklist options state (unused).
        :return: tuple matching the Dash outputs: updated figure, stats bar children, checklist
            options, checklist value, subtitle text, and interval disabled flag.
        """

        # disable interval as soon as shutdown starts
        with self._lock:
            active = self._updates_active
        if not active:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                True,
            )

        self._consume_queue()
        subtitle = (
            f"Last step: {self._last_step}" if self._last_step is not None else ""
        )

        # known observables (exclude objective)
        observable_names = sorted(
            [k for k in self._series.keys() if k != self.objective_label]
        )
        options = [{"label": name, "value": name} for name in observable_names]

        # keep the last selection stable, only accept values that exist
        if selected is None:
            selected = list(self._selected_observables)
        selected = [s for s in selected if s in observable_names]
        self._selected_observables = list(selected)

        # if nothing changed, avoid re-render
        selected_tuple = tuple(selected)
        if (
            self._data_version == self._last_rendered_version
            and selected_tuple == self._last_rendered_selected
        ):
            return (
                dash.no_update,
                dash.no_update,
                options,
                selected,
                dash.no_update,
                False,
            )

        self._last_rendered_version = self._data_version
        self._last_rendered_selected = selected_tuple

        fig = self._empty_figure()
        fig["data"] = self._build_traces(selected)

        stats = self._build_stats_bar(selected)
        return fig, stats, options, selected, subtitle, False


class LiveMonitoringCallback:
    """
    NetKet callback that streams scalar metrics into a :class:`LiveMonitor`.

    This callback is designed to be registered with a NetKet/neuraLQX driver. On each
    iteration it extracts a scalar objective statistic and any scalar observables from
    the driver log, then enqueues them into the live monitor. If the UI requested an
    abort, it returns False to ask the driver to stop.

    :param host: Host/interface for the live monitor server.
    :param port: Preferred starting port for the live monitor server.
    :param objective_label: Display label for the objective/loss trace.
     On non-master MPI ranks, monitoring is disabled (`monitor=None`).
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        *,
        objective_label: str = "Constraint",
    ):
        """
        Initialise the training-side live monitoring callback.

        On the global MPI master process, constructs a :class:`LiveMonitor`. On all other ranks,
        disables monitoring by setting `monitor=None` to avoid multiple servers and duplicate I/O.

        :param host: Host/interface for the live monitor server.
        :param port: Preferred starting port for the live monitor server.
        :param objective_label: Display label for the objective/loss trace.
        """

        if not _dist.is_global_master():
            self.monitor = None
            return
        self.monitor = LiveMonitor(host, port, objective_label=objective_label)

    def __call__(self, step: int, log_data: dict[str, Any], driver: Any):
        """
        Consume driver log data, enqueue scalar metrics, and optionally request abort.

        If the UI has requested abort, returns False to ask the driver to stop. Otherwise,
        extracts the objective statistic from the driver’s configured loss name (falling back
        to "Loss") and parses any other scalar Stats-like entries as observables.

        :param step: Current optimisation step.
        :param log_data: Mapping from metric names to Stats-like objects or scalar values.
        :param driver: Driver instance providing `_loss_name` (optional) and context.
        :return: False if an abort was requested; True otherwise.
        """

        # if the UI requested abort, stop the simulation.
        if getattr(self.monitor, "abort_requested", False):
            return False

        if self.monitor is None or not self.monitor.is_running:
            return True

        # identify the objective/loss metric
        loss_metric_name = getattr(driver, "_loss_name", "Loss")
        loss_obj = log_data.get(loss_metric_name, None)
        if loss_obj is None and loss_metric_name != "Loss":
            loss_obj = log_data.get("Loss", None)

        objective_stat = _as_scalar_stat(loss_obj)

        # extract any other scalar Stats as observables
        observables: dict[str, tuple[float, float]] = {}
        for name, val in log_data.items():
            if name in {loss_metric_name, "Loss"}:
                continue
            st = _as_scalar_stat(val)
            if st is None:
                continue
            observables[str(name)] = (st.mean, st.sigma)

        # enqueue (tiny payload)
        if objective_stat is not None:
            self.monitor.log(
                step,
                objective=(objective_stat.mean, objective_stat.sigma),
                observables=observables,
            )
        elif observables:
            # still enqueue observables even if objective is missing
            self.monitor.log(step, objective=None, observables=observables)

        return True

    def shutdown(self) -> None:
        """
        Shut down the underlying live monitor, if present.
        """

        if self.monitor is None:
            return
        self.monitor.shutdown()
