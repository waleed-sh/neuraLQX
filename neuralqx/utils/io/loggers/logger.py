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

import copy
import os
from typing import Any
import html
import json


from datetime import datetime
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box

from netket.stats.mc_stats import Stats

from ._abstract_logger import AbstractLogger
from ....utils.misc.auth import Authenticator
from ...._version import __version__
from ..printing import NQXPrinter
from ....utils import distributed as _dist

console = Console()


def escape_html(text) -> Any:
    """Helper to escape HTML characters."""
    return html.escape(text)


def format_value(value) -> Any:
    """Format the value, escaping special characters."""
    return escape_html(str(value))


def _to_jsonable(obj: Any) -> Any:
    """Convert arbitrary objects into JSON-serializable data (fallback to str)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Stats):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    try:
        item = getattr(obj, "item", None)
        if callable(item):
            return _to_jsonable(obj.item())
    except Exception:
        pass
    return str(obj)


def _flatten_for_rows(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested dicts into (key_path, value) rows; keep lists/scalars as values."""
    rows: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix} - {k}" if prefix else str(k)
            if isinstance(v, dict):
                rows.extend(_flatten_for_rows(v, kp))
            else:
                rows.append((kp, v))
    else:
        rows.append((prefix or "value", obj))
    return rows


def dict_to_html(data_dict: dict, *, meta: dict | None = None) -> str:
    meta = meta or {}
    safe_data = _to_jsonable(data_dict)
    safe_meta = _to_jsonable(meta)

    # embed JSON safely inside script tags
    # only escape '</script>' possibility
    data_json = json.dumps(safe_data, ensure_ascii=False).replace("</", "<\\/")
    meta_json = json.dumps(safe_meta, ensure_ascii=False).replace("</", "<\\/")

    sections = list(safe_data.keys())

    # STATIC FALLBACK
    # render all sections as accordions
    fallback_parts = []
    for sec in sections:
        rows = _flatten_for_rows(safe_data.get(sec, {}))
        # table rows
        trows = []
        for k, v in rows:
            sv = "" if v is None else v
            sv_str = sv if isinstance(sv, str) else json.dumps(sv, ensure_ascii=False)
            k_esc = escape_html(str(k))
            v_esc = escape_html(str(sv_str))

            # long values: use <details> fallback
            if len(str(sv_str)) > 140 or "\n" in str(sv_str) or "\r" in str(sv_str):
                cell = (
                    f"<details class='nqx-details'>"
                    f"<summary class='nqx-summary'>Show value</summary>"
                    f"<pre class='nqx-pre-static'>{v_esc}</pre>"
                    f"</details>"
                )
            else:
                cell = f"<span class='nqx-inline-static'>{v_esc}</span>"

            trows.append(
                f"<tr><td class='nqx-key-static'>{k_esc}</td>"
                f"<td class='nqx-val-static'>{cell}</td></tr>"
            )

        fallback_parts.append(f"""
            <details class="nqx-fallback-section" open>
              <summary class="nqx-fallback-title">{escape_html(sec)}</summary>
              <div class="nqx-fallback-tablewrap">
                <table class="nqx-fallback-table">
                  <thead>
                    <tr><th>Key</th><th>Value</th></tr>
                  </thead>
                  <tbody>
                    {''.join(trows) if trows else '<tr><td colspan="2" class="nqx-empty">No entries</td></tr>'}
                  </tbody>
                </table>
              </div>
            </details>
            """)

    fallback_html = f"""
    <div class="nqx-noscript">
      <div class="nqx-noscript-card">
        <div class="nqx-noscript-title">Interactive view needs JavaScript</div>
        <div class="nqx-noscript-sub">You’re seeing the static fallback (still complete). Open in a browser with scripts enabled for the sidebar UI.</div>
      </div>
      {''.join(fallback_parts) if fallback_parts else '<div class="nqx-empty">No sections in log.</div>'}
    </div>
    """

    # JS ENHANCED VIEW
    return f"""
<div class="nqx-shell">
  <div class="nqx-card" id="nqx-app" aria-hidden="false">
    <aside class="nqx-sidebar">
      <div class="nqx-brand">
        <div>
          <div class="nqx-title">neuraLQX Output Log</div>
        </div>
      </div>

      <div class="nqx-searchwrap">
        <input class="nqx-search" id="nqx-section-search" type="search" placeholder="Search sections…" spellcheck="false"/>
      </div>

      <nav class="nqx-nav" id="nqx-nav"></nav>

      <div class="nqx-sidebar-footer">
      </div>
    </aside>

    <main class="nqx-main">
      <header class="nqx-main-header">
        <div class="nqx-main-head-left">
          <div class="nqx-h1" id="nqx-active-title">—</div>
          <div class="nqx-h2" id="nqx-active-subtitle">—</div>
        </div>
        <div class="nqx-main-head-right">
          <input class="nqx-search nqx-search-main" id="nqx-row-search" type="search" placeholder="Filter keys/values…" spellcheck="false"/>
        </div>
      </header>

      <section class="nqx-content">
        <div class="nqx-tablewrap">
          <table class="nqx-table">
            <thead>
              <tr>
                <th class="nqx-th-key">Key</th>
                <th class="nqx-th-val">Value</th>
              </tr>
            </thead>
            <tbody id="nqx-tbody"></tbody>
          </table>
        </div>
      </section>

      <footer class="nqx-footer">
        <div class="nqx-foot-left">Built by neuraLQX</div>
        <div class="nqx-foot-right" id="nqx-foot-meta">—</div>
      </footer>
    </main>
  </div>

  <noscript>
    {fallback_html}
  </noscript>

  <div id="nqx-fallback" style="display:none;">
    {fallback_html}
  </div>
</div>

<script id="nqx-data" type="application/json">{data_json}</script>
<script id="nqx-meta" type="application/json">{meta_json}</script>

<style>
  :root {{
    --bg: #F5F5F7;
    --card: rgba(255,255,255,0.96);
    --text: #111111;
    --muted: #6E6E73;
    --line: rgba(0,0,0,0.08);
    --shadow: 0 20px 60px rgba(0,0,0,0.12);
    --radius: 22px;
    --accent: #007AFF;
    --accent2: rgba(0,122,255,0.12);
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  }}

  html, body {{
    height: 100%;
    margin: 0;
    padding: 0;
    background: var(--bg);
    font-family: var(--sans);
    color: var(--text);
  }}

  .nqx-shell {{
    height: 100vh;
    width: 100vw;
    display: grid;
    place-items: center;
  }}

  .nqx-card {{
    width: min(1320px, 94vw);
    height: min(860px, 92vh);
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    overflow: hidden;
    display: grid;
    grid-template-columns: 320px 1fr;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
  }}

  .nqx-sidebar {{
    position: relative;
    padding: 18px 14px 14px 14px;
    border-right: 1px solid var(--line);
    background: linear-gradient(180deg, rgba(0,0,0,0.02), rgba(0,0,0,0.00));
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
  }}

  .nqx-sidebar::before {{
    content: "";
    position: absolute;
    top: 0; left: 0;
    height: 100%;
    width: 5px;
    background: var(--accent);
    opacity: 0.95;
  }}

  .nqx-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 10px;
    border-radius: 16px;
  }}

  .nqx-dot {{
    width: 12px;
    height: 12px;
    border-radius: 999px;
    background: var(--accent);
    box-shadow: 0 0 0 6px var(--accent2);
  }}

  .nqx-title {{
    font-size: 14px;
    font-weight: 650;
    letter-spacing: 0.2px;
  }}

  .nqx-subtitle {{
    margin-top: 3px;
    font-size: 12px;
    color: var(--muted);
  }}

  .nqx-searchwrap {{
    padding: 10px 10px 8px 10px;
  }}

  .nqx-search {{
    width: 100%;
    padding: 10px 12px;
    border-radius: 14px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.85);
    color: var(--text);
    outline: none;
    font-size: 13px;
  }}

  .nqx-search:focus {{
    border-color: rgba(0,0,0,0.18);
    box-shadow: 0 0 0 6px var(--accent2);
  }}

  .nqx-nav {{
    padding: 4px 6px;
    margin: 2px 4px 10px 4px;
    flex: 1 1 auto;
    min-height: 0;
    overflow: auto;
    scrollbar-gutter: stable;
  }}

  .nqx-item {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding: 10px 10px;
    margin: 6px 0;
    border-radius: 14px;
    cursor: pointer;
    border: 1px solid transparent;
    user-select: none;
  }}

  .nqx-item:hover {{
    background: rgba(0,0,0,0.03);
    border-color: var(--line);
  }}

  .nqx-item.active {{
    background: var(--accent2);
    border-color: rgba(0,0,0,0.10);
  }}

  .nqx-item-left {{
    display: flex;
    flex-direction: column;
    gap: 3px;
    min-width: 0;
  }}

  .nqx-item-name {{
    font-size: 13px;
    font-weight: 650;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  .nqx-item-hint {{
    font-size: 12px;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  .nqx-badge {{
    font-size: 12px;
    color: var(--muted);
    border: 1px solid var(--line);
    padding: 4px 8px;
    border-radius: 999px;
    background: rgba(255,255,255,0.80);
  }}

  .nqx-sidebar-footer {{
    padding: 14px 14px 14px 14px;
    margin-top: auto;   /* pushes footer to bottom */
  }}

  .nqx-chip {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 8px 10px;
    background: rgba(255,255,255,0.80);
  }}

  .nqx-main {{
    display: grid;
    grid-template-rows: auto 1fr auto;
    min-width: 0;
    min-height: 0;
  }}

  .nqx-main-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 14px;
    padding: 18px 18px 12px 18px;
    border-bottom: 1px solid var(--line);
  }}

  .nqx-h1 {{
    font-size: 18px;
    font-weight: 750;
    letter-spacing: 0.2px;
  }}

  .nqx-h2 {{
    margin-top: 5px;
    font-size: 12px;
    color: var(--muted);
  }}

  .nqx-search-main {{
    width: min(420px, 42vw);
  }}

  .nqx-content {{
    min-height: 0;
    overflow: hidden;
    padding: 14px 18px;
  }}

  .nqx-tablewrap {{
    height: 100%;
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: rgba(255,255,255,0.78);
  }}

  .nqx-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    table-layout: fixed;
  }}

  .nqx-table thead th {{
    position: sticky;
    top: 0;
    z-index: 2;
    background: rgba(255,255,255,0.95);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    text-align: left;
    padding: 12px 14px;
    border-bottom: 1px solid var(--line);
    font-weight: 650;
  }}

  .nqx-th-key {{ width: 34%; }}
  .nqx-th-val {{ width: 66%; }}

  .nqx-table tbody td {{
    padding: 12px 14px;
    border-bottom: 1px solid var(--line);
    vertical-align: top;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}

  .nqx-row:hover td {{
    background: rgba(0,0,0,0.02);
  }}

  .nqx-key {{
    font-weight: 650;
    color: rgba(0,0,0,0.78);
    white-space: normal;
    overflow-wrap: anywhere;
  }}

  .nqx-val {{
    color: rgba(0,0,0,0.72);
    line-height: 1.35;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}

  .nqx-val .nqx-inline {{
    display: inline;
    white-space: normal;
    overflow: visible;
  }}

  .nqx-expand {{
    margin-left: 10px;
    font-size: 12px;
    border: 1px solid var(--line);
    background: transparent;
    color: var(--muted);
    padding: 4px 8px;
    border-radius: 999px;
    cursor: pointer;
  }}

  .nqx-pre {{
    margin-top: 10px;
    padding: 10px 12px;
    border-radius: 14px;
    border: 1px solid var(--line);
    background: rgba(0,0,0,0.03);
    font-family: var(--mono);
    font-size: 12px;
    overflow: auto;
    max-height: 260px;
    display: none;
    white-space: pre-wrap;
  }}

  .nqx-pre.show {{
    display: block;
  }}

  .nqx-footer {{
    flex: 0 0 auto;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 18px 14px 18px;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 12px;
  }}

  /* Fallback styles */
  .nqx-noscript {{
    width: min(1320px, 94vw);
    height: min(860px, 92vh);
    overflow: auto;
    padding: 18px;
    box-sizing: border-box;
  }}
  .nqx-noscript-card {{
    background: rgba(255,255,255,0.96);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 14px;
  }}
  .nqx-noscript-title {{
    font-weight: 750;
    font-size: 14px;
  }}
  .nqx-noscript-sub {{
    color: var(--muted);
    font-size: 12px;
    margin-top: 6px;
    line-height: 1.3;
  }}
  .nqx-fallback-section {{
    background: rgba(255,255,255,0.96);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 10px 12px;
    margin: 10px 0;
  }}
  .nqx-fallback-title {{
    cursor: pointer;
    font-weight: 750;
    font-size: 13px;
    padding: 6px 0;
  }}
  .nqx-fallback-tablewrap {{
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 14px;
    margin-top: 10px;
    background: rgba(255,255,255,0.85);
  }}
  .nqx-fallback-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  .nqx-fallback-table th, .nqx-fallback-table td {{
    padding: 10px 12px;
    border-bottom: 1px solid var(--line);
    text-align: left;
    vertical-align: top;
  }}
  .nqx-key-static {{ font-weight: 650; }}
  .nqx-inline-static {{
    display: inline-block;
    max-width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .nqx-details > summary {{
    cursor: pointer;
    color: var(--muted);
    font-size: 12px;
  }}
  .nqx-pre-static {{
    margin: 10px 0 0 0;
    padding: 10px 12px;
    border-radius: 12px;
    border: 1px solid var(--line);
    background: rgba(0,0,0,0.03);
    font-family: var(--mono);
    font-size: 12px;
    white-space: pre-wrap;
  }}
  .nqx-empty {{
    color: var(--muted);
    font-size: 13px;
    padding: 12px;
  }}
</style>

<script>
(function() {{
  let DATA = {{}};
  let META = {{}};
  try {{
    DATA = JSON.parse(document.getElementById("nqx-data").textContent || "{{}}");
    META = JSON.parse(document.getElementById("nqx-meta").textContent || "{{}}");
  }} catch (e) {{
    const fb = document.getElementById("nqx-fallback");
    if (fb) fb.style.display = "block";
    const app = document.getElementById("nqx-app");
    if (app) app.style.display = "none";
    return;
  }}

  const nav = document.getElementById("nqx-nav");
  const tbody = document.getElementById("nqx-tbody");
  const sectionSearch = document.getElementById("nqx-section-search");
  const rowSearch = document.getElementById("nqx-row-search");
  const footMeta = document.getElementById("nqx-foot-meta");
  const titleEl = document.getElementById("nqx-active-title");
  const subtitleEl = document.getElementById("nqx-active-subtitle");

  if (!nav || !tbody) {{
    const fb = document.getElementById("nqx-fallback");
    if (fb) fb.style.display = "block";
    const app = document.getElementById("nqx-app");
    if (app) app.style.display = "none";
    return;
  }}

  const PALETTE = ["#007AFF","#34C759","#FF9500","#AF52DE","#FF3B30","#5AC8FA","#5856D6","#FF2D55","#0B3A3B"];

  function hashColor(str) {{
    let h = 0;
    for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) >>> 0;
    return PALETTE[h % PALETTE.length];
  }}

  function setAccent(color) {{
    document.documentElement.style.setProperty("--accent", color);
    const m = /^#?([a-f\\d]{{2}})([a-f\\d]{{2}})([a-f\\d]{{2}})$/i.exec(color);
    if (!m) return;
    const r = parseInt(m[1], 16), g = parseInt(m[2], 16), b = parseInt(m[3], 16);
    document.documentElement.style.setProperty("--accent2", `rgba(${{r}},${{g}},${{b}},0.12)`);
  }}

  function fmtMeta() {{
    const ts = META.timestamp || "";
    const hash = META.hash || META.random_seed || "";
    const seed = META.seed || META.solver_seed || "";
    const ver = META.version || "";
    const line = [ts, hash ? `hash: ${{hash}}` : "", seed ? `seed: ${{seed}}` : "", ver ? `v${{ver}}` : ""]
      .filter(Boolean).join(" · ");
    footMeta.textContent = line || "—";
  }}

  function flatten(obj, prefix="") {{
    const out = [];
    if (obj && typeof obj === "object" && !Array.isArray(obj)) {{
      for (const k of Object.keys(obj)) {{
        const v = obj[k];
        const kp = prefix ? `${{prefix}} - ${{k}}` : String(k);
        if (v && typeof v === "object" && !Array.isArray(v)) out.push(...flatten(v, kp));
        else out.push({{ key: kp, value: stringifyValue(v) }});
      }}
      return out;
    }}
    out.push({{ key: prefix || "value", value: stringifyValue(obj) }});
    return out;
  }}

  function stringifyValue(v) {{
    if (v === null || v === undefined) return "";
    if (typeof v === "string") return v;
    if (typeof v === "number" || typeof v === "boolean") return String(v);
    try {{ return JSON.stringify(v); }} catch (e) {{ return String(v); }}
  }}

  function getActiveSection(sections) {{
    const saved = localStorage.getItem("nqx.activeSection");
    if (saved && sections.includes(saved)) return saved;
    return sections[0] || "";
  }}

  function setActiveSection(name) {{
    localStorage.setItem("nqx.activeSection", name);
    setAccent(hashColor(name));
  }}

  function markActive(name) {{
    for (const it of nav.querySelectorAll(".nqx-item")) {{
      it.classList.toggle("active", it.dataset.section === name);
    }}
  }}

  function renderSection(name) {{
    const rows = flatten(DATA[name] || {{}});
    titleEl.textContent = name || "—";
    subtitleEl.textContent = rows.length ? `${{rows.length}} entr${{rows.length===1 ? "y" : "ies"}}` : "No entries";

    const q = (rowSearch.value || "").trim().toLowerCase();
    const filtered = q ? rows.filter(r => (r.key + " " + r.value).toLowerCase().includes(q)) : rows;

    tbody.innerHTML = "";
    for (const r of filtered) {{
      const tr = document.createElement("tr");
      tr.className = "nqx-row";

      const tdK = document.createElement("td");
      tdK.className = "nqx-key";
      tdK.textContent = r.key;

      const tdV = document.createElement("td");
      tdV.className = "nqx-val";

      const inline = document.createElement("span");
      inline.className = "nqx-inline";
      inline.textContent = r.value;
      tdV.appendChild(inline);

      const longish = r.value.length > 140 || r.value.includes("\\n") || r.value.includes("\\r");
      if (longish) {{
        const btn = document.createElement("button");
        btn.className = "nqx-expand";
        btn.type = "button";
        btn.textContent = "Show";

        const pre = document.createElement("div");
        pre.className = "nqx-pre";
        let expandedText = r.value;
        try {{
          const parsed = JSON.parse(r.value);
          expandedText = JSON.stringify(parsed, null, 2);
        }} catch (e) {{}}
        pre.textContent = expandedText;

        btn.addEventListener("click", () => {{
          const showing = pre.classList.toggle("show");
          btn.textContent = showing ? "Hide" : "Show";
        }});

        tdV.appendChild(btn);
        tdV.appendChild(pre);
      }}

      tdV.title = r.value;

      tr.appendChild(tdK);
      tr.appendChild(tdV);
      tbody.appendChild(tr);
    }}
  }}

  function buildNav(sections) {{
    nav.innerHTML = "";
    for (const name of sections) {{
      const rows = flatten(DATA[name] || {{}});
      const item = document.createElement("div");
      item.className = "nqx-item";
      item.dataset.section = name;

      const left = document.createElement("div");
      left.className = "nqx-item-left";

      const nm = document.createElement("div");
      nm.className = "nqx-item-name";
      nm.textContent = name;

      const hint = document.createElement("div");
      hint.className = "nqx-item-hint";
      hint.textContent = rows.length ? `${{rows.length}} entr${{rows.length===1 ? "y" : "ies"}}` : "empty";

      left.appendChild(nm);
      left.appendChild(hint);

      const badge = document.createElement("div");
      badge.className = "nqx-badge";
      badge.textContent = String(rows.length);

      item.appendChild(left);
      item.appendChild(badge);

      item.addEventListener("click", () => {{
        setActiveSection(name);
        renderSection(name);
        markActive(name);
      }});

      nav.appendChild(item);
    }}
  }}

  function filterSections() {{
    const q = (sectionSearch.value || "").trim().toLowerCase();
    for (const it of nav.querySelectorAll(".nqx-item")) {{
      const name = (it.dataset.section || "").toLowerCase();
      it.style.display = name.includes(q) ? "" : "none";
    }}
  }}

  fmtMeta();
  const sections = Object.keys(DATA || {{}});
  if (!sections.length) {{
    // nothing to show -> fallback
    const fb = document.getElementById("nqx-fallback");
    if (fb) fb.style.display = "block";
    const app = document.getElementById("nqx-app");
    if (app) app.style.display = "none";
    return;
  }}
  const initial = getActiveSection(sections);
  setActiveSection(initial);
  buildNav(sections);
  markActive(initial);
  renderSection(initial);

  sectionSearch.addEventListener("input", filterSections);
  rowSearch.addEventListener("input", () => {{
    const active = localStorage.getItem("nqx.activeSection") || sections[0] || "";
    renderSection(active);
  }});
}})();
</script>
"""


class Logger(AbstractLogger):
    """
    Structured logger for neuraLQX solver executions.

    This class collects, organizes, and manages simulation metadata, configuration parameters, and
    numerical results produced during solver runs. Logged data is stored in a hierarchical
    dictionary structure with predefined categories that can be extended dynamically at runtime.

    The logger supports:
      - Incremental logging of scalar and structured values
      - Runtime extension of log fields
      - Recursive sanitisation of unset entries
      - Human-readable display using Rich panels
      - Persistent export to a signed HTML file

    The logger is distributed-aware: all logging, display, and file output operations are
    performed exclusively on process 0 to avoid duplicated output.
    """

    def __init__(
        self,
        random_seed: Any,
        solver_seed: Any,
    ):
        """
        Initialise a new Logger instance with a fixed random seed.

        This constructor sets up the default log schema, initialises internal state, and records
        metadata required for reproducibility and version tracking.

        :param random_seed: The random seed associated with the solver run, used for identification
          and embedded in log outputs.
        """

        self._p = NQXPrinter()
        self._random_seed = random_seed
        self._version = __version__
        self._solver_seed = solver_seed

        self._LOG_ITEMS = {
            "Optimization Results": {
                "Exact diagonalization result": None,
                "Network result": None,
                "Accuracy": None,
                "Average min<C> over last 100 iterations (stddev)": None,
                "Average R_Hat over last 100 iterations (stddev)": None,
            },
            "Network Configs": {
                "Network type": None,
                "Number of network parameters": None,
            },
            "Optimizer Configs": {
                "Optimizer type": None,
                "Learning rate": None,
                "Number of iterations": None,
                "Diagonal shift": None,
            },
            "Sampler Configs": {
                "Sampler type": None,
                "Number of samples": None,
                "Number of chains": None,
                "Number of chains per process": None,
                "Number of sweeps": None,
                "Machine power": None,
                "Reset chains": None,
            },
            "Physical System": {
                "Gravity model": None,
                "Gauge group": None,
                "Edges/Graph": None,
                "Number of vertices": None,
                "Number of edges": None,
                "Number of vertices (dual graph)": None,
                "Minimal loops": None,
                "Number of minimal loops": None,
                "Minimal loops (dual graph)": None,
                "Number of minimal loops (dual graph)": None,
                "Allowed degrees of freedom": None,
                "Hilbert space dimension": None,
                "Finite Hilbert space": None,
                "Indexable Hilbert space": None,
                "Gauge invariant Hilbert space": None,
            },
            "Distributed Runtime": {
                "Backend": None,
                "Distributed enabled": None,
                "Process index": None,
                "Process count": None,
                "Local process index": None,
                "Number of hosts": None,
                "Processes on this host": None,
                "CPUs per process": None,
                "Total number of CPUs": None,
                "Available GPUs": None,
                "JAX version": None,
                "Python implementation": None,
                "Python version": None,
            },
        }

        self._ORIGINAL_LOG_ITEMS = copy.deepcopy(self._LOG_ITEMS)

    def add_field(
        self,
        parent_key: str,
        new_field_key: str,
    ) -> None:
        """
        Add a new logging field under an existing parent log category.

        This method allows the log schema to be extended dynamically at runtime by inserting a new
        field initialised with a null value. The field becomes available for subsequent logging
        operations.

        :param parent_key: The name of the parent log category to extend.
        :param new_field_key: The name of the new field to add under the parent category.

        :raises KeyError: If the specified parent category does not exist in the log.
        """

        # skip on all workers
        if not _dist.is_global_master():
            return

        if parent_key in self._LOG_ITEMS:
            self._LOG_ITEMS[parent_key][new_field_key] = None
        else:
            raise KeyError(f"The `{parent_key}` entry in the logger was not found.")

    def log(
        self,
        field_key: str | list[str],
        value: Any | list[Any],
    ) -> None:
        """
        Record one or more values in the log.

        This method assigns values to existing log fields. It supports both single key-value logging
        and batch logging by providing lists of field keys and corresponding values.

        :param field_key: The field name(s) to which values should be logged.
        :param value: The value(s) to associate with the specified field(s).

        :raises KeyError: If a specified field does not exist in the log.
        :raises ValueError: If lists are provided with mismatched lengths or incompatible types.
        """

        # skip on workers
        if not _dist.is_global_master():
            return

        if isinstance(field_key, list) and isinstance(value, list):
            # the case where both are lists
            if len(field_key) != len(value):
                raise ValueError(
                    "You have entered an unequal amount of attributes and values to be logged."
                )
            for key, val in zip(field_key, value):
                self._log(key, val)
        elif isinstance(field_key, str) and not isinstance(
            value, list
        ):  # dev: check for any iterable?
            # the case where both are single values
            self._log(field_key, value)
        else:
            raise ValueError(
                "Both fieldKeys and values must either be lists of the same length or single items."
            )

    def _log(
        self,
        field_key: str,
        value: Any,
    ) -> None:
        """
        Internal logging primitive for assigning a value to a single log field.

        This method searches all parent categories for the specified field key and updates its value
        once found.

        :param field_key: The name of the field to update.
        :param value: The value to assign to the field.

        :raises KeyError: If the field key does not exist in any log category.
        """

        found_flag = False

        for parent_dict in self._LOG_ITEMS.values():
            if field_key in parent_dict:
                parent_dict[field_key] = value
                found_flag = True
                break

        if not found_flag:
            raise KeyError(
                f"The Log key `{field_key}` was not found in any log. "
                f"Try using addField(`<parentLogKey>`, `{field_key}`) to first add the log key to "
                f"your desired"
                f"`<parentLogKey>` log before setting a value to it."
            )

    def get_log(self) -> dict:
        """
        Retrieve the complete structured log.

        :returns: The full hierarchical log dictionary, including all parent categories and their
          associated fields.
        """
        return self._LOG_ITEMS

    def get_value_of(self, key: str) -> Any:
        """
        Retrieve the value associated with a given log key.

        If the key corresponds to a parent category, the entire sub-dictionary is returned. If the
        key corresponds to a leaf field, only its value is returned.

        :param key: The name of a log field or parent category.

        :returns: The requested value or sub-log.

        :raises KeyError: If the key does not exist anywhere in the log.
        """

        if key in self._LOG_ITEMS:
            return self._LOG_ITEMS[key]

        for parent_dict in self._LOG_ITEMS.values():
            if key in parent_dict:
                return parent_dict[key]

        raise KeyError(f"The field `{key}` was not found in the logger")

    def write_log_to_file(self, path: str) -> None:
        """
        Export the current log to a signed HTML file on disk.

        The log is sanitized before export, rendered into a human-readable HTML document,
        cryptographically signed, and written to the specified directory. The output file name
        includes the random seed and a timestamp.

        This operation is performed only on the global master process.

        :param path: Directory where the HTML log file should be written.
        """

        if _dist.is_global_master():

            auth = Authenticator()

            prk, puk, pem = auth.generate_key_pair()  # pylint: disable=W0612

            # sanitize the log first
            self._LOG_ITEMS = self.sanitize(self._LOG_ITEMS)

            data = self._LOG_ITEMS

            meta = {
                "timestamp": datetime.now().strftime("%H:%M:%S - %d.%m.%Y"),
                "hash": str(self._random_seed),
                "seed": str(self._solver_seed),
                "version": str(self._version),
            }

            html_content = f"""<!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>neuraLQX Output Log</title>
            </head>
            <body>
              {dict_to_html(data, meta=meta)}

              <div id="hidden-data" style="display:none;">
                <p id="sgn">{auth.sign_data(prk, data.__str__())}</p>
                <p id="rd">{data.__str__()}</p>
                <p id="pk">{pem}</p>
              </div>
            </body>
            </html>
            """

            datetime_str = datetime.now().strftime("%d%m%Y_%H%M%S")
            filename = os.path.join(
                path,
                f"{self._random_seed}_ModelData_{datetime_str}.html",
            )

            with open(filename, "w", encoding="utf-8") as html_file:
                html_file.write(html_content)

            self._p.print("Data exported to disk.")

        # wait for process-0 to finish I/O
        _dist.barrier()

    def display_log(self) -> None:
        """
        Display the current log in a formatted Rich panel.

        The log is sanitized prior to display and rendered as a nested table structure for
        interactive inspection in the console.

        This operation is performed only on the global master process.
        """

        # silent on worker ranks
        if not _dist.is_global_master():
            return

        # sanitize the log first
        self._LOG_ITEMS = self.sanitize(self._LOG_ITEMS)

        output_panel = self._print_config_parameters(self._LOG_ITEMS)
        console.print(output_panel)

    def sanitize(self, log_dict: dict) -> dict:
        """
        Recursively remove unset entries from a log dictionary.

        This method traverses the provided dictionary and removes any key-value pairs where the
        value is ``None``. Nested dictionaries are cleaned recursively and removed entirely if
        empty.

        :param log_dict: A log dictionary to sanitize.

        :returns: A cleaned version of the input dictionary containing only populated entries.
        """

        clean_log = {}

        for key, value in log_dict.items():
            if isinstance(value, dict):
                # recursively clean nested dictionaries
                nested_dict = self.sanitize(value)
                if nested_dict:
                    clean_log[key] = nested_dict
            elif value is not None:
                clean_log[key] = value

        return clean_log

    def erase_parent_content(self, parent: str) -> None:
        """
        Clear all logged values under a specified parent category.

        Fields that were part of the original log schema are reset to ``None``, while fields added
        dynamically at runtime are removed entirely.

        :param parent: The name of the parent log category to clear.

        :raises KeyError: If the specified parent category does not exist.
        """

        # skip on all workers
        if not _dist.is_global_master():
            return

        if parent not in self._LOG_ITEMS:
            raise KeyError(
                f"The requested parent key {parent} does not exist in the current log."
            )

        # get the set of keys that originally belonged to this parent
        original_keys = set(self._ORIGINAL_LOG_ITEMS.get(parent, {}).keys())

        # iterate over a copy of the keys to avoid modifying the dict during iteration
        for key in list(self._LOG_ITEMS[parent].keys()):
            if key in original_keys:
                self._LOG_ITEMS[parent][key] = None
            else:
                del self._LOG_ITEMS[parent][key]

    def _print_config_parameters(
        self,
        config_params,
        title: str = "neuraLQX Output Log",
        indent: int = 0,
        is_child: bool = False,
    ) -> Panel:
        table = Table(box=box.SIMPLE, show_header=False)

        for key, value in config_params.items():
            if isinstance(value, dict):
                child_panel = self._print_config_parameters(
                    value,
                    title=key,
                    indent=indent + 1,
                    is_child=True,
                )
                table.add_row(f"[bold]{key}:[/bold]", child_panel)
            else:
                table.add_row(f"[bold]{key}:[/bold]", str(value))

        if not is_child:
            date_str = datetime.now().strftime("%H:%M:%S - %d.%m.%Y")
            title_with_date = f"[bold]{title}[/bold]\n[dim]{date_str} (hash:{self._random_seed}) (seed:{self._solver_seed}) v{self._version}[/dim]"
            return Panel(table, title=title_with_date, border_style="black")

        return Panel(table, border_style="black")
