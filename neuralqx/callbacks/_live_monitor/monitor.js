/* Copyright (c) 2026 The neuraLQX Authors. Apache-2.0. */
"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const canvas = $("chart"), wrap = $("chart-wrap"), ctx = canvas.getContext("2d");
  const overlay = document.createElement("canvas");
  overlay.style.cssText = "position:absolute;inset:0;pointer-events:none";
  overlay.setAttribute("aria-hidden", "true");
  wrap.insertBefore(overlay, $("tooltip"));
  const hover = overlay.getContext("2d");
  const palette = ["#397fac", "#69905b", "#c08d4d", "#9a76af", "#bd626b", "#4e9b9a", "#737fb5"];
  const selected = new Set(["__objective__"]), colors = new Map([["__objective__", "#164f4d"]]);
  const metricNodes = new Map(), latest = new Map();
  let rows = [], cursor = 0, metadata = {}, paused = false, errors = true;
  let view = null, geometry = null, drag = null, size = {}, requestFrame = 0, timer = null;
  let lastDiagnostics = {}, plotSeries = [], fetching = false;
  let hoverRow = null, hoverPinned = false, renderedHoverRow = null;
  let selectionVersion = 0, renderedSelectionVersion = -1;
  const decimal = new Intl.NumberFormat("en-US", {maximumFractionDigits:3});
  const precise = new Intl.NumberFormat("en-US", {maximumFractionDigits:5});

  function format(value) {
    if (value == null || !Number.isFinite(value)) return "—";
    const abs = Math.abs(value);
    if (abs && (abs < 0.001 || abs >= 1e6)) return value.toExponential(3);
    return (abs < 1 ? precise : decimal).format(value);
  }
  function tickStep(span, count) {
    const raw = span / count, power = 10 ** Math.floor(Math.log10(raw));
    return ([1, 2, 2.5, 5, 10].find(value => value * power >= raw) || 10) * power;
  }
  const label = name => name === "__objective__" ? metadata.objective_label || "Objective" : name;
  function color(name) {
    if (!colors.has(name)) colors.set(name, palette[(colors.size - 1) % palette.length]);
    return colors.get(name);
  }
  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text != null) element.textContent = text;
    return element;
  }
  function elapsed(seconds) {
    const value = Math.max(0, Math.floor(seconds || 0));
    const parts = [Math.floor(value / 60) % 60, value % 60];
    if (value >= 3600) parts.unshift(Math.floor(value / 3600));
    return parts.map(part => String(part).padStart(2, "0")).join(":");
  }
  function renderSummary() {
    $("objective-label").textContent = metadata.objective_label || "Objective";
    const stat = latest.get("__objective__");
    $("objective-value").textContent = format(stat?.[0]);
    $("objective-error").textContent = stat?.[1] != null ? `± ${format(stat[1])}` : "";
    $("step").textContent = metadata.step == null ? "—" : format(metadata.step);
    $("elapsed").textContent = elapsed(metadata.elapsed);
    $("acceptance").textContent = lastDiagnostics.acceptance == null ? "—" : `${(lastDiagnostics.acceptance * 100).toFixed(1)}%`;
    $("variance").textContent = format(lastDiagnostics.variance);
    $("rhat").textContent = lastDiagnostics.r_hat == null ? "—" : lastDiagnostics.r_hat.toFixed(3);
    $("point-count").textContent = `${rows.length.toLocaleString()} recorded points`;
    const status = metadata.status || "connecting";
    $("status").dataset.status = status;
    $("status-text").textContent = ({running:"Running", completed:"Completed", stopped:"Stopped", stopping:"Stopping", failed:"Run failed", offline:"Reconnecting", connecting:"Connecting"})[status] || status;
    $("stop").disabled = status !== "running";
    $("stop").textContent = status === "stopping" ? "Stop requested" : "Stop simulation";
    $("stop-help").textContent = ["completed", "stopped", "failed"].includes(status) ? "Run finished. History is ready to export." : "Stops after the current step.";
    $("connection-note").textContent = paused ? "View paused · Your simulation continues" : metadata.message || (status === "offline" ? "Waiting for the local monitor to reconnect…" : `${status === "running" ? "Live updates" : "Recorded history"} · ${metadata.max_points?.toLocaleString() || "10,000"} point history limit${metadata.dropped ? ` · ${metadata.dropped} updates skipped` : ""}`);
    renderMetrics();
  }
  function renderMetrics() {
    const names = ["__objective__", ...(metadata.metrics || []).filter(name => name !== "__objective__")];
    for (const name of names) {
      let item = metricNodes.get(name);
      if (!item) {
        const container = node("label", "metric");
        const input = document.createElement("input");
        input.type = "checkbox";
        const info = node("span", "metric-info"), title = node("span", "metric-label");
        const swatch = node("span", "swatch");
        swatch.style.background = color(name);
        const text = node("span", "", label(name));
        title.append(swatch, text);
        const value = node("span", "metric-value", "—");
        info.append(title, value);
        container.append(input, info);
        $("metrics").append(container);
        input.addEventListener("change", () => {
          if (input.checked) selected.add(name); else selected.delete(name);
          selectionVersion++;
          renderMetrics(); scheduleDraw();
        });
        item = {input, value, text};
        metricNodes.set(name, item);
      }
      item.input.checked = selected.has(name);
      item.text.textContent = label(name);
      const stat = latest.get(name);
      item.value.textContent = stat ? `${format(stat[0])}${stat[1] == null ? "" : ` ± ${format(stat[1])}`}` : "—";
    }
    $("metrics-empty").hidden = names.length > 1;
    $("select-all").disabled = !names.length;
    $("select-all").textContent = names.every(name => selected.has(name)) ? "Deselect all" : "Select all";
  }
  function scheduleDraw() {
    if (!requestFrame) requestFrame = requestAnimationFrame(() => { requestFrame = 0; draw(); });
  }
  function resize() {
    const rect = wrap.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
    size = {width: rect.width, height: rect.height, ratio};
    for (const item of [canvas, overlay]) {
      item.width = Math.round(rect.width * ratio); item.height = Math.round(rect.height * ratio);
      item.getContext("2d").setTransform(ratio, 0, 0, ratio, 0, 0);
    }
    scheduleDraw();
  }
  function draw() {
    const {width: w, height: h} = size;
    if (!w) return;
    ctx.clearRect(0, 0, w, h); hover.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
    const left = w < 450 ? 52 : 65, right = w - 16, top = 26, bottom = h - 44;
    let xmin = rows.length ? rows[0].step : 0, xmax = rows.length ? rows[rows.length - 1].step : 10;
    if (xmin === xmax) { xmin -= 0.5; xmax += 0.5; }
    if (view) [xmin, xmax] = view;
    const logarithmic = $("scale").value === "log";
    const transform = value => logarithmic ? (value > 0 ? Math.log10(value) : NaN) : value;
    let ymin = Infinity, ymax = -Infinity;
    plotSeries = [];
    for (const name of selected) {
      const points = [];
      for (const row of rows) {
        const stat = row.metrics[name];
        if (!stat || stat[0] == null || row.step < xmin || row.step > xmax) continue;
        const y = transform(stat[0]);
        if (!Number.isFinite(y)) continue;
        let low = errors && stat[1] != null ? transform(stat[0] - Math.abs(stat[1])) : y;
        let high = errors && stat[1] != null ? transform(stat[0] + Math.abs(stat[1])) : y;
        if (!Number.isFinite(low)) low = y;
        if (!Number.isFinite(high)) high = y;
        ymin = Math.min(ymin, low); ymax = Math.max(ymax, high);
        points.push({x: row.step, y, low, high, stat});
      }
      if (points.length) plotSeries.push({name, points});
    }
    const hasData = Number.isFinite(ymin);
    $("empty").hidden = hasData;
    if (!hasData) {
      ymin = 0; ymax = 1;
      $("empty").querySelector("strong").textContent = !selected.size ? "Choose a metric to plot" : rows.length ? (logarithmic ? "No positive values in this view" : "No values in this range") : "Waiting for the first step";
      $("empty").querySelector("span:last-child").textContent = !selected.size ? "Select one or more metrics from the panel." : rows.length ? "Reset the view or select a different scale." : "The chart will update as the run progresses.";
    }
    const positive = ymin >= 0;
    const pad = (ymax - ymin || Math.abs(ymax) * 0.1 || 1) * 0.1;
    ymin -= pad; ymax += pad;
    const yStep = tickStep(ymax - ymin, 5);
    if (!logarithmic) {
      ymin = Math.floor(ymin / yStep) * yStep;
      ymax = Math.ceil(ymax / yStep) * yStep;
      if (positive) ymin = Math.max(0, ymin);
    }
    const xpx = x => left + (x - xmin) / (xmax - xmin) * (right - left);
    const ypx = y => bottom - (y - ymin) / (ymax - ymin) * (bottom - top);
    geometry = {left, right, top, bottom, xmin, xmax, xpx, ypx};
    ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.lineWidth = 1; ctx.textBaseline = "middle";
    const yTicks = logarithmic ? Array.from({length:5}, (_,i) => ymin + (ymax - ymin) * i / 4) : Array.from({length:Math.round((ymax-ymin)/yStep)+1}, (_,i) => ymin+i*yStep);
    for (const y of yTicks) {
      const py = ypx(y);
      ctx.strokeStyle = "#edf0ee"; ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
      ctx.fillStyle = "#8a9690"; ctx.textAlign = "right";
      ctx.fillText(format(logarithmic ? 10 ** y : y), left - 10, py);
    }
    const ticks = w < 450 ? 3 : 5;
    const xStep = Math.max(1, tickStep(xmax - xmin, ticks));
    for (let x = Math.ceil(xmin / xStep) * xStep; x <= xmax; x += xStep) {
      ctx.textAlign = "center";
      ctx.fillText(format(x), xpx(x), bottom + 17);
    }
    ctx.textAlign = "left"; ctx.fillStyle = "#7c8982";
    ctx.fillText(metadata.y_label || "Expectation", left, 10);
    ctx.textAlign = "center"; ctx.fillText(metadata.x_label || "Iteration", (left + right) / 2, h - 5);
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, right - left, bottom - top); ctx.clip();
    for (const {name, points} of plotSeries) {
      // Keep extrema in each pixel column when histories outgrow the screen.
      let visible = points;
      if (points.length > (right - left) * 2) {
        const buckets = new Map();
        for (let i = 0; i < points.length; i++) {
          const p = points[i], pixel = Math.floor(xpx(p.x));
          let bucket = buckets.get(pixel);
          if (!bucket) { bucket = {first:i, min:i, max:i, last:i}; buckets.set(pixel, bucket); }
          if (p.y < points[bucket.min].y) bucket.min = i;
          if (p.y > points[bucket.max].y) bucket.max = i;
          bucket.last = i;
        }
        visible = [...buckets.values()].flatMap(b => [...new Set([b.first, b.min, b.max, b.last])].sort((a,b) => a-b).map(i => points[i]));
      }
      ctx.strokeStyle = color(name); ctx.fillStyle = color(name);
      if (errors) {
        ctx.globalAlpha = 0.25; ctx.lineWidth = 1; ctx.beginPath();
        for (const p of visible) {
          const x = xpx(p.x), a = ypx(p.low), b = ypx(p.high);
          ctx.moveTo(x, a); ctx.lineTo(x, b);
          if (visible.length < 120) { ctx.moveTo(x - 2, a); ctx.lineTo(x + 2, a); ctx.moveTo(x - 2, b); ctx.lineTo(x + 2, b); }
        }
        ctx.stroke(); ctx.globalAlpha = 1;
      }
      ctx.lineWidth = name === "__objective__" ? 1.9 : 1.5;
      ctx.beginPath();
      visible.forEach((p, i) => { if (i) ctx.lineTo(xpx(p.x), ypx(p.y)); else ctx.moveTo(xpx(p.x), ypx(p.y)); });
      ctx.stroke();
      if (visible.length < 65) for (const p of visible) { ctx.beginPath(); ctx.arc(xpx(p.x), ypx(p.y), 2.2, 0, 2 * Math.PI); ctx.fill(); }
    }
    ctx.restore();
    $("view-badge").hidden = !view;
    renderHover();
  }
  function clearHover() {
    hoverRow = null; hoverPinned = false;
    $("tooltip").hidden = true;
    hover.clearRect(0, 0, size.width, size.height);
  }
  function renderHover() {
    hover.clearRect(0, 0, size.width, size.height);
    if (!hoverRow || !geometry || !selected.size) { clearHover(); return; }
    const {top, bottom, xmin, xmax, xpx} = geometry;
    if (hoverRow.step >= xmin && hoverRow.step <= xmax) {
      const x = xpx(hoverRow.step);
      hover.strokeStyle = "#aabdb2"; hover.setLineDash([3, 4]);
      hover.beginPath(); hover.moveTo(x, top); hover.lineTo(x, bottom); hover.stroke();
    }
    // Reuse the card while the pointer stays on the same sample. Live redraws
    // must also preserve its scroll position while someone is reading it.
    if (renderedHoverRow !== hoverRow || renderedSelectionVersion !== selectionVersion) {
      const values = $("tip-values"), fragment = document.createDocumentFragment();
      let count = 0;
      for (const name of selected) {
        const stat = hoverRow.metrics[name]; if (!stat) continue;
        const row = node("div", "tip-row"), swatch = node("span", "swatch");
        swatch.style.background = color(name);
        const value = node("span", "tip-stat");
        value.append(node("strong", "", format(stat[0])));
        if (stat[1] != null) value.append(node("small", "", `± ${format(stat[1])}`));
        row.append(swatch, node("span", "tip-name", label(name)), value);
        fragment.append(row); count++;
      }
      const scrollTop = renderedHoverRow === hoverRow ? values.scrollTop : 0;
      values.replaceChildren(fragment);
      values.scrollTop = scrollTop;
      $("tip-step").textContent = `${metadata.x_label || "Iteration"} ${hoverRow.step}`;
      $("tip-count").textContent = `${count} ${count === 1 ? "metric" : "metrics"}`;
      renderedHoverRow = hoverRow; renderedSelectionVersion = selectionVersion;
    }
    $("tooltip").dataset.pinned = String(hoverPinned);
    $("tooltip").style.width = !hoverPinned && size.width >= 480 ? `${Math.min(300, size.width / 2 - 16)}px` : "300px";
    $("tip-pin").hidden = !hoverPinned;
    $("tip-close").hidden = !hoverPinned;
    $("tip-help-title").textContent = hoverPinned ? "Card pinned" : "Click the chart to pin this card";
    $("tip-help-detail").textContent = hoverPinned ? "Scroll through the values, or Unpin to follow your pointer." : "Then move here to scroll through the values.";
    $("tooltip").hidden = false;
  }
  function showHover(event, pin = false) {
    if (!geometry || !rows.length || !selected.size || (hoverPinned && !pin)) return;
    const rect = canvas.getBoundingClientRect(), px = event.clientX - rect.left, py = event.clientY - rect.top;
    const {left, right, top, bottom, xmin, xmax} = geometry;
    if (px < left || px > right || py < top || py > bottom) return;
    const step = xmin + (px - left) / (right - left) * (xmax - xmin);
    let closest = rows[0];
    for (const row of rows) if (Math.abs(row.step - step) < Math.abs(closest.step - step)) closest = row;
    // Unpinned cards let pointer events reach the chart underneath them.
    // Moving across the chart can therefore always reveal the covered area.
    $("tooltip").dataset.side = size.width >= 480
      ? (px > size.width / 2 ? "left" : "right")
      : (py > size.height / 2 ? "top" : "bottom");
    if (closest === hoverRow && !pin) return;
    hoverRow = closest; hoverPinned = pin; renderHover();
  }
  function resetView() { clearHover(); view = null; scheduleDraw(); }
  canvas.addEventListener("wheel", event => {
    if (!geometry || !rows.length) return;
    event.preventDefault();
    const rect = canvas.getBoundingClientRect(), g = geometry;
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - g.left) / (g.right - g.left)));
    const span = Math.max(.5, Math.min(Math.max(1, rows[rows.length - 1].step - rows[0].step) * 4, (g.xmax - g.xmin) * Math.exp(Math.max(-1, Math.min(1, event.deltaY * .002)))));
    const center = g.xmin + fraction * (g.xmax - g.xmin);
    view = [center - fraction * span, center + (1 - fraction) * span]; scheduleDraw();
  }, {passive:false});
  canvas.addEventListener("pointerdown", event => {
    if (!geometry || !rows.length) return;
    drag = {x:event.clientX, y:event.clientY, moved:false, range:[geometry.xmin, geometry.xmax]}; canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", event => {
    if (drag && geometry) {
      drag.moved ||= Math.hypot(event.clientX - drag.x, event.clientY - drag.y) > 4;
      if (!drag.moved) return;
      const delta = (event.clientX - drag.x) / (geometry.right - geometry.left) * (drag.range[1] - drag.range[0]);
      view = drag.range.map(value => value - delta); scheduleDraw();
    } else showHover(event);
  });
  canvas.addEventListener("pointerup", event => {
    if (drag && !drag.moved) showHover(event, true);
    drag = null;
  });
  canvas.addEventListener("pointercancel", () => { drag = null; });
  wrap.addEventListener("pointerleave", () => { if (!hoverPinned) clearHover(); });
  canvas.addEventListener("dblclick", resetView);
  canvas.addEventListener("keydown", event => {
    if (event.key === "Escape") resetView();
    if ((event.key === "ArrowLeft" || event.key === "ArrowRight") && rows.length && selected.size) {
      event.preventDefault();
      const current = hoverRow ? rows.findIndex(row => row === hoverRow) : rows.length - 1;
      const index = Math.max(0, Math.min(rows.length - 1, current + (event.key === "ArrowLeft" ? -1 : 1)));
      hoverRow = rows[index]; hoverPinned = true; renderHover();
    }
  });
  $("tip-pin").addEventListener("click", () => { hoverPinned = !hoverPinned; renderHover(); });
  $("tip-close").addEventListener("click", clearHover);
  $("tooltip").addEventListener("keydown", event => { if (event.key === "Escape") { clearHover(); canvas.focus(); } });
  $("reset").addEventListener("click", resetView);
  $("scale").addEventListener("change", scheduleDraw);
  $("errors").addEventListener("click", () => { errors = !errors; $("errors").setAttribute("aria-pressed", String(errors)); scheduleDraw(); });
  $("pause").addEventListener("click", () => {
    paused = !paused; $("pause").textContent = paused ? "Resume view" : "Pause view";
    $("pause").setAttribute("aria-pressed", String(paused)); renderSummary(); if (!paused) poll();
  });
  $("select-all").addEventListener("click", () => {
    const names = ["__objective__", ...(metadata.metrics || []).filter(name => name !== "__objective__")];
    const all = names.every(name => selected.has(name));
    for (const name of names) { if (all) selected.delete(name); else selected.add(name); }
    selectionVersion++;
    renderMetrics(); scheduleDraw();
  });
  $("save-image").addEventListener("click", () => {
    const link = node("a"); link.download = "neuralqx-convergence.png";
    link.href = canvas.toDataURL("image/png"); link.click();
  });
  $("stop").addEventListener("click", async () => {
    $("stop").disabled = true;
    try {
      const response = await fetch("/api/stop", {method:"POST", headers:{"X-Monitor-Token":metadata.token}});
      if (!response.ok) throw new Error("Stop request failed");
      metadata.status = "stopping"; renderSummary();
    } catch { $("stop").disabled = false; $("stop-help").textContent = "Could not send the stop request. Try again."; }
  });
  async function poll() {
    if (fetching) return;
    clearTimeout(timer);
    if (paused || document.hidden) { timer = setTimeout(poll, 2000); return; }
    fetching = true;
    try {
      const response = await fetch(`/api/updates?after=${cursor}`, {signal:AbortSignal.timeout(5000)});
      if (!response.ok) throw new Error("Monitor unavailable");
      metadata = await response.json();
      if (metadata.reset) { rows = []; latest.clear(); }
      rows.push(...metadata.rows); rows = rows.slice(-metadata.max_points); cursor = metadata.cursor;
      for (const row of metadata.rows) {
        for (const [name, stat] of Object.entries(row.metrics)) latest.set(name, stat);
        lastDiagnostics = row.diagnostics;
      }
      renderSummary();
      if (metadata.rows.length || !geometry) scheduleDraw();
    } catch { metadata.status = "offline"; renderSummary(); }
    finally {
      fetching = false;
      timer = setTimeout(poll, metadata.status === "running" || metadata.status === "stopping" ? metadata.interval || 500 : 3000);
    }
  }
  document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
  new ResizeObserver(resize).observe(wrap);
  poll();
})();
