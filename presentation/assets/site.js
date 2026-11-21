/* Charts of the static presentation website, without external libraries
   (so the site also works offline in the lecture hall).

   Every story page embeds its numbers as JSON in <script id="story-data">
   (written by presentation/build_site.py). This script draws:
     - the data as "value over position" (scatter plot),
     - the mean comparisons per search (bars on a logarithmic scale),
     - the "what if" cost curves T(q) = preparation + q * time per query,
       with a slider for q and the recommendation of the rule book for every q. */
(function () {
  "use strict";
  const SVG = "http://www.w3.org/2000/svg";
  const de = new Intl.NumberFormat("de-DE", { maximumFractionDigits: 1 });
  const deInt = new Intl.NumberFormat("de-DE", { maximumFractionDigits: 0 });

  function el(name, attrs, parent) {
    const node = document.createElementNS(SVG, name);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }

  function text(parent, x, y, content, attrs) {
    const node = el("text", Object.assign({ x, y }, attrs || {}), parent);
    node.textContent = content;
    return node;
  }

  /** 1.500 -> "1,5 k", 2.000.000 -> "2 Mio." (short axis labels). */
  function shortNumber(v) {
    if (Math.abs(v) >= 1e6) return de.format(v / 1e6) + " Mio.";
    if (Math.abs(v) >= 1e3) return de.format(v / 1e3) + " Tsd.";
    return de.format(v);
  }

  /** Nanoseconds in a readable unit, German decimals. */
  function formatTime(ns) {
    if (ns >= 1e9) return de.format(ns / 1e9) + " s";
    if (ns >= 1e6) return de.format(ns / 1e6) + " ms";
    if (ns >= 1e3) return de.format(ns / 1e3) + " µs";
    return de.format(ns) + " ns";
  }

  function niceTicks(max, count) {
    const raw = max / count;
    const power = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(f => f * power).find(s => s >= raw) || raw;
    const ticks = [];
    for (let t = 0; t <= max + 1e-9; t += step) ticks.push(t);
    return ticks;
  }

  // --- Value over position ------------------------------------------------------------
  function scatter(container, points) {
    const W = 760, H = 320, m = { l: 70, r: 16, t: 12, b: 44 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Wert über Position" }, container);
    const maxX = Math.max(...points.map(p => p[0]), 1), maxY = Math.max(...points.map(p => p[1]), 1);
    const sx = x => m.l + (x / maxX) * (W - m.l - m.r);
    const sy = y => H - m.b - (y / maxY) * (H - m.t - m.b);
    for (const t of niceTicks(maxY, 4)) {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: "grid-line" }, svg);
      text(svg, m.l - 8, sy(t) + 4, shortNumber(t), { "text-anchor": "end" });
    }
    for (const t of niceTicks(maxX, 5)) text(svg, sx(t), H - m.b + 18, shortNumber(t), { "text-anchor": "middle" });
    text(svg, (m.l + W - m.r) / 2, H - 6, "Position im Speicher", { "text-anchor": "middle" });
    text(svg, 14, (m.t + H - m.b) / 2, "Wert", { "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` });
    for (const [x, y] of points) el("circle", { cx: sx(x), cy: sy(y), r: 2.2, class: "point" }, svg);
  }

  // --- Comparisons per search (log bars) ----------------------------------------------
  function bars(container, counts) {
    const max = Math.max(...counts.map(c => c.value), 10);
    const width = v => Math.max(1.5, (Math.log10(Math.max(v, 1)) + 0.15) / (Math.log10(max) + 0.15) * 82); // percent of the track
    const sorted = counts.slice().sort((a, b) => a.value - b.value);
    for (const c of sorted) {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML =
        `<div class="bar-name">${c.name}</div>` +
        `<div class="bar-track"><div class="bar${c.recommended ? " recommended" : ""}" style="width:${width(c.value)}%"></div>` +
        `<span class="bar-value" style="left:${width(c.value)}%">${de.format(c.value)}</span></div>`;
      container.appendChild(row);
    }
  }

  // --- What if: cost curves with slider -----------------------------------------------
  function whatIf(container, data) {
    const W = 860, H = 380, m = { l: 70, r: 250, t: 14, b: 46 };
    const qMin = 1, qMax = 1e7;
    const totals = data.curves.flatMap(c => [c.prep + qMin * c.query, c.prep + qMax * c.query]);
    const yMin = Math.pow(10, Math.floor(Math.log10(Math.max(Math.min(...totals), 1))));
    const yMax = Math.pow(10, Math.ceil(Math.log10(Math.max(...totals))));
    const lx = q => m.l + (Math.log10(q) - Math.log10(qMin)) / (Math.log10(qMax) - Math.log10(qMin)) * (W - m.l - m.r);
    const ly = v => H - m.b - (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)) * (H - m.t - m.b);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Gesamtzeit über die Anzahl Anfragen" }, container.querySelector(".chart"));

    for (let e = Math.log10(yMin); e <= Math.log10(yMax); e++) {
      el("line", { x1: m.l, x2: W - m.r, y1: ly(10 ** e), y2: ly(10 ** e), class: "grid-line" }, svg);
      text(svg, m.l - 8, ly(10 ** e) + 4, formatTime(10 ** e), { "text-anchor": "end" });
    }
    for (let e = 0; e <= 7; e++) {
      el("line", { x1: lx(10 ** e), x2: lx(10 ** e), y1: m.t, y2: H - m.b, class: "grid-line" }, svg);
      if (e % 2 === 0) text(svg, lx(10 ** e), H - m.b + 18, deInt.format(10 ** e), { "text-anchor": "middle" });
    }
    text(svg, (m.l + W - m.r) / 2, H - 6, "Anzahl Anfragen q", { "text-anchor": "middle" });

    const qs = [];
    for (let i = 0; i <= 80; i++) qs.push(10 ** (i / 80 * 7));
    const lines = data.curves.map(c => {
      const d = qs.map((q, i) => `${i ? "L" : "M"}${lx(q).toFixed(1)},${ly(c.prep + q * c.query).toFixed(1)}`).join(" ");
      const path = el("path", { d, fill: "none", "stroke-width": 2.5, stroke: "var(--muted)" }, svg);
      const label = text(svg, W - m.r + 8, ly(c.prep + qMax * c.query) + 4, c.name, {});
      return { c, path, label };
    });
    const marker = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--orange)", "stroke-width": 2, "stroke-dasharray": "6 4" }, svg);
    const dots = data.curves.map(() => el("circle", { r: 5.5 }, svg));

    const slider = container.querySelector("input[type=range]");
    const steps = data.steps;
    slider.max = steps.length - 1;
    slider.value = steps.indexOf(data.default_q);

    function update() {
      const q = steps[Number(slider.value)];
      const decision = data.decisions[String(q)];
      container.querySelector("output").textContent = deInt.format(q) + " Anfragen";
      marker.setAttribute("x1", lx(q));
      marker.setAttribute("x2", lx(q));
      lines.forEach(({ c, path, label }, i) => {
        const chosen = c.name === decision.recommended;
        path.setAttribute("stroke", chosen ? "var(--blue)" : "var(--muted)");
        path.setAttribute("stroke-width", chosen ? 4.5 : 2.5);
        path.setAttribute("opacity", chosen ? 1 : 0.75);
        label.setAttribute("font-weight", chosen ? 700 : 400);
        label.setAttribute("style", chosen ? "fill: var(--blue)" : "");
        if (chosen) svg.appendChild(path); // draw the recommendation on top
        dots[i].setAttribute("cx", lx(q));
        dots[i].setAttribute("cy", ly(c.prep + q * c.query));
        dots[i].setAttribute("fill", chosen ? "var(--blue)" : "var(--muted)");
        svg.appendChild(dots[i]);
      });
      container.querySelector("[data-field=recommended]").textContent = decision.recommended;
      container.querySelector("[data-field=total]").textContent = formatTime(decision.total_ns);
      container.querySelector("[data-field=agreement]").textContent = decision.agreement;
    }
    slider.addEventListener("input", update);
    update();
  }

  // The race is a separate page in an iframe: fit its height to the content
  // (the estimate used for the iframe's height attribute leaves empty space).
  function fitRaceFrames() {
    document.querySelectorAll("iframe.race-frame").forEach(function (frame) {
      function fit() {
        try {
          frame.style.height = frame.contentDocument.body.scrollHeight + 16 + "px"; // the content, not the (at least frame-high) document
        } catch (e) { /* not same-origin (e.g. opened as file://): keep the estimated height */ }
      }
      frame.addEventListener("load", fit);
      if (frame.contentDocument && frame.contentDocument.readyState === "complete") fit();
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    fitRaceFrames();
    const source = document.getElementById("story-data");
    if (!source) return;
    const data = JSON.parse(source.textContent);
    scatter(document.getElementById("chart-data"), data.points);
    bars(document.getElementById("chart-counts"), data.counts);
    whatIf(document.getElementById("what-if"), data.what_if);
  });
})();
