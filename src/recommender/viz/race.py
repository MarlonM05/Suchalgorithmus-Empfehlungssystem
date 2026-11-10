"""Animated "race" of several search algorithms on the same array.

The idea follows the popular side-by-side animations of linear versus binary
search: every algorithm gets its own lane with the same array, and all lanes
advance one comparison per tick. A lane that found its answer stops and shows
after how many steps it finished, so the audience sees *why* one method is
faster, not just *that* it is.

The traces come from :mod:`recommender.viz.stepper`, so every lane follows
exactly the logic of the compiled algorithm. The exact operation counts of the
compiled counting variant are shown next to the lanes, because the step count
of a trace is only a didactic approximation of the comparison count (for
example, the interpolation guard needs one step without a key comparison).

The result is a self-contained HTML document (no external resources) that
Streamlit renders in an iframe via ``st.components.v1.html``. Playback runs in
the browser, so the Streamlit server is not involved while the animation runs.
"""

import html
import json
from dataclasses import dataclass

from ..algorithms import ALGORITHMS, as_search_array
from .stepper import TRACERS, Step, trace

# Lane colours follow the palette of the figures (plots.ALGORITHM_STYLE).
LANE_COLOURS = {
    "linear": "#2a78d6",
    "binary": "#eb6834",
    "jump": "#4a3aa7",
    "exponential": "#e87ba4",
    "fibonacci": "#008300",
    "interpolation": "#e34948",
}
DEFAULT_RACE = ("linear", "binary")
MAX_RACE_SIZE = 64


@dataclass(frozen=True)
class Lane:
    """One competitor of the race."""

    key: str
    name: str
    complexity: str
    steps: tuple[Step, ...]
    comparisons: int  # exact count of the compiled counting variant
    accesses: int
    result: int

    @property
    def step_count(self) -> int:
        return len(self.steps)


def build_lanes(values, key: int, algorithms=DEFAULT_RACE) -> list[Lane]:
    """Trace every algorithm on ``values`` and attach the exact operation counts.

    All algorithms except the linear search need sorted data; the caller is
    responsible for passing a sorted array when such an algorithm takes part.
    """
    arr = as_search_array(values)
    lanes = []
    for algorithm in algorithms:
        if algorithm not in TRACERS:
            raise ValueError(f"no step-by-step trace for {algorithm!r}")
        counted = ALGORITHMS[algorithm].find_counted(arr, int(key))
        lanes.append(
            Lane(
                key=algorithm,
                name=TRACERS[algorithm][0],
                complexity=ALGORITHMS[algorithm].complexity,
                steps=tuple(trace(algorithm, arr, int(key))),
                comparisons=counted.comparisons,
                accesses=counted.accesses,
                result=counted.index,
            )
        )
    return lanes


def race_height(lanes: list[Lane], n: int) -> int:
    """Pixel height of the iframe: controls plus one block per lane."""
    rows_per_lane = max(1, -(-n // 18))  # about 18 cells per row before wrapping
    return 100 + len(lanes) * (88 + 44 * rows_per_lane)


_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #0b0b0b; background: transparent; }
.race-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem; margin: 0 0 0.8rem 0; }
.race-controls button { font: inherit; font-size: 0.85rem; padding: 0.35rem 0.8rem; border-radius: 6px; border: 1px solid #c3c2b7; background: #fff; cursor: pointer; }
.race-controls button.primary { background: #2a78d6; border-color: #2a78d6; color: #fff; font-weight: 600; }
.race-controls button:disabled { opacity: 0.45; cursor: default; }
.race-controls label { font-size: 0.8rem; color: #52514e; display: flex; align-items: center; gap: 0.4rem; }
.race-controls .tick { margin-left: auto; font-size: 0.8rem; color: #52514e; }
.lane { border: 1px solid #e1e0d9; border-radius: 8px; padding: 0.55rem 0.7rem 0.5rem 0.7rem; margin-bottom: 0.6rem; background: #fff; border-left-width: 5px; }
.lane-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.6rem; margin-bottom: 0.35rem; }
.lane-head .name { font-weight: 600; font-size: 0.95rem; }
.lane-head .complexity { font-size: 0.78rem; color: #52514e; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.lane-head .counter { margin-left: auto; font-size: 0.82rem; color: #52514e; white-space: nowrap; }
.lane-head .counter b { color: #0b0b0b; font-size: 1rem; }
.lane-head .badge { font-size: 0.78rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 999px; background: #d7f0d7; color: #0b4d0b; display: none; }
.lane-head .badge.miss { background: #fde3d6; color: #7a2a0c; }
.lane.done .badge { display: inline-block; }
.cells { display: flex; flex-wrap: wrap; gap: 3px; margin: 0.2rem 0 0.35rem 0; }
.cell { width: 2.6rem; border: 1px solid #c3c2b7; border-radius: 4px; text-align: center; padding: 2px 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #fff; transition: background 0.15s, border-color 0.15s; }
.cell .idx { font-size: 0.6rem; color: #898781; }
.cell .val { font-size: 0.8rem; }
.cell.out { background: #f0efec; color: #a3a29c; border-color: #e1e0d9; }
.cell.range { background: #e3eefb; border-color: #2a78d6; }
.cell.probe { background: #fde3d6; border: 2px solid #eb6834; font-weight: 600; padding: 1px 0; }
.cell.found { background: #d7f0d7; border: 2px solid #008300; font-weight: 600; padding: 1px 0; }
.cell.mark { box-shadow: inset 0 0 0 2px #4a3aa7; }
.status { font-size: 0.8rem; color: #52514e; min-height: 1.2em; }
.legend { font-size: 0.75rem; color: #52514e; margin-top: 0.2rem; }
.legend i { display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 3px; vertical-align: -2px; margin: 0 0.3rem 0 0.8rem; }
"""

_JS = r"""
(function () {
  const data = __DATA__;
  const lanes = data.lanes;
  const values = data.values;
  const n = values.length;
  let tick = 0, timer = null;
  const maxSteps = Math.max.apply(null, lanes.map(l => l.steps.length));
  const speeds = [1400, 900, 600, 350, 180];

  function cellHtml(i) {
    return '<div class="cell" data-i="' + i + '"><div class="idx">' + i + '</div><div class="val">' + values[i] + '</div></div>';
  }
  function render() {
    const root = document.getElementById('race');
    root.innerHTML = lanes.map((lane, li) =>
      '<div class="lane" id="lane-' + li + '" style="border-left-color:' + lane.colour + '">' +
        '<div class="lane-head">' +
          '<span class="name">' + lane.name + '</span>' +
          '<span class="complexity">' + lane.complexity + '</span>' +
          '<span class="badge" id="badge-' + li + '"></span>' +
          '<span class="counter">Schritte: <b id="count-' + li + '">0</b> / ' + lane.steps.length + '</span>' +
        '</div>' +
        '<div class="cells" id="cells-' + li + '">' + values.map((_, i) => cellHtml(i)).join('') + '</div>' +
        '<div class="status" id="status-' + li + '">Start: gesucht wird ' + data.key + '.</div>' +
      '</div>'
    ).join('');
  }
  function paint() {
    let finished = 0;
    lanes.forEach((lane, li) => {
      const pos = Math.min(tick, lane.steps.length);
      const step = pos > 0 ? lane.steps[pos - 1] : null;
      const cells = document.getElementById('cells-' + li).children;
      for (let i = 0; i < n; i++) {
        let cls = 'cell';
        if (step) {
          if (step.result !== null && step.result === i) cls += ' found';
          else if (step.probe === i) cls += ' probe';
          else if (i >= step.lo && i < step.hi) cls += ' range';
          else cls += ' out';
          if (step.marks.indexOf(i) >= 0) cls += ' mark';
        }
        cells[i].className = cls;
      }
      document.getElementById('count-' + li).textContent = pos;
      document.getElementById('status-' + li).textContent = step ? step.text : 'Start: gesucht wird ' + data.key + '.';
      const laneEl = document.getElementById('lane-' + li);
      const badge = document.getElementById('badge-' + li);
      const done = step !== null && step.result !== null;
      laneEl.classList.toggle('done', done);
      if (done) {
        finished += 1;
        const hit = step.result >= 0;
        badge.className = 'badge' + (hit ? '' : ' miss');
        badge.textContent = (hit ? 'gefunden' : 'nicht vorhanden') + ' nach ' + pos + (pos === 1 ? ' Schritt' : ' Schritten');
      }
    });
    document.getElementById('tick').textContent = 'Takt ' + tick + ' von ' + maxSteps;
    const atEnd = tick >= maxSteps;
    document.getElementById('btn-step').disabled = atEnd;
    if (atEnd) pause();
  }
  function play() {
    if (timer) return;
    if (tick >= maxSteps) tick = 0;
    document.getElementById('btn-play').textContent = '❚❚ Pause';
    timer = setInterval(() => { tick = Math.min(tick + 1, maxSteps); paint(); }, speeds[document.getElementById('speed').value]);
  }
  function pause() {
    if (timer) { clearInterval(timer); timer = null; }
    document.getElementById('btn-play').textContent = '▶ Start';
  }
  document.getElementById('btn-play').addEventListener('click', () => timer ? pause() : play());
  document.getElementById('btn-step').addEventListener('click', () => { pause(); tick = Math.min(tick + 1, maxSteps); paint(); });
  document.getElementById('btn-back').addEventListener('click', () => { pause(); tick = Math.max(tick - 1, 0); paint(); });
  document.getElementById('btn-reset').addEventListener('click', () => { pause(); tick = 0; paint(); });
  document.getElementById('speed').addEventListener('input', () => { if (timer) { pause(); play(); } });
  render();
  paint();
  if (data.autoplay) play();
})();
"""


def race_html(values, key: int, lanes: list[Lane], autoplay: bool = False) -> str:
    """Self-contained HTML document with the animated race."""
    payload = {
        "key": int(key),
        "values": [int(v) for v in values],
        "autoplay": bool(autoplay),
        "lanes": [
            {
                "name": html.escape(lane.name),
                "complexity": html.escape(lane.complexity),
                "colour": LANE_COLOURS.get(lane.key, "#898781"),
                "steps": [
                    {
                        "lo": step.lo,
                        "hi": step.hi,
                        "probe": step.probe,
                        "result": step.result,
                        "marks": list(step.marks),
                        "text": html.escape(step.text),
                    }
                    for step in lane.steps
                ],
            }
            for lane in lanes
        ],
    }
    # "</" must not appear inside the script element, json.dumps keeps "/" as is.
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return (
        "<!doctype html><html lang=\"de\"><head><meta charset=\"utf-8\">"
        f"<style>{_CSS}</style></head><body>"
        '<div class="race-controls">'
        '<button id="btn-play" class="primary">▶ Start</button>'
        '<button id="btn-step">Schritt ▶</button>'
        '<button id="btn-back">◀ Zurück</button>'
        '<button id="btn-reset">⟲ Neustart</button>'
        '<label>Tempo <input id="speed" type="range" min="0" max="4" value="2"></label>'
        '<span class="tick" id="tick"></span>'
        "</div>"
        '<div id="race"></div>'
        '<div class="legend"><i style="background:#e3eefb;border:1px solid #2a78d6"></i>noch möglicher Bereich'
        '<i style="background:#fde3d6;border:2px solid #eb6834"></i>gerade verglichen'
        '<i style="background:#f0efec;border:1px solid #e1e0d9"></i>ausgeschlossen'
        '<i style="background:#d7f0d7;border:2px solid #008300"></i>gefunden</div>'
        f"<script>{_JS.replace('__DATA__', data)}</script></body></html>"
    )
