"""Browser-side playback as a self-contained HTML snippet (Plotly.js + a little JavaScript).

The full history of every optimizer is sent once. Play/Pause and the slider then restyle only the moving traces
(paths, heads, loss and gradient-norm curves) with slices of those histories, so Python is not involved while playing:
Streamlit does not rerun, nothing flickers, and the contour, the surface and all axes stay exactly where they are.
(Plotly's own animate buttons blank contour traces, hence this small custom player.)

Speed, trail length, iteration markers and the 3-D camera are controls inside the page, so changing them never
rebuilds the figure in Python.
"""

import json

import numpy as np
import plotly.io as pio

from .figures import PLOT_CONFIG

SPEEDS = [("Slow", 250), ("Normal", 100), ("Fast", 40), ("Very fast", 15)]

_JS = """
<script>
(function() {
  const gd = document.getElementById("fig");
  const D = __DATA__, TS = __TS__, INT = __INT__;
  let TRAIL = 0, TICK = 0, MS = 100;
  const slider = document.getElementById("slider"), label = document.getElementById("label"),
        table = document.getElementById("table"), internals = document.getElementById("internals");
  const ITER = D.map(o => o.x.map((_, k) => k));
  let i = 0, playing = false, token = 0;
  const fmt = v => (v === null || v === undefined) ? "-" :
      ((v === 0 || (Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5)) ? v.toPrecision(4) : v.toExponential(3));
  const vec = a => (a === null || a === undefined) ? "-" : "(" + fmt(a[0]) + ", " + fmt(a[1]) + ")";
  function show(k) {
    i = k; slider.value = k; label.textContent = "Iteration: " + TS[k];
    const X = [], Y = [], Z = [], IDX = [], IDZ = [];
    const put = (id, x, y, z) => { X.push(x); Y.push(y); IDX.push(id); if (z) { Z.push(z); IDZ.push(id); } };
    let rows = "", irows = "";
    D.forEach((o, j) => {
      const e = Math.min(TS[k], o.x.length - 1), s = TRAIL > 0 ? Math.max(0, e - TRAIL + 1) : 0, I = o.idx;
      put(I.path, o.x.slice(s, e + 1), o.y.slice(s, e + 1));
      const tx = [], ty = [];
      if (TICK > 0) for (let q = s; q <= e; q++) if (q % TICK === 0) { tx.push(o.x[q]); ty.push(o.y[q]); }
      put(I.tick, tx, ty);
      put(I.head, [o.x[e]], [o.y[e]]);
      put(I.lline, ITER[j].slice(0, e + 1), o.loss.slice(0, e + 1));
      put(I.lhead, [e], [o.loss[e]]);
      put(I.gline, ITER[j].slice(0, e + 1), o.gnorm.slice(0, e + 1));
      put(I.ghead, [e], [o.gnorm[e]]);
      put(I.p3, o.x.slice(s, e + 1), o.y.slice(s, e + 1), o.z.slice(s, e + 1));
      put(I.h3, [o.x[e]], [o.y[e]], [o.z[e]]);
      const sw = "<span style='color:" + o.color + "'>&#9632;</span> " + o.name;
      const status = (!o.alive && e < TS[k]) ? "stopped (non-finite)" : "running";
      rows += "<tr><td>" + sw + "</td><td>" + e + "</td><td>" + fmt(o.lr[e]) + "</td><td>(" + o.x[e].toFixed(3) + ", " +
              o.y[e].toFixed(3) + ")</td><td>" + fmt(o.loss[e]) + "</td><td>" + fmt(o.gnorm[e]) + "</td><td>" + status + "</td></tr>";
      const f = INT[j][k];
      const bufs = Object.keys(f.v).map(n => n + " = " + vec(f.v[n])).join(", ");
      irows += "<tr><td>" + sw + "</td><td>" + vec(f.g) + "</td><td>" + vec(f.s) + "</td><td>" + vec(f.r) + "</td><td>" + bufs + "</td></tr>";
    });
    table.innerHTML = rows; internals.innerHTML = irows;
    return Plotly.restyle(gd, {x: X, y: Y}, IDX).then(() => Plotly.restyle(gd, {z: Z}, IDZ));
  }
  async function play() {
    if (playing) return;
    playing = true; const mine = ++token;
    if (i >= TS.length - 1) await show(0);
    while (playing && mine === token && i < TS.length - 1) {
      await show(i + 1);
      await new Promise(r => setTimeout(r, MS));
    }
    if (mine === token) playing = false;
  }
  document.getElementById("play").onclick = play;
  document.getElementById("pause").onclick = () => { playing = false; token++; };
  slider.oninput = () => { playing = false; token++; show(+slider.value); };
  document.getElementById("speed").onchange = ev => { MS = +ev.target.value; };
  document.getElementById("trail").onchange = ev => { TRAIL = Math.max(0, Math.floor(+ev.target.value || 0)); show(i); };
  document.getElementById("tick").onchange = ev => { TICK = Math.max(0, Math.floor(+ev.target.value || 0)); show(i); };
  // 3-D camera presets and auto-rotate (these only touch the camera, not the data)
  const CAMS = {iso: {x: 1.2, y: -1.3, z: 0.8}, top: {x: 0.0001, y: -0.0001, z: 2.6}, side: {x: 2.1, y: 0.0001, z: 0.3}};
  const setEye = eye => Plotly.relayout(gd, {"scene.camera": {eye: eye, up: {x: 0, y: 0, z: 1}, center: {x: 0, y: 0, z: 0}}});
  document.querySelectorAll("[data-cam]").forEach(b => { b.onclick = () => setEye(CAMS[b.dataset.cam]); });
  let spin = null;
  document.getElementById("spin").onchange = ev => {
    if (spin) { clearInterval(spin); spin = null; }
    if (ev.target.checked) spin = setInterval(() => {
      const c = (gd._fullLayout.scene.camera || {}).eye || CAMS.iso, r = Math.hypot(c.x, c.y), a = Math.atan2(c.y, c.x) + 0.04;
      setEye({x: r * Math.cos(a), y: r * Math.sin(a), z: c.z});
    }, 60);
  };
  const ready = setInterval(() => { if (gd && gd.data) { clearInterval(ready); show(0); } }, 50);
})();
</script>"""

_CSS = "font:13px sans-serif;display:flex;align-items:center;flex-wrap:wrap;gap:8px;padding:4px 2px;"


def _internals(o, t):
    """What the optimizer 'knew' when it produced iterate t: gradient, step, step/|g| per coordinate, buffers."""
    e = min(t, len(o["grad"]) - 1)
    g, s = o["grad"][e], o["step"][e]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(g) > 1e-12, np.abs(s) / np.abs(g), np.nan)
    f = lambda a: None if not np.all(np.isfinite(a)) and not np.any(np.isfinite(a)) else \
        [float(f"{v:.4g}") if np.isfinite(v) else None for v in a]
    return {"g": f(g), "s": f(s), "r": f(r), "v": {k: f(v) for k, v in o["view"][e].items()}}


def player_html(fig, info, last, max_frames=200):
    """HTML (controls + figure + live tables) that replays iterations 0..``last``. Returns ``(html, height_px)``."""
    ts = sorted({int(round(v)) for v in np.linspace(0, last, min(last + 1, max_frames))})
    data, ints = [], []
    for name, o in info["opt"].items():
        data.append({"name": name, "color": o["color"], "idx": o["idx"], "x": _r(o["x"]), "y": _r(o["y"]),
                     "z": _r(o["z"]), "loss": [float(f"{v:.6g}") for v in o["loss"]],
                     "gnorm": [float(f"{v:.6g}") for v in o["gnorm"]],
                     "lr": [None if v is None else float(f"{v:.6g}") for v in o["lr"]], "alive": o["alive"]})
        ints.append([_internals(o, t) for t in ts])
    height = int(fig.layout.height or 900)
    plot = pio.to_html(fig, full_html=False, include_plotlyjs="cdn", config=PLOT_CONFIG, div_id="fig",
                       default_width="100%", default_height=f"{height}px")
    speed = "".join(f"<option value='{ms}'{' selected' if label == 'Normal' else ''}>{label}</option>" for label, ms in SPEEDS)
    controls = f"""
<div style="{_CSS}">
  <button id="play" style="padding:3px 10px">&#9654; Play</button>
  <button id="pause" style="padding:3px 10px">&#10074;&#10074; Pause</button>
  <input id="slider" type="range" min="0" max="{len(ts) - 1}" value="0" style="flex:1;min-width:160px">
  <span id="label" style="min-width:110px">Iteration: 0</span>
</div>
<div style="{_CSS}color:#333">
  <label>Speed <select id="speed">{speed}</select></label>
  <label>Trail <input id="trail" type="number" min="0" value="0" style="width:64px" title="0 = whole path"></label>
  <label>Mark every <input id="tick" type="number" min="0" value="0" style="width:56px" title="0 = off"> it.</label>
  <span style="margin-left:12px">3-D view:</span>
  <button data-cam="iso">Isometric</button><button data-cam="top">Top</button><button data-cam="side">Side</button>
  <label><input id="spin" type="checkbox"> auto-rotate</label>
  <span style="color:#777">(drag the surface to rotate, scroll to zoom)</span>
</div>"""
    th = "text-align:left;border-bottom:1px solid #ccc;padding:2px 8px 2px 0"
    glossary = "<br>".join(f"<b>{n}</b>: {o['glossary']}" for n, o in info["opt"].items() if o["glossary"])
    tables = f"""
<div style="font:12px sans-serif;margin:6px 2px">
<b>Now</b>
<table style="border-collapse:collapse;width:100%"><thead><tr>
<th style="{th}">Optimizer</th><th style="{th}">iteration</th><th style="{th}">learning rate</th><th style="{th}">point (x, y)</th>
<th style="{th}">loss</th><th style="{th}">grad norm</th><th style="{th}">status</th></tr></thead><tbody id="table"></tbody></table>
<div style="margin-top:8px"><b>Inside the optimizer</b> at this iteration (the update that produced it)</div>
<table style="border-collapse:collapse;width:100%"><thead><tr>
<th style="{th}">Optimizer</th><th style="{th}">minibatch gradient g</th><th style="{th}">step &Delta;w</th>
<th style="{th}">step / |g| per coordinate</th><th style="{th}">internal state</th></tr></thead><tbody id="internals"></tbody></table>
<div style="color:#555;margin-top:4px">{glossary}</div>
</div>"""
    script = (_JS.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__TS__", json.dumps(ts))
              .replace("__INT__", json.dumps(ints, separators=(",", ":"))))
    extra = 48 + 40 + 60 + 24 * (len(data) + 1) * 2 + 20 * len(data) + 40
    # An explicit light card: the plots are white, so the page must not inherit Streamlit's dark theme (black text on dark).
    style = "<style>html,body{margin:0;background:#fff;color:#222}</style>"
    card = "<div style='background:#fff;color:#222;padding:6px 8px;border-radius:6px'>"
    return style + card + controls + plot + tables + "</div>" + script, height + extra


def player_page(snippet, title="2-D optimizer dashboard"):
    """The player as a standalone HTML file (needs internet only for plotly.js from the CDN)."""
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title></head><body style='margin:8px'>{snippet}</body></html>")


def _r(values):
    return [round(float(v), 6) for v in values]
