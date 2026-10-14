"""Browser-side playback as a self-contained HTML snippet (Plotly.js + a little JavaScript).

The full history of every optimizer is sent once. Play/Pause and the slider then restyle only the moving traces
(paths, heads, loss and gradient-norm curves) with slices of those histories, so Python is not involved while playing:
Streamlit does not rerun, nothing flickers, and the contour, the surface and all axes stay exactly where they are.
(Plotly's own animate buttons blank contour traces, hence this small custom player.)
"""

import json

import plotly.io as pio

from .figures import PLOT_CONFIG

_JS = """
<script>
(function() {
  const gd = document.getElementById("fig");
  const D = __DATA__, TS = __TS__, TRAIL = __TRAIL__, TICK = __TICK__;
  const slider = document.getElementById("slider"), label = document.getElementById("label"),
        table = document.getElementById("table");
  const ITER = D.map(o => o.x.map((_, k) => k));
  let i = 0, playing = false, token = 0;
  const fmt = v => (v === 0 || (Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5)) ? v.toPrecision(4) : v.toExponential(3);
  function show(k) {
    i = k; slider.value = k; label.textContent = "Iteration: " + TS[k];
    const X = [], Y = [], Z = [], IDX = [], IDZ = [];
    const put = (id, x, y, z) => { X.push(x); Y.push(y); IDX.push(id); if (z) { Z.push(z); IDZ.push(id); } };
    let rows = "";
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
      rows += "<tr><td><span style='color:" + o.color + "'>&#9632;</span> " + o.name + "</td><td>" + e + "</td><td>" +
              (o.lr[e] === null ? "-" : fmt(o.lr[e])) + "</td><td>(" + o.x[e].toFixed(3) + ", " + o.y[e].toFixed(3) +
              ")</td><td>" + fmt(o.loss[e]) + "</td><td>" + fmt(o.gnorm[e]) + "</td><td>" +
              ((!o.alive && e < TS[k]) ? "stopped (non-finite)" : "running") + "</td></tr>";
    });
    table.innerHTML = rows;
    return Plotly.restyle(gd, {x: X, y: Y}, IDX).then(() => Plotly.restyle(gd, {z: Z}, IDZ));
  }
  async function play() {
    if (playing) return;
    playing = true; const mine = ++token;
    if (i >= TS.length - 1) await show(0);
    while (playing && mine === token && i < TS.length - 1) {
      await show(i + 1);
      await new Promise(r => setTimeout(r, __MS__));
    }
    if (mine === token) playing = false;
  }
  document.getElementById("play").onclick = play;
  document.getElementById("pause").onclick = () => { playing = false; token++; };
  slider.oninput = () => { playing = false; token++; show(+slider.value); };
  const ready = setInterval(() => { if (gd && gd.data) { clearInterval(ready); show(0); } }, 50);
})();
</script>"""


def player_html(fig, info, last, frame_ms=100, trail=0, tick=0, max_frames=200):
    """HTML (controls + figure + live table) that replays iterations 0..``last``. Returns ``(html, height_px)``."""
    ts = sorted({int(round(v)) for v in _linspace(0, last, min(last + 1, max_frames))})
    data = []
    for name, o in info["opt"].items():
        data.append({"name": name, "color": o["color"], "idx": o["idx"], "x": _r(o["x"]), "y": _r(o["y"]),
                     "z": _r(o["z"]), "loss": [float(f"{v:.6g}") for v in o["loss"]],
                     "gnorm": [float(f"{v:.6g}") for v in o["gnorm"]],
                     "lr": [None if v is None else float(f"{v:.6g}") for v in o["lr"]], "alive": o["alive"]})
    height = int(fig.layout.height or 820)
    plot = pio.to_html(fig, full_html=False, include_plotlyjs="cdn", config=PLOT_CONFIG, div_id="fig",
                       default_width="100%", default_height=f"{height}px")
    controls = f"""
<div style="font:13px sans-serif;display:flex;align-items:center;gap:8px;padding:4px 2px;">
  <button id="play" style="padding:3px 10px">&#9654; Play</button>
  <button id="pause" style="padding:3px 10px">&#10074;&#10074; Pause</button>
  <input id="slider" type="range" min="0" max="{len(ts) - 1}" value="0" style="flex:1">
  <span id="label" style="min-width:110px">Iteration: 0</span>
</div>"""
    table = """
<table style="font:12px sans-serif;border-collapse:collapse;margin:4px 2px;width:100%">
<thead><tr style="text-align:left;border-bottom:1px solid #ccc"><th>Optimizer</th><th>iteration</th><th>learning rate</th>
<th>point (x, y)</th><th>loss</th><th>grad norm</th><th>status</th></tr></thead><tbody id="table"></tbody></table>"""
    script = (_JS.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__TS__", json.dumps(ts))
              .replace("__TRAIL__", str(int(trail))).replace("__TICK__", str(int(tick))).replace("__MS__", str(int(frame_ms))))
    return controls + plot + table + script, height + 48 + 40 + 24 * len(data)


def _linspace(a, b, n):
    return [a + (b - a) * k / max(n - 1, 1) for k in range(n)]


def _r(values):
    return [round(float(v), 6) for v in values]
