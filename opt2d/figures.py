"""The 2x2 figure: contour + paths, loss, gradient norm, 3-D surface + paths.

``build_figure`` draws everything that never moves (contour, surface, markers, axes with FIXED ranges) and one set
of 'moving' traces per optimizer, starting at iteration 0. ``player.py`` then replays a finished run by restyling only
those traces, so contours, surface and axes stay put while the paths grow.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .optimizers import OPTIMIZERS

PLOT_CONFIG = {"responsive": True, "toImageButtonOptions": {"format": "png", "scale": 2}}
CONTOUR_COLORS = [[0, "#f4f7fa"], [1, "#a9b7c6"]]
TITLES = ["Trajectory", "Loss (full objective)", "Gradient norm", "Objective surface and trajectory"]


def color(name):
    return OPTIMIZERS[name].color


# ---------------------------------------------------------------- numbers behind the pictures
def grid_values(objective, data, xs, ys):
    """Loss on the grid xs x ys. Tries one vectorised call (fast), checks it, else falls back to a loop."""
    X, Y = np.meshgrid(xs, ys)
    Z = None
    try:
        Z = np.asarray(objective.loss(np.stack([X, Y]), data), dtype=float)
        if Z.shape != X.shape:
            Z = None
        else:
            for a, b in ((0, 0), (len(ys) - 1, len(xs) // 2), (len(ys) // 2, len(xs) - 1)):
                if not np.isclose(float(objective.loss(np.array([X[a, b], Y[a, b]]), data)), Z[a, b], rtol=1e-6,
                                  atol=1e-9, equal_nan=True):
                    Z = None
                    break
    except Exception:
        Z = None
    if Z is None:
        Z = np.array([[float(objective.loss(np.array([x, y]), data)) for x in xs] for y in ys])
    Z[~np.isfinite(Z)] = np.nan
    return Z


def transform(values, floor, log):
    """Colour / height scale: raw loss, or log10(loss - floor + 1e-3) so that valleys stay visible."""
    values = np.asarray(values, dtype=float)
    return np.log10(np.maximum(values - floor, 0.0) + 1e-3) if log else values


def fit_range(arrays, log):
    """Axis range covering every array (log axes get padding in decades)."""
    v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    if not v.size:
        return [0.0, 1.0], 0.0
    if log:
        lo, hi = np.log10(max(v.min(), 1e-16)), np.log10(max(v.max(), 1e-16))
        pad = 0.05 * (hi - lo) + 0.15
        return [float(lo - pad), float(hi + pad)], float(hi - lo)
    lo, hi = float(v.min()), float(v.max())
    pad = 0.05 * ((hi - lo) or 1.0)
    return [lo - pad, hi + pad], hi - lo


def grid_xyz(run, cx, cy, half, n):
    """xs, ys, loss grid over the window (the expensive part of a figure: the app caches it)."""
    xs, ys = np.linspace(cx - half, cx + half, n), np.linspace(cy - half, cy + half, n)
    return xs, ys, grid_values(run.objective, run.data, xs, ys)


def window_of(run, view, ghost):
    """(cx, cy, half) of the plotted window: fixed, or fitted to start, optima and every path point."""
    cx, cy, half = view["cx"], view["cy"], view["half"]
    if view["window"] != "fit":
        return cx, cy, half
    pts = [run.w0[None, :]] + [np.array(h["w"]) for h in run.history.values()]
    pts += [np.array(g) for g in (ghost["w"].values() if ghost else [])] + [np.array(o)[None, :] for o in run.objective.optima]
    P = np.vstack(pts)
    P = P[np.all(np.abs(P - [cx, cy]) <= 20 * half, axis=1)]          # ignore points that flew off
    lo, hi = P.min(0), P.max(0)
    return float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(max(0.5, 0.6 * (hi - lo).max() * 1.15))


# ---------------------------------------------------------------- the figure
def build_figure(run, view, ghost=None, click=False, grid_fn=grid_xyz):
    """Returns ``(fig, info)``. ``view``: window ('fixed'|'fit'), cx, cy, half, grid, log_z, log_loss, log_norm.

    ``info`` has everything the player needs: per optimizer the full history and the trace indices to move.
    ``click`` adds an invisible grid of markers on the contour so that a click can set the start point.
    ``grid_fn(run, cx, cy, half, n)`` returns ``(xs, ys, Z)``; pass a cached version to skip the grid evaluation.
    """
    obj = run.objective
    cx, cy, half = window_of(run, view, ghost)
    n = int(view["grid"])
    xs, ys, Z = grid_fn(run, cx, cy, half, n)

    loss_paths = [h["loss"] for h in run.history.values()] + [[run.loss0]]
    if ghost:
        loss_paths += list(ghost["loss"].values())
    floor = float(np.nanmin(np.concatenate([Z.ravel()[np.isfinite(Z.ravel())]] + [np.asarray(v) for v in loss_paths])))
    log_z = bool(view["log_z"])
    Zt = transform(Z, floor, log_z)

    fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{}, {"type": "scene"}]], subplot_titles=TITLES,
                        horizontal_spacing=0.07, vertical_spacing=0.10)
    for ann in fig.layout.annotations:
        ann.font = dict(size=12)

    def add(trace, row, col):
        fig.add_trace(trace, row=row, col=col)
        return len(fig.data) - 1

    # --- trajectory panel
    add(go.Contour(x=xs, y=ys, z=Zt, showscale=False, ncontours=18, hoverinfo="skip", colorscale=CONTOUR_COLORS,
                   line=dict(width=0.6, color="#6b7b8c")), 1, 1)
    click_idx = None
    if click:
        gx, gy = np.meshgrid(np.linspace(xs[0], xs[-1], 15), np.linspace(ys[0], ys[-1], 15))
        click_idx = add(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", marker=dict(size=14, opacity=0),
                                   hovertemplate="x=%{x:.3g}, y=%{y:.3g}<extra>click to set the start</extra>",
                                   showlegend=False), 1, 1)
    optima = np.array(obj.optima, dtype=float).reshape(-1, 2)
    if len(optima):
        add(go.Scatter(x=optima[:, 0], y=optima[:, 1], mode="markers", name="minimum",
                       marker=dict(symbol="x", size=9, color="black")), 1, 1)
    add(go.Scatter(x=[run.w0[0]], y=[run.w0[1]], mode="markers", name="start",
                   marker=dict(symbol="circle", size=9, color="white", line=dict(color="black", width=2))), 1, 1)

    # --- surface (the same window and colour scale as the contour)
    z_lift = 0.0
    si = np.unique(np.round(np.linspace(0, n - 1, min(n, 31))).astype(int))        # a coarser grid is enough in 3-D
    add(go.Surface(x=xs[si], y=ys[si], z=Zt[np.ix_(si, si)], showscale=False, opacity=0.85, colorscale="Blues", hoverinfo="skip",
                   contours=dict(z=dict(show=False))), 2, 2)

    def z_of(loss):
        return transform(loss, floor, log_z)

    # heights of every path point that falls inside the window decide the z range
    inside = lambda w: (np.abs(w[:, 0] - cx) <= half) & (np.abs(w[:, 1] - cy) <= half)
    zs = [Zt[np.isfinite(Zt)]]
    for h in run.history.values():
        w = np.array(h["w"])
        zs.append(z_of(np.array(h["loss"]))[inside(w)])
    zs.append(z_of(np.array([run.loss0])))
    zmin, zmax = float(min(a.min() for a in zs if a.size)), float(max(a.max() for a in zs if a.size))
    z_lift = 0.01 * (zmax - zmin)
    if len(optima):
        zo = z_of(np.array([float(obj.loss(o, run.data)) for o in optima])) + z_lift
        add(go.Scatter3d(x=optima[:, 0], y=optima[:, 1], z=zo, mode="markers", name="minimum", showlegend=False,
                         marker=dict(symbol="x", size=4, color="black")), 2, 2)
    add(go.Scatter3d(x=[run.w0[0]], y=[run.w0[1]], z=[float(z_of(np.array([run.loss0]))[0]) + z_lift], mode="markers",
                     name="start", showlegend=False, marker=dict(size=4, color="white", line=dict(color="black", width=2))), 2, 2)

    # --- pinned run (static, faded, underneath the live traces)
    for name in (ghost["names"] if ghost else []):
        w = np.array(ghost["w"][name])
        style = dict(color=color(name))
        add(go.Scatter(x=w[:, 0], y=w[:, 1], mode="lines", hoverinfo="skip", showlegend=False, opacity=0.4,
                       line=dict(width=2, dash="dot", **style)), 1, 1)
        for key, row in (("loss", 1), ("gnorm", 2)):
            y = np.maximum(ghost[key][name], 1e-16)
            add(go.Scatter(x=list(range(len(y))), y=y, mode="lines", hoverinfo="skip", showlegend=False, opacity=0.4,
                           line=dict(width=2, dash="dot", **style)), *((1, 2) if key == "loss" else (2, 1)))
        add(go.Scatter3d(x=w[:, 0], y=w[:, 1], z=z_of(np.array(ghost["loss"][name])) + z_lift, mode="lines",
                         hoverinfo="skip", showlegend=False, opacity=0.4, line=dict(width=3, dash="dot", **style)), 2, 2)

    # --- live traces: one set per optimizer, drawn at iteration 0 and moved by the player
    info = {"opt": {}, "click": click_idx, "window": (cx, cy, half), "z_lift": z_lift}
    for name in run.names:
        h = run.history[name]
        c = color(name)
        w0, l0, g0 = run.w0, max(h["loss"][0], 1e-16), max(h["gnorm"][0], 1e-16)
        z0 = float(z_of(np.array([h["loss"][0]]))[0]) + z_lift
        ids = dict(
            path=add(go.Scatter(x=[w0[0]], y=[w0[1]], mode="lines", name=name, legendgroup=name,
                                line=dict(color=c, width=2.5)), 1, 1),
            tick=add(go.Scatter(x=[], y=[], mode="markers", legendgroup=name, showlegend=False, hoverinfo="skip",
                                marker=dict(symbol="diamond", size=6, color=c)), 1, 1),
            head=add(go.Scatter(x=[w0[0]], y=[w0[1]], mode="markers", legendgroup=name, showlegend=False,
                                hoverinfo="skip", marker=dict(size=10, color=c, line=dict(color="white", width=1.5))), 1, 1),
            lline=add(go.Scatter(x=[0], y=[l0], mode="lines", legendgroup=name, showlegend=False,
                                 line=dict(color=c, width=2)), 1, 2),
            lhead=add(go.Scatter(x=[0], y=[l0], mode="markers", legendgroup=name, showlegend=False, hoverinfo="skip",
                                 marker=dict(size=9, color=c, line=dict(color="white", width=1))), 1, 2),
            gline=add(go.Scatter(x=[0], y=[g0], mode="lines", legendgroup=name, showlegend=False,
                                 line=dict(color=c, width=2)), 2, 1),
            ghead=add(go.Scatter(x=[0], y=[g0], mode="markers", legendgroup=name, showlegend=False, hoverinfo="skip",
                                 marker=dict(size=9, color=c, line=dict(color="white", width=1))), 2, 1),
            p3=add(go.Scatter3d(x=[w0[0]], y=[w0[1]], z=[z0], mode="lines", legendgroup=name, showlegend=False,
                                line=dict(color=c, width=5)), 2, 2),
            h3=add(go.Scatter3d(x=[w0[0]], y=[w0[1]], z=[z0], mode="markers", legendgroup=name, showlegend=False,
                                hoverinfo="skip", marker=dict(size=5, color=c)), 2, 2),
        )
        info["opt"][name] = dict(
            idx=ids, color=c,
            x=np.array(h["w"])[:, 0].tolist(), y=np.array(h["w"])[:, 1].tolist(),
            z=(z_of(np.array(h["loss"])) + z_lift).tolist(),
            loss=np.maximum(h["loss"], 1e-16).tolist(), gnorm=np.maximum(h["gnorm"], 1e-16).tolist(),
            lr=[None if v != v else v for v in h["lr"]], alive=bool(run.alive[name]),
            grad=np.array(h["grad"]), step=np.array(h["step"]), view=h["view"], glossary=OPTIMIZERS[name].glossary)

    # --- fixed axes: nothing rescales while the run is replayed
    fig.update_xaxes(range=[cx - half, cx + half], row=1, col=1, title_text="x", title_standoff=2)
    fig.update_yaxes(range=[cy - half, cy + half], row=1, col=1, title_text="y", title_standoff=2)
    x_iter = [0, max(run.max_iters, 1)]
    log_loss = bool(view["log_loss"]) and all(min(a) > 0 for a in loss_paths if len(a))
    log_norm = bool(view["log_norm"])
    gn_paths = [h["gnorm"] for h in run.history.values()] + [[run.gnorm0]] + (list(ghost["gnorm"].values()) if ghost else [])
    for key, row, col, paths, log, label in (("loss", 1, 2, loss_paths, log_loss, "loss"),
                                             ("gnorm", 2, 1, gn_paths, log_norm, "gradient norm")):
        rng, decades = fit_range([np.maximum(p, 1e-16) if log else p for p in paths], log)
        wide = log and decades >= 2                          # one labelled tick per power of ten
        step = 1 if decades <= 8 else 2 if decades <= 16 else 4
        fig.update_xaxes(range=x_iter, title_text="iteration", title_standoff=2, row=row, col=col)
        fig.update_yaxes(type="log" if log else "linear", range=rng, title_text=label, title_standoff=2,
                         dtick=step if wide else None, exponentformat="power" if wide else "B", row=row, col=col)
    zrange = [zmin - 0.02 * (zmax - zmin), zmax + 0.02 * (zmax - zmin)]
    fig.update_layout(
        template="plotly_white", font=dict(size=11), height=900, margin=dict(l=55, r=30, t=40, b=60),
        uirevision="keep", modebar=dict(orientation="v"), legend=dict(orientation="h", x=0, y=-0.06),
        scene=dict(xaxis=dict(range=[cx - half, cx + half], title="x", nticks=5, tickfont=dict(size=9)),
                   yaxis=dict(range=[cy - half, cy + half], title="y", nticks=5, tickfont=dict(size=9)),
                   zaxis=dict(range=zrange, title="log loss" if log_z else "loss", nticks=5, tickfont=dict(size=9)),
                   aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.7),
                   camera=dict(eye=dict(x=1.2, y=-1.3, z=0.8), up=dict(x=0, y=0, z=1))))
    return go.Figure(fig), info
