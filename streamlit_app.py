"""Streamlit front end of the 2-D optimizer dashboard.  Run:  streamlit run streamlit_app.py

Streamlit re-runs this script on every interaction, so all state lives in ``st.session_state``: the control values
(see opt2d/settings.py), the computed ``Run`` and the pinned ghost run.
Run computes every iteration at once; the 2x2 figure then replays it in the browser (opt2d/player.py), so playing
causes no reruns and no flicker, and the contour, surface and axes stay fixed. Speed, trail, markers and the 3-D camera
are controls inside the player, so they never rebuild anything in Python.
"""

import os
from dataclasses import replace

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from opt2d import OPTIMIZERS, get_objectives
from opt2d import settings as S
from opt2d.analysis import convergence_rows
from opt2d.export import history_csv
from opt2d.figures import PLOT_CONFIG, build_figure, build_picker, grid_xyz
from opt2d.lessons import LESSONS
from opt2d.objectives import CUSTOM_NAME
from opt2d.player import player_html, player_page
from opt2d.schedules import SCHEDULES, sparkline_svg

# Custom objectives execute user code: only switch them on for a local / trusted app (ALLOW_CUSTOM=1).
ALLOW_CUSTOM = os.environ.get("ALLOW_CUSTOM", "") == "1"
WINDOWS = {"fixed": "Fixed window (centre and half-width below)", "fit": "Fit to the paths after a run"}
LOSS_TOLS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]
GNORM_TOLS = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]

st.set_page_config(page_title="2-D optimizer dashboard", page_icon="📉", layout="wide")
ss = st.session_state


@st.cache_resource
def objectives():
    return get_objectives()


@st.cache_data(max_entries=24, show_spinner=False)
def cached_grid(key, _run, cx, cy, half, n):
    """Contour / surface grid. The key (objective, code, data size and seed) identifies the loss; ``_run`` is not hashed."""
    return grid_xyz(_run, cx, cy, half, n)


def grid_fn(cfg):
    key = (cfg.objective, cfg.code, cfg.n, cfg.seed)
    return lambda run, cx, cy, half, n: cached_grid(key, run, cx, cy, half, n)


OBJ = objectives()
if "init" not in ss:
    ss.update(S.defaults(OBJ))
    ss.update(init=True, run=None, run_cfg=None, ghost=None, figs={}, error="", click_n=0)


# ------------------------------------------------------------------ callbacks (run before the script re-runs)
def touch():
    """Any manual change: the settings no longer match the chosen guided example exactly."""
    if ss.lesson != S.MANUAL:
        ss.lesson = S.MANUAL


def forget_run():
    ss.run, ss.figs, ss.error = None, {}, ""


def on_lesson():
    if ss.lesson != S.MANUAL:
        S.apply_lesson(ss, ss.lesson, OBJ)
        forget_run()
        ss.ghost = None


def on_objective():
    S.load_objective(ss, ss.objective, OBJ)
    touch()
    forget_run()


def restore(name):
    ss.update(S.optimizer_defaults(name, OBJ[ss.objective].own_lr.get(name) if ss.objective in OBJ else None))
    touch()


def on_pin():
    ss.ghost = S.pin_from(ss.run, f"{ss.objective}; ")


def on_unpin():
    ss.ghost = None


def set_start_from_click(key, click_index):
    """A click on a contour sets the start point (the point is taken from the invisible click grid)."""
    ev = ss.get(key)
    points = (ev or {}).get("selection", {}).get("points", []) if ev else []
    if not points or click_index is None:
        return False
    pt = points[0]
    if pt.get("curve_number") != click_index:
        return False
    ss.start_x, ss.start_y = round(float(pt["x"]), 3), round(float(pt["y"]), 3)
    ss.click_n += 1                                    # a new chart key clears the selection
    touch()
    return True


def on_click():
    """Before a run: click the preview contour to set the start."""
    cached = ss.figs.get("preview")
    if cached is not None and set_start_from_click(f"preview_{ss.click_n}", cached[2]["click"]):
        ss.figs = {}


def on_pick():
    """After a run: click the small contour under the player to set the NEXT start (the run stays on screen)."""
    cached = ss.figs.get("pick")
    if cached is not None:
        set_start_from_click(f"pick_{ss.click_n}", cached[2])


def embed(html, height):
    """Show our own (trusted) HTML+JavaScript player in an iframe."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:                                              # older Streamlit
        components.html(html, height=height)


def plot(fig, key, **kw):
    try:
        return st.plotly_chart(fig, width="stretch", key=key, config=PLOT_CONFIG, **kw)
    except TypeError:                                  # older Streamlit
        return st.plotly_chart(fig, use_container_width=True, key=key, config=PLOT_CONFIG, **kw)


# ------------------------------------------------------------------ sidebar
def sidebar():
    st.header("Guided example")
    st.selectbox("What to look for", [S.MANUAL] + [L.name for L in LESSONS], key="lesson", on_change=on_lesson)

    st.header("Problem")
    st.selectbox("Objective", list(OBJ) + [CUSTOM_NAME], key="objective", on_change=on_objective)
    if ss.objective != CUSTOM_NAME:
        st.caption(OBJ[ss.objective].description)
    else:
        if not ALLOW_CUSTOM:
            st.warning("Custom code is switched off. Start the app with ALLOW_CUSTOM=1 to enable it.")
        st.text_area("def loss(w): ...   (np = autograd.numpy)", key="code", height=170, on_change=forget_run)
        st.caption("Set the plotted window under View.")
    has_data = ss.objective != CUSTOM_NAME and OBJ[ss.objective].make_data is not None
    c = st.columns(3)
    c[0].number_input("N samples", 10, 100000, step=50, key="n", disabled=not has_data, on_change=touch)
    c[1].number_input("Batch m", 1, 100000, step=8, key="m", disabled=not has_data, on_change=touch)
    c[2].number_input("Seed", 0, 10**6, key="seed", disabled=not has_data, on_change=touch)
    st.markdown("**Start point** (or click the contour before a run)")
    c = st.columns(2)
    c[0].number_input("x", step=0.1, format="%.3f", key="start_x", on_change=touch)
    c[1].number_input("y", step=0.1, format="%.3f", key="start_y", on_change=touch)

    st.header("Optimizers")
    for name, opt in OPTIMIZERS.items():
        st.checkbox(name, key=S.use_key(name), on_change=touch)
        with st.expander("settings"):
            st.text_input("own learning rate (blank = global)", key=S.lr_key(name), on_change=touch)
            for p, default in opt.params.items():
                key, kind = S.param_key(name, p), S.param_kind(default)
                if kind == "bool":
                    st.checkbox(p, key=key, on_change=touch)
                elif kind == "int":
                    st.number_input(p, 0, 1000, step=1, key=key, on_change=touch)
                elif kind == "float":
                    st.text_input(p, key=key, on_change=touch)
                else:
                    st.selectbox(p, default, key=key, on_change=touch)
            st.button("Restore defaults", key=f"restore_{name}", on_click=restore, args=(name,))

    st.header("Run")
    st.text_input("Global learning rate", key="lr", on_change=touch)
    st.number_input("Max iterations", 1, 20000, step=10, key="iters", on_change=touch)
    labels = dict((v, k) for k, v in SCHEDULES)
    st.selectbox("Learning-rate schedule", list(labels), format_func=labels.get, key="schedule", on_change=touch)
    st.markdown(sparkline_svg(ss.schedule, int(ss.iters)), unsafe_allow_html=True)

    with st.expander("View"):
        st.selectbox("Window", list(WINDOWS), format_func=WINDOWS.get, key="window")
        c = st.columns(3)
        c[0].number_input("Centre x", step=0.5, key="cx")
        c[1].number_input("Centre y", step=0.5, key="cy")
        c[2].number_input("Half-width", 0.1, 1000.0, step=0.5, key="half_width")
        st.slider("Grid resolution", 21, 81, step=10, key="grid")
        st.checkbox("Log-scale contour and surface", key="log_z")
        st.checkbox("Log loss axis", key="log_loss")
        st.checkbox("Log gradient-norm axis", key="log_norm")
        st.caption("Speed, trail length, iteration markers and the 3-D camera are controls above the plots.")


with st.sidebar:
    sidebar()

# ------------------------------------------------------------------ main area: buttons and run
st.title("2-D Optimizer Dashboard")
st.caption("Luca Steyn · Statistics and Actuarial Science, Stellenbosch University · "
           "Compare SGD with momentum, AdaGrad, RMSProp and Adam on a function of two variables")
lesson = next((L for L in LESSONS if L.name == ss.lesson), None)
if lesson:
    st.info(lesson.note)

try:
    cfg, cfg_error = S.config_from(ss), ""
except ValueError as e:
    cfg, cfg_error = None, str(e)

b = st.columns([1, 1, 1, 1, 1.3, 1.3])
run_clicked = b[0].button("Run", type="primary", width="stretch", disabled=cfg is None)
b[1].button("Reset", on_click=forget_run, width="stretch")
b[2].button("Pin run", on_click=on_pin, disabled=ss.run is None, width="stretch")
b[3].button("Clear pin", on_click=on_unpin, disabled=ss.ghost is None, width="stretch")

if run_clicked:
    try:
        run = S.build_run(cfg, OBJ, ALLOW_CUSTOM)
        bar = st.progress(0.0, text="Computing all iterations...")
        for k in range(run.max_iters):
            run.step()
            if k % 10 == 0:
                bar.progress((k + 1) / run.max_iters, text=f"Computing iteration {k + 1} of {run.max_iters}")
        bar.empty()
        if lesson:                                      # the example's baseline run, pinned automatically
            base_cfg, prefix = S.lesson_ghost_config(cfg, lesson.name)
            if base_cfg is not None:
                base = S.build_run(base_cfg, OBJ, ALLOW_CUSTOM)
                base.run()
                ss.ghost = S.pin_from(base, prefix)
        ss.update(run=run, run_cfg=cfg, figs={}, error="")
        run_ok = True
    except Exception as e:                              # bad input or user code: show it, keep the app alive
        ss.error = f"{type(e).__name__}: {e}"
        run_ok = False
    if run_ok:
        st.rerun()                                      # redraw the buttons (Pin, ...) now that a run exists

if cfg_error:
    st.error(cfg_error)
if ss.error:
    st.error(ss.error)
if ss.ghost:
    st.caption(f"Pinned (faded, dotted): {ss.ghost['label']}")
if ss.run is not None:
    for msg in ss.run.messages:
        st.warning(msg)
    if cfg is not None and cfg != ss.run_cfg:
        st.caption("Settings changed since this run: press Run to apply them.")


# ------------------------------------------------------------------ figures (cached in session_state)
def view_dict(window):
    return dict(window=window, cx=float(ss.cx), cy=float(ss.cy), half=float(ss.half_width), grid=int(ss.grid),
                log_z=bool(ss.log_z), log_loss=bool(ss.log_loss), log_norm=bool(ss.log_norm))


def play_html(run):
    """(html, height) of the player; rebuilt only when the run, the window/grid/log options or the pin change."""
    view = view_dict(ss.window)
    sig = (id(run), tuple(view.values()), id(ss.ghost))
    cached = ss.figs.get("play")
    if cached is None or cached[0] != sig:
        fig, info = build_figure(run, view, ss.ghost, grid_fn=grid_fn(ss.run_cfg))
        ss.figs["play"] = cached = (sig, player_html(fig, info, run.iteration))
    return cached[1]


def show_preview(run):
    view = view_dict("fixed")
    sig = (id(run), tuple(view.values()), ss.click_n, id(ss.ghost))
    cached = ss.figs.get("preview")
    if cached is None or cached[0] != sig:
        fig, info = build_figure(run, view, ghost=ss.ghost, click=True, grid_fn=grid_fn(cfg))
        ss.figs["preview"] = cached = (sig, fig, info)
    plot(cached[1], f"preview_{ss.click_n}", on_select=on_click, selection_mode="points")
    st.caption("Press Run to compute the optimizers, or click the contour to set the start point."
               + (" The pinned run is shown dotted." if ss.ghost else ""))


def show_picker(run):
    """After a run: choose the next start by clicking, with the finished and the pinned run drawn for reference."""
    if not st.checkbox("Pick the next start point by clicking the contour", key="pick_on"):
        return
    view = view_dict(ss.window)
    new_start = (float(ss.start_x), float(ss.start_y))
    sig = (id(run), tuple(view.values()), id(ss.ghost), ss.click_n, new_start)
    cached = ss.figs.get("pick")
    if cached is None or cached[0] != sig:
        fig, click_idx = build_picker(run, view, ss.ghost, new_start, grid_fn=grid_fn(ss.run_cfg))
        ss.figs["pick"] = cached = (sig, fig, click_idx)
    plot(cached[1], f"pick_{ss.click_n}", on_select=on_pick, selection_mode="points")
    st.caption("Click to set the start point, then press Run. The solid paths are the run above; dotted paths are the "
               "pinned run (use Pin run first to keep it for comparison). The star marks the next start.")


def show_convergence(run):
    c = st.columns(2)
    c[0].selectbox("Loss gap tolerance", LOSS_TOLS, key="tol_loss", format_func=lambda v: f"loss - best <= {v:g}")
    c[1].selectbox("Gradient-norm tolerance", GNORM_TOLS, key="tol_gnorm", format_func=lambda v: f"||grad|| <= {v:g}")
    rows, best, known = convergence_rows(run, float(ss.tol_loss), float(ss.tol_gnorm))
    st.dataframe([{"Optimizer": r["name"],
                   "iterations to loss gap": "not reached" if r["loss_iter"] is None else r["loss_iter"],
                   "iterations to gradient norm": "not reached" if r["gnorm_iter"] is None else r["gnorm_iter"],
                   "final loss gap": f"{r['final_gap']:.3g}", "final gradient norm": f"{r['final_gnorm']:.3g}",
                   "status": "stopped (non-finite)" if r["stopped"] else "finished"} for r in rows],
                 hide_index=True, width="stretch")
    fig = go.Figure()
    for r in rows:
        reached = r["loss_iter"] is not None
        fig.add_bar(x=[r["name"]], y=[r["loss_iter"] if reached else run.max_iters], name=r["name"],
                    marker=dict(color=OPTIMIZERS[r["name"]].color, opacity=1.0 if reached else 0.3),
                    text=[str(r["loss_iter"]) if reached else "not reached"], textposition="outside", showlegend=False)
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=40, r=20, t=30, b=40),
                      title=dict(text=f"Iterations until loss - best <= {float(ss.tol_loss):g}", font=dict(size=13)),
                      yaxis=dict(title="iterations", range=[0, run.max_iters * 1.15]))
    plot(fig, "convergence_bar")
    st.caption(("best = loss at the best known minimum (full data)." if known else
                "No minimum is known here, so best = the lowest loss seen in this run.") +
               " With minibatches the measured gradient norm is on the full data and may never get small. "
               "A faded bar means the tolerance was not reached within the maximum iterations.")


if ss.run is None:
    if cfg is not None:
        try:
            show_preview(S.build_run(replace(cfg, chosen={}), OBJ, ALLOW_CUSTOM, need_optimizers=False))
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
else:
    html = play_html(ss.run)
    with b[4]:
        st.download_button("Download CSV", history_csv(ss.run), "optimizer_run.csv", "text/csv", width="stretch")
    with b[5]:
        st.download_button("Download page", player_page(html[0]), "optimizer_run.html", "text/html", width="stretch")
    dash, conv = st.tabs(["Dashboard", "Convergence"])
    with dash:
        st.caption("Press Play or drag the slider: the browser replays the stored run, so nothing is recomputed. "
                   "The contour, surface and axes stay fixed while the paths grow.")
        embed(*html)
        show_picker(ss.run)
    with conv:
        show_convergence(ss.run)
