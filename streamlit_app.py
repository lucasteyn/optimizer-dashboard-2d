"""Streamlit front end of the 2-D optimizer dashboard.  Run:  streamlit run streamlit_app.py

Streamlit re-runs this script on every interaction, so all state lives in ``st.session_state``: the control values
(see opt2d/settings.py), the computed ``Run`` and the pinned ghost run.
Run computes every iteration at once; the 2x2 figure then replays it in the browser (opt2d/player.py), so playing
causes no reruns and no flicker, and the contour, surface and axes stay fixed.
"""

import os
from dataclasses import replace

import streamlit as st
import streamlit.components.v1 as components

from opt2d import OPTIMIZERS, get_objectives
from opt2d import settings as S
from opt2d.export import history_csv
from opt2d.figures import PLOT_CONFIG, build_figure
from opt2d.objectives import CUSTOM_NAME
from opt2d.player import player_html
from opt2d.schedules import SCHEDULES, sparkline_svg

# Custom objectives execute user code: only switch them on for a local / trusted app (ALLOW_CUSTOM=1).
ALLOW_CUSTOM = os.environ.get("ALLOW_CUSTOM", "") == "1"
FRAME_MS = {"Slow": 250, "Normal": 100, "Fast": 40, "Very fast": 15}      # browser animation: ms per frame
WINDOWS = {"fixed": "Fixed window (centre and half-width below)", "fit": "Fit to the paths after a run"}

st.set_page_config(page_title="2-D optimizer dashboard", layout="wide")
ss = st.session_state


@st.cache_resource
def objectives():
    return get_objectives()


OBJ = objectives()
if "init" not in ss:
    ss.update(S.defaults(OBJ))
    ss.update(init=True, run=None, run_cfg=None, ghost=None, figs={}, error="", click_n=0)


# ------------------------------------------------------------------ callbacks (run before the script re-runs)
def forget_run():
    ss.run, ss.figs, ss.error = None, {}, ""


def on_objective():
    S.load_objective(ss, ss.objective, OBJ)
    forget_run()


def restore(name):
    ss.update(S.optimizer_defaults(name, OBJ[ss.objective].own_lr.get(name) if ss.objective in OBJ else None))


def on_pin():
    ss.ghost = S.pin_from(ss.run, f"{ss.objective}; ")


def on_unpin():
    ss.ghost = None


def on_click():
    """A click on the contour sets the start point (before a run)."""
    ev = ss.get(f"preview_{ss.click_n}")
    points = (ev or {}).get("selection", {}).get("points", []) if ev else []
    cached = ss.figs.get("preview")
    if not points or cached is None:
        return
    pt = points[0]
    if pt.get("curve_number") == cached[2]["click"]:
        ss.start_x, ss.start_y = round(float(pt["x"]), 3), round(float(pt["y"]), 3)
        ss.click_n += 1
        ss.figs = {}


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
    c[0].number_input("N samples", 10, 100000, step=50, key="n", disabled=not has_data)
    c[1].number_input("Batch m", 1, 100000, step=8, key="m", disabled=not has_data)
    c[2].number_input("Seed", 0, 10**6, key="seed", disabled=not has_data)
    st.markdown("**Start point** (or click the contour before a run)")
    c = st.columns(2)
    c[0].number_input("x", step=0.1, format="%.3f", key="start_x")
    c[1].number_input("y", step=0.1, format="%.3f", key="start_y")

    st.header("Optimizers")
    for name, opt in OPTIMIZERS.items():
        st.checkbox(name, key=S.use_key(name))
        with st.expander("settings"):
            st.text_input("own learning rate (blank = global)", key=S.lr_key(name))
            for p, default in opt.params.items():
                key, kind = S.param_key(name, p), S.param_kind(default)
                if kind == "bool":
                    st.checkbox(p, key=key)
                elif kind == "int":
                    st.number_input(p, 0, 1000, step=1, key=key)
                elif kind == "float":
                    st.text_input(p, key=key)
                else:
                    st.selectbox(p, default, key=key)
            st.button("Restore defaults", key=f"restore_{name}", on_click=restore, args=(name,))

    st.header("Run")
    st.text_input("Global learning rate", key="lr")
    st.number_input("Max iterations", 1, 20000, step=10, key="iters")
    labels = dict((v, k) for k, v in SCHEDULES)
    st.selectbox("Learning-rate schedule", list(labels), format_func=labels.get, key="schedule")
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
        st.number_input("Trail length (0 = whole path)", 0, 20000, step=10, key="trail")
        st.number_input("Mark every k-th iteration (0 = off)", 0, 1000, step=5, key="tick")


with st.sidebar:
    sidebar()

# ------------------------------------------------------------------ main area: buttons and run
st.title("2-D optimizer dashboard")

try:
    cfg, cfg_error = S.config_from(ss), ""
except ValueError as e:
    cfg, cfg_error = None, str(e)

b = st.columns([1, 1, 1, 1, 1.4, 1.4])
run_clicked = b[0].button("Run", type="primary", width="stretch", disabled=cfg is None)
b[1].button("Reset", on_click=forget_run, width="stretch")
b[2].button("Pin run", on_click=on_pin, disabled=ss.run is None, width="stretch")
b[3].button("Clear pin", on_click=on_unpin, disabled=ss.ghost is None, width="stretch")
b[4].selectbox("Play speed", list(FRAME_MS), key="speed", label_visibility="collapsed")

if run_clicked:
    try:
        run = S.build_run(cfg, OBJ, ALLOW_CUSTOM)
        bar = st.progress(0.0, text="Computing all iterations...")
        for k in range(run.max_iters):
            run.step()
            if k % 10 == 0:
                bar.progress((k + 1) / run.max_iters, text=f"Computing iteration {k + 1} of {run.max_iters}")
        bar.empty()
        ss.update(run=run, run_cfg=cfg, figs={}, error="")
        run_ok = True
    except Exception as e:                              # bad input or user code: show it, keep the app alive
        ss.error = f"{type(e).__name__}: {e}"
        run_ok = False
    if run_ok:
        st.rerun()                                      # redraw the buttons (Pin, ...) now that a run exists
with b[5]:
    if ss.run is not None:
        st.download_button("Download CSV", history_csv(ss.run), "optimizer_run.csv", "text/csv", width="stretch")

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


def show_run(run):
    view = view_dict(ss.window)
    sig = (id(run), tuple(view.values()), id(ss.ghost), int(ss.trail), int(ss.tick), ss.speed)
    cached = ss.figs.get("play")
    if cached is None or cached[0] != sig:
        fig, info = build_figure(run, view, ss.ghost)
        html = player_html(fig, info, run.iteration, FRAME_MS[ss.speed], int(ss.trail), int(ss.tick))
        ss.figs["play"] = cached = (sig, html)
    st.caption("Press Play or drag the slider: the browser replays the stored run, so nothing is recomputed. "
               "The contour, surface and axes stay fixed while the paths grow.")
    embed(*cached[1])


def show_preview(run):
    view = view_dict("fixed")
    sig = (id(run), tuple(view.values()), ss.click_n)
    cached = ss.figs.get("preview")
    if cached is None or cached[0] != sig:
        fig, info = build_figure(run, view, click=True)
        ss.figs["preview"] = cached = (sig, fig, info)
    plot(cached[1], f"preview_{ss.click_n}", on_select=on_click, selection_mode="points")
    st.caption("Press Run to compute the optimizers, or click the contour to set the start point.")


def status_table(run):
    rows = []
    for name in run.names:
        h = run.history[name]
        end = len(h["loss"]) - 1
        rows.append({"Optimizer": name, "iteration": end, "loss": f"{h['loss'][end]:.5g}",
                     "grad norm": f"{h['gnorm'][end]:.4g}", "point": f"({h['w'][end][0]:.4g}, {h['w'][end][1]:.4g})",
                     "status": "finished" if run.alive[name] else "stopped (non-finite)"})
    st.markdown(f"**Final values** (iteration {run.iteration})")
    st.dataframe(rows, hide_index=True, width="stretch")


if ss.run is None:
    if cfg is not None:
        try:
            show_preview(S.build_run(replace(cfg, chosen={}), OBJ, ALLOW_CUSTOM, need_optimizers=False))
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
else:
    show_run(ss.run)
    status_table(ss.run)
