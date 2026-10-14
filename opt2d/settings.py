"""The control panel as plain data (no UI library).

The front end keeps one flat dict of widget values (for Streamlit: ``st.session_state``) under the keys made here.
``defaults()`` fills it, ``load_objective`` changes it the way choosing an objective does, ``config_from`` turns it
into a ``RunConfig`` and ``build_run`` into a ``Run`` (validated).

Float hyper-parameters are kept as text (so 1e-8 stays readable); ``config_from`` parses them.
"""

import math

import numpy as np

from .config import RunConfig, parse_lr
from .lessons import LESSONS
from .objectives import CUSTOM_NAME, CUSTOM_TEMPLATE, get_objectives, make_custom_objective
from .optimizers import OPTIMIZERS
from .runner import Run


MANUAL = "Manual / modified"


def use_key(name): return f"use::{name}"
def lr_key(name): return f"lr::{name}"
def param_key(name, p): return f"p::{name}::{p}"


def _as_value(default):
    """Default of a parameter -> value stored in the widget dict (floats as text, dropdowns as their first entry)."""
    if isinstance(default, list):
        return default[0]
    if isinstance(default, (bool, int)):
        return default
    return f"{default:g}"


def param_kind(default):
    """'bool', 'int', 'float' or 'choice' (what widget a parameter needs)."""
    if isinstance(default, list):
        return "choice"
    if isinstance(default, bool):
        return "bool"
    return "int" if isinstance(default, int) else "float"


def optimizer_defaults(name, own_lr=None):
    """Widget values that restore one optimizer card to its defaults (``own_lr``: objective's suggestion)."""
    values = {lr_key(name): "" if own_lr is None else f"{own_lr:g}"}
    values.update({param_key(name, p): _as_value(v) for p, v in OPTIMIZERS[name].params.items()})
    return values


def load_objective(values, objective, objectives):
    """What choosing an objective changes: start, window, learning rates, iterations, colour scale."""
    if objective == CUSTOM_NAME:
        values["code"] = values.get("code") or CUSTOM_TEMPLATE
        return
    obj = objectives[objective]
    values.update(start_x=float(obj.start[0]), start_y=float(obj.start[1]), cx=float(obj.center[0]),
                  cy=float(obj.center[1]), half_width=float(obj.half_width), lr=f"{obj.lr:g}", iters=obj.iters,
                  log_z=obj.log_z)
    for name in OPTIMIZERS:
        values[lr_key(name)] = "" if name not in obj.own_lr else f"{obj.own_lr[name]:g}"


def apply_lesson(values, name, objectives):
    """Load a guided example into the widget dict."""
    lesson = next(L for L in LESSONS if L.name == name)
    values["objective"] = lesson.objective
    load_objective(values, lesson.objective, objectives)
    obj = objectives[lesson.objective]
    start = lesson.start or obj.start
    values.update(start_x=float(start[0]), start_y=float(start[1]), lr=f"{lesson.lr:g}", iters=lesson.iters)
    if lesson.n:
        values["n"] = lesson.n
    if lesson.m:
        values["m"] = lesson.m
    for opt in OPTIMIZERS:
        values[use_key(opt)] = opt in lesson.optimizers
        values.update(optimizer_defaults(opt, lesson.own_lr.get(opt)))
        for p, v in lesson.optimizers.get(opt, {}).items():
            values[param_key(opt, p)] = v if isinstance(v, (str, bool, int)) and not isinstance(v, float) else f"{v:g}"
    values["lesson"] = name


def lesson_ghost_config(cfg, name):
    """Config of the lesson's baseline run (or None): same settings, some optimizer parameters overridden."""
    lesson = next((L for L in LESSONS if L.name == name), None)
    if lesson is None or not lesson.ghost:
        return None, ""
    chosen = {n: {**cfg.chosen[n], **ov} for n, ov in lesson.ghost.items() if n in cfg.chosen}
    if not chosen:
        return None, ""
    from dataclasses import replace
    return replace(cfg, chosen=chosen, m=lesson.ghost_m or cfg.m), lesson.ghost_label


def defaults(objectives=None):
    """A complete widget dict: the first objective plus every view/option default."""
    objectives = objectives or get_objectives()
    values = {"n": 500, "m": 32, "seed": 0, "code": CUSTOM_TEMPLATE, "schedule": "constant", "objective": next(iter(objectives)),
              "window": "fixed", "grid": 41, "log_loss": True, "log_norm": True, "lesson": MANUAL,
              "tol_loss": 1e-3, "tol_gnorm": 1e-2}
    for name in OPTIMIZERS:
        values[use_key(name)] = True
        values.update(optimizer_defaults(name))
    load_objective(values, values["objective"], objectives)
    return values


# ---------------------------------------------------------------- widget dict -> RunConfig -> Run
def chosen_from(values):
    chosen = {}
    for name, opt in OPTIMIZERS.items():
        if not values.get(use_key(name)):
            continue
        params = {}
        for p, default in opt.params.items():
            v = values[param_key(name, p)]
            if param_kind(default) == "float":
                try:
                    v = float(v)
                except ValueError:
                    raise ValueError(f"{name}: '{p}' must be a number (got '{v}')") from None
            params[p] = v
        own = parse_lr(values.get(lr_key(name), ""))
        if own is not None:
            params["lr"] = own
        chosen[name] = params
    return chosen


def config_from(values):
    objective = values["objective"]
    try:
        lr = float(values["lr"])
    except ValueError:
        raise ValueError("the global learning rate must be a number") from None
    return RunConfig(objective=objective, code=values.get("code", "") if objective == CUSTOM_NAME else "",
                     start=(float(values["start_x"]), float(values["start_y"])), lr=lr, iters=int(values["iters"]),
                     n=int(values["n"]), m=int(values["m"]), seed=int(values["seed"]), chosen=chosen_from(values),
                     schedule=values["schedule"],
                     window=(float(values["cx"]), float(values["cy"]), float(values["half_width"])))


def objective_for(cfg, objectives, allow_custom=False):
    if cfg.objective == CUSTOM_NAME:
        if not allow_custom:
            raise ValueError("custom objectives are switched off on this server (start the app with ALLOW_CUSTOM=1)")
        return make_custom_objective(cfg.code, cfg.start, cfg.window[:2], cfg.window[2], cfg.lr)
    return objectives[cfg.objective]


def build_run(cfg, objectives, allow_custom=False, need_optimizers=True):
    """Validated ``Run`` for a config (raises ValueError with a readable message)."""
    if need_optimizers:
        cfg.validate()
    elif not (math.isfinite(cfg.lr) and cfg.lr > 0):
        raise ValueError("the global learning rate must be a positive number")
    objective = objective_for(cfg, objectives, allow_custom)
    w0 = np.array(cfg.start, dtype=float)
    if not np.all(np.isfinite(w0)):
        raise ValueError("the start point must be finite")
    if cfg.window[2] <= 0:
        raise ValueError("the window half-width must be positive")
    data = objective.make_data(max(2, cfg.n), np.random.default_rng(0)) if objective.make_data else None
    if not np.isfinite(float(objective.loss(w0, data))):
        raise ValueError("the loss at the start point is not finite")
    return Run(objective, cfg.chosen, w0, lr=cfg.lr, max_iters=cfg.iters, n_samples=cfg.n, batch_size=cfg.m,
               seed=cfg.seed, schedule=cfg.schedule)


def pin_from(run, label_prefix=""):
    """A faded 'ghost' copy of a finished run (used by the figure)."""
    rates = ", ".join(f"{n} {run.rate(n):g}" for n in run.names)
    return {"names": list(run.names),
            "label": f"{label_prefix}lr {rates}; {run.schedule}; {run.iteration} iterations",
            "w": {n: np.array(run.history[n]["w"]) for n in run.names},
            "loss": {n: list(run.history[n]["loss"]) for n in run.names},
            "gnorm": {n: list(run.history[n]["gnorm"]) for n in run.names}}
