"""Core numerics: gradients, optimizer steps, runner, objective grids. Run: python -m pytest"""

import numpy as np
import pytest
from autograd import grad

from opt2d import RunConfig, Run, get_objectives, OPTIMIZERS
from opt2d import settings as S
from opt2d.figures import grid_values, build_figure
from opt2d.player import player_html
from opt2d.objectives import CUSTOM_NAME, make_custom_objective, CUSTOM_TEMPLATE

OBJ = get_objectives()


def fd_grad(f, w, h=1e-6):
    return np.array([(f(w + h * e) - f(w - h * e)) / (2 * h) for e in np.eye(2)])


@pytest.mark.parametrize("name", list(OBJ))
def test_gradient_matches_finite_differences(name):
    o = OBJ[name]
    data = o.make_data(200, np.random.default_rng(0)) if o.make_data else None
    w = np.array(o.start) + 0.1
    f = lambda v: float(o.loss(v, data))
    assert np.allclose(grad(o.loss)(w, data), fd_grad(f, w), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", list(OBJ))
def test_known_optima_have_zero_gradient(name):
    o = OBJ[name]
    if o.make_data:
        return
    for p in o.optima:
        assert np.linalg.norm(grad(o.loss)(np.array(p), None)) < 1e-3


def test_sgd_momentum_by_hand():
    w, g = np.array([1.0, 2.0]), np.array([0.5, -1.0])
    s = OPTIMIZERS["SGD + momentum"].init(w)
    out = OPTIMIZERS["SGD + momentum"].step(w, g, s, 0.1, momentum=0.9, nesterov=False, convention="classic")
    assert np.allclose(out, w - 0.1 * g)                      # first classic step: m = g
    out2 = OPTIMIZERS["SGD + momentum"].step(out, g, s, 0.1, momentum=0.9, nesterov=False, convention="classic")
    assert np.allclose(out2, out - 0.1 * (0.9 * g + g))


def test_adagrad_rmsprop_adam_first_step():
    w, g = np.array([0.0, 0.0]), np.array([2.0, -4.0])
    ada = OPTIMIZERS["AdaGrad"]
    assert np.allclose(ada.step(w, g, ada.init(w), 0.1, eps=0.0), -0.1 * np.sign(g))
    rms = OPTIMIZERS["RMSProp"]
    assert np.allclose(rms.step(w, g, rms.init(w), 0.1, rho=0.9, eps=0.0), -0.1 * g / np.sqrt(0.1 * g * g))
    adam = OPTIMIZERS["Adam"]
    assert np.allclose(adam.step(w, g, adam.init(w), 0.1, beta1=0.9, beta2=0.999, eps=0.0), -0.1 * np.sign(g))


def test_run_decreases_loss_on_every_default_objective():
    for name, o in OBJ.items():
        v = S.defaults(OBJ)
        v["objective"] = name
        S.load_objective(v, name, OBJ)
        run = S.build_run(S.config_from(v), OBJ)
        run.run()
        assert len(run.history["Adam"]["loss"]) == run.max_iters + 1
        assert min(run.history["Adam"]["loss"]) < run.loss0, name


def test_minibatch_is_shared_and_metrics_use_full_data():
    o = OBJ["Least squares (data, minibatch)"]
    every = S.chosen_from(S.defaults(OBJ))
    chosen = {n: every[n] for n in ("SGD + momentum", "Adam")}
    a = Run(o, chosen, np.array(o.start), lr=0.05, max_iters=5, n_samples=100, batch_size=10, seed=3)
    b = Run(o, {"Adam": every["Adam"]}, np.array(o.start), lr=0.05, max_iters=5, n_samples=100, batch_size=10, seed=3)
    a.run(); b.run()
    assert np.allclose(a.history["Adam"]["loss"], b.history["Adam"]["loss"])        # same batches -> same path


def test_divergence_is_reported_not_raised():
    o = OBJ["Rosenbrock (banana valley)"]
    run = Run(o, {"SGD + momentum": S.chosen_from(S.defaults(OBJ))["SGD + momentum"]}, np.array(o.start), lr=5.0, max_iters=60)
    run.run()
    assert not run.alive["SGD + momentum"] and run.messages


def test_custom_objective_and_errors():
    o = make_custom_objective(CUSTOM_TEMPLATE, (0.0, 0.0), (0, 0), 2.0, 0.01)
    assert abs(float(o.loss(np.array([1.0, 1.0])))) < 1e-12
    with pytest.raises(ValueError):
        make_custom_objective("x = 1", (0.0, 0.0), (0, 0), 2.0, 0.01)
    with pytest.raises(ValueError):
        make_custom_objective("def loss(w):\n    return w[0] / 0.0", (0.0, 0.0), (0, 0), 2.0, 0.01)


def test_custom_objective_not_allowed_by_default():
    v = S.defaults(OBJ)
    v["objective"] = CUSTOM_NAME
    with pytest.raises(ValueError, match="switched off"):
        S.build_run(S.config_from(v), OBJ, allow_custom=False)
    S.build_run(S.config_from(v), OBJ, allow_custom=True)


def test_grid_vectorised_matches_loop_and_data_fallback():
    xs = ys = np.linspace(-1, 1, 7)
    for name in ("Rosenbrock (banana valley)", "Least squares (data, minibatch)"):
        o = OBJ[name]
        data = o.make_data(50, np.random.default_rng(0)) if o.make_data else None
        Z = grid_values(o, data, xs, ys)
        ref = np.array([[float(o.loss(np.array([x, y]), data)) for x in xs] for y in ys])
        assert np.allclose(Z, ref)


def test_figure_and_player_build_for_every_objective_with_ghost():
    for name, o in OBJ.items():
        v = S.defaults(OBJ)
        v["objective"] = name
        S.load_objective(v, name, OBJ)
        run = S.build_run(S.config_from(v), OBJ)
        run.run(30)
        view = dict(window="fit", cx=v["cx"], cy=v["cy"], half=v["half_width"], grid=21, log_z=o.log_z,
                    log_loss=True, log_norm=True)
        fig, info = build_figure(run, view, ghost=S.pin_from(run))
        html, height = player_html(fig, info, run.iteration, tick=5, trail=10)
        assert "Plotly" in html and height > 800
