"""Objective functions of a 2-D point w = (x, y).

Every loss has the signature ``loss(w, data)`` and uses ``autograd.numpy`` so that ``autograd.grad`` can
differentiate it. ``data`` is None for analytic objectives, or a tuple of arrays (rows = samples) for the
least-squares objective. Write ``w[0]`` / ``w[1]`` (or ``x, y = w``) so that the same code also works on a whole grid
of points at once (used to draw the contours quickly).

To add an objective: write a loss and append an ``Objective(...)`` in ``get_objectives``.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import autograd.numpy as np
import numpy as onp  # plain numpy for data generation


@dataclass
class Objective:
    name: str
    description: str
    loss: Callable                          # loss(w, data) -> scalar
    start: tuple                            # default starting point
    center: tuple                           # default centre of the plotted window
    half_width: float = 3.0                 # default half-width of the plotted window
    lr: float = 0.05                        # default global learning rate
    own_lr: dict = field(default_factory=dict)   # optional per-optimizer learning rates
    iters: int = 200
    optima: tuple = ()                      # known minimisers, drawn as x markers
    log_z: bool = False                     # colour contours / surface by log10(loss - min loss)
    make_data: Optional[Callable] = None    # make_data(N, rng) -> tuple of arrays


def subset(data, idx):
    """Rows ``idx`` of every array in ``data`` (a minibatch)."""
    return None if data is None else tuple(a[idx] for a in data)


# ---------------------------------------------------------------- losses

def bowl(w, data=None):
    return 0.5 * ((w[0] - 1.0) ** 2 + (w[1] + 0.5) ** 2)


def ill_conditioned(w, data=None):
    return 0.5 * (w[0] ** 2 + 25.0 * w[1] ** 2)


def rotated(w, data=None):
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    u, v = c * w[0] + s * w[1], -s * w[0] + c * w[1]
    return 0.5 * (25.0 * u ** 2 + v ** 2)


def rosenbrock(w, data=None):
    return 100.0 * (w[1] - w[0] ** 2) ** 2 + (1.0 - w[0]) ** 2


def himmelblau(w, data=None):
    return (w[0] ** 2 + w[1] - 11.0) ** 2 + (w[0] + w[1] ** 2 - 7.0) ** 2


def beale(w, data=None):
    x, y = w[0], w[1]
    return ((1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2)


def rastrigin(w, data=None):
    x, y = w[0], w[1]
    return 20.0 + x ** 2 + y ** 2 - 10.0 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))


W_TRUE = onp.array([1.5, -1.0])
INPUT_MIX = onp.array([[2.0, 0.0], [1.2, 0.4]])      # correlated features with unequal scales


def least_squares(w, data):
    """Mean squared error of the linear model y ~ X w on the samples in ``data``."""
    X, y = data
    r = np.dot(X, w) - y
    return 0.5 * np.mean(r * r)


def make_regression_data(N, rng):
    X = rng.standard_normal((N, 2)) @ INPUT_MIX
    return X, X @ W_TRUE + 0.3 * rng.standard_normal(N)


# ---------------------------------------------------------------- registry

def get_objectives():
    """Ordered dict name -> Objective."""
    objs = [
        Objective("Quadratic bowl", "0.5 ((x-1)^2 + (y+0.5)^2): equal curvature in every direction.",
                  bowl, start=(-2.0, 2.0), center=(1.0, -0.5), half_width=3.5, lr=0.3, optima=((1.0, -0.5),)),
        Objective("Ill-conditioned quadratic", "0.5 (x^2 + 25 y^2): a narrow valley, curvature ratio 25.",
                  ill_conditioned, start=(-3.0, 1.0), center=(0.0, 0.0), half_width=3.5, lr=0.03,
                  own_lr={"AdaGrad": 0.5, "RMSProp": 0.05, "Adam": 0.1}, optima=((0.0, 0.0),)),
        Objective("Rotated ill-conditioned quadratic", "The same valley rotated by 30 degrees, so axis-wise (adaptive) "
                  "scaling cannot remove the ill-conditioning.",
                  rotated, start=(-2.5, 2.0), center=(0.0, 0.0), half_width=3.0, lr=0.03,
                  own_lr={"AdaGrad": 0.5, "RMSProp": 0.05, "Adam": 0.1}, optima=((0.0, 0.0),)),
        Objective("Rosenbrock (banana valley)", "100 (y - x^2)^2 + (1 - x)^2, minimum at (1, 1). Colours are log scaled.",
                  rosenbrock, start=(-1.2, 1.0), center=(0.3, 0.5), half_width=2.2, lr=0.002, iters=800,
                  own_lr={"AdaGrad": 0.3, "RMSProp": 0.01, "Adam": 0.05}, optima=((1.0, 1.0),), log_z=True),
        Objective("Himmelblau (four minima)", "(x^2 + y - 11)^2 + (x + y^2 - 7)^2: four equal minima, where you end "
                  "depends on the optimizer and the start.",
                  himmelblau, start=(0.0, 0.0), center=(0.0, 0.0), half_width=5.0, lr=0.01,
                  own_lr={"AdaGrad": 0.5, "RMSProp": 0.05, "Adam": 0.1},
                  optima=((3.0, 2.0), (-2.805118, 3.131312), (-3.77931, -3.283186), (3.584428, -1.848126)), log_z=True),
        Objective("Beale", "A flat plateau with a sharp valley, minimum at (3, 0.5).",
                  beale, start=(1.0, 1.0), center=(1.5, 0.5), half_width=3.0, lr=0.004, iters=600,
                  own_lr={"AdaGrad": 0.5, "RMSProp": 0.02, "Adam": 0.05}, optima=((3.0, 0.5),), log_z=True),
        Objective("Rastrigin (many local minima)", "20 + x^2 + y^2 - 10 (cos 2 pi x + cos 2 pi y): the global minimum "
                  "at the origin is surrounded by local ones.",
                  rastrigin, start=(2.2, 2.7), center=(0.0, 0.0), half_width=3.5, lr=0.01,
                  own_lr={"AdaGrad": 0.3, "RMSProp": 0.02, "Adam": 0.05}, optima=((0.0, 0.0),)),
        Objective("Least squares (data, minibatch)", "Linear regression y = X w + noise with correlated features. "
                  "N and the minibatch size m apply here.",
                  least_squares, start=(-2.0, 2.0), center=tuple(W_TRUE), half_width=3.5, lr=0.05,
                  own_lr={"AdaGrad": 0.5, "RMSProp": 0.03, "Adam": 0.1}, optima=(tuple(W_TRUE),),
                  make_data=make_regression_data),
    ]
    return {o.name: o for o in objs}


# ---------------------------------------------------------------- custom objective

CUSTOM_NAME = "Custom (write your own)"

CUSTOM_TEMPLATE = '''# w = (x, y).  `np` is autograd.numpy: use np.sin, np.exp, np.sum, ...
def loss(w):
    x, y = w[0], w[1]
    return (1 - x) ** 2 + 5 * (y - x ** 2) ** 2
'''


def make_custom_objective(code, start, center, half_width, lr):
    """Build an Objective from user code defining ``loss(w)`` (``objective(w)`` also works)."""
    namespace = {"np": np}
    exec(code, namespace)
    user_loss = namespace.get("loss") or namespace.get("objective")
    if not callable(user_loss):
        raise ValueError("The code must define loss(w) returning a scalar.")

    def loss(w, data=None):
        return user_loss(w)

    value = loss(onp.array(start, dtype=float))          # fail early with a clear error
    if not onp.isfinite(float(value)):
        raise ValueError("loss(start) is not finite")
    return Objective(CUSTOM_NAME, "User-defined loss(w).", loss, start=tuple(start), center=tuple(center),
                     half_width=half_width, lr=lr)
