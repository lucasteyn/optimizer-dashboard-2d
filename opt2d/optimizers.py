"""Optimizers as small functions. No classes, no inheritance.

Each optimizer is an ``Optimizer`` record with
    init(w)                         -> state dict (arrays that persist between steps)
    step(w, g, state, lr, **params) -> new w   (updates ``state`` in place)
    params                          -> default hyper-parameters; the dashboard builds its boxes from this

A default that is a bool/int/float gives a checkbox / integer box / float box; a list gives a dropdown whose
first entry is the default.

To add an optimizer: write init + step, add an ``Optimizer(...)`` to ``OPTIMIZERS``.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Optimizer:
    name: str
    init: Callable
    step: Callable
    params: dict
    color: str


# ---------------------------------------------------------------- SGD with momentum

def sgd_init(w):
    return {"m": np.zeros_like(w)}


def sgd_step(w, g, s, lr, momentum, nesterov, convention):
    if convention == "ema":                          # m = b m + (1-b) g
        s["m"] = momentum * s["m"] + (1 - momentum) * g
        d = (1 - momentum) * g + momentum * s["m"] if nesterov else s["m"]
    else:                                            # classical heavy ball: m = b m + g
        s["m"] = momentum * s["m"] + g
        d = g + momentum * s["m"] if nesterov else s["m"]
    return w - lr * d


# ---------------------------------------------------------------- AdaGrad

def adagrad_init(w):
    return {"sum_sq": np.zeros_like(w)}


def adagrad_step(w, g, s, lr, eps):
    s["sum_sq"] = s["sum_sq"] + g * g
    return w - lr * g / (np.sqrt(s["sum_sq"]) + eps)


# ---------------------------------------------------------------- RMSProp

def rmsprop_init(w):
    return {"v": np.zeros_like(w)}


def rmsprop_step(w, g, s, lr, rho, eps):
    s["v"] = rho * s["v"] + (1 - rho) * g * g
    return w - lr * g / (np.sqrt(s["v"]) + eps)


# ---------------------------------------------------------------- Adam

def adam_init(w):
    return {"m": np.zeros_like(w), "v": np.zeros_like(w), "t": 0}


def adam_step(w, g, s, lr, beta1, beta2, eps):
    s["t"] += 1
    s["m"] = beta1 * s["m"] + (1 - beta1) * g
    s["v"] = beta2 * s["v"] + (1 - beta2) * g * g
    m_hat = s["m"] / (1 - beta1 ** s["t"])
    v_hat = s["v"] / (1 - beta2 ** s["t"])
    return w - lr * m_hat / (np.sqrt(v_hat) + eps)


# ---------------------------------------------------------------- registry

OPTIMIZERS = {
    "SGD + momentum": Optimizer("SGD + momentum", sgd_init, sgd_step,
                                {"momentum": 0.9, "nesterov": False, "convention": ["ema", "classic"]}, "#0072B2"),
    "AdaGrad": Optimizer("AdaGrad", adagrad_init, adagrad_step, {"eps": 1e-8}, "#E69F00"),
    "RMSProp": Optimizer("RMSProp", rmsprop_init, rmsprop_step, {"rho": 0.9, "eps": 1e-8}, "#009E73"),
    "Adam": Optimizer("Adam", adam_init, adam_step, {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}, "#D55E00"),
}
