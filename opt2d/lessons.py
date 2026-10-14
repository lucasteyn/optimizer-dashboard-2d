"""Guided examples: each sets the objective, start, learning rates and optimizers, and says what to look for.

``optimizers`` maps the optimizers that are switched on to parameter overrides ({} = defaults).
``ghost`` (optional) maps optimizers to parameter overrides for a BASELINE run that Run computes and pins automatically
(dotted, faded) so the effect of the change is visible at once.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Lesson:
    name: str
    objective: str
    note: str
    optimizers: dict
    lr: float
    own_lr: dict = field(default_factory=dict)       # optimizer -> own learning rate
    iters: int = 150
    start: Optional[tuple] = None
    ghost: Optional[dict] = None
    ghost_label: str = ""
    n: Optional[int] = None
    m: Optional[int] = None
    ghost_m: Optional[int] = None                    # minibatch size of the baseline run (default: same as the live run)


ALL = {"SGD + momentum": {}, "AdaGrad": {}, "RMSProp": {}, "Adam": {}}

LESSONS = [
    Lesson("Momentum in a narrow valley", "Ill-conditioned quadratic",
           "Solid: SGD with classical momentum 0.9. Dotted: the same learning rate with momentum 0. In a narrow valley "
           "the gradient points mostly across the valley and only weakly along it. The momentum buffer cancels the "
           "alternating cross-valley components and accumulates the consistent along-valley ones, so it travels about "
           "ten times faster along the valley (compare the loss curves).",
           {"SGD + momentum": {"momentum": 0.9, "convention": "classic"}}, lr=0.03, iters=60,
           ghost={"SGD + momentum": {"momentum": 0.0}}, ghost_label="no momentum: "),
    Lesson("Adaptive steps fix an axis-aligned valley", "Ill-conditioned quadratic",
           "The valley is aligned with the axes, so dividing each coordinate by its own gradient scale (AdaGrad, RMSProp, "
           "Adam) makes the problem look well conditioned: they head almost straight to the minimum, while plain SGD "
           "must keep its learning rate below 2/25 for the steep direction and so crawls along the flat one. Open the internals table: the per-coordinate "
           "step/|g| differs between x and y for the adaptive methods but not for SGD.",
           ALL, lr=0.03, own_lr={"AdaGrad": 0.5, "RMSProp": 0.1, "Adam": 0.1}, iters=50),
    Lesson("Rotation breaks per-coordinate scaling", "Rotated ill-conditioned quadratic",
           "The same valley rotated by 30 degrees. Per-coordinate scaling can only rescale the x and y axes, not rotate "
           "them, so the adaptive methods are no longer clearly ahead: AdaGrad and Adam now trail plain SGD. Compare with the "
           "previous example.",
           ALL, lr=0.03, own_lr={"AdaGrad": 0.5, "RMSProp": 0.1, "Adam": 0.1}, iters=50),
    Lesson("AdaGrad's shrinking steps", "Rosenbrock (banana valley)",
           "AdaGrad accumulates squared gradients forever, so its steps only shrink: it makes fast early progress, then "
           "slows down. RMSProp and Adam forget old gradients and keep moving. In the internals table watch G grow and "
           "step/|g| fall for AdaGrad.",
           {"AdaGrad": {}, "RMSProp": {}, "Adam": {}}, lr=0.3, own_lr={"RMSProp": 0.01, "Adam": 0.05}, iters=800),
    Lesson("A learning rate that is too large", "Ill-conditioned quadratic",
           "Plain SGD on a quadratic is stable only for lr < 2/lambda_max = 2/25 = 0.08. At lr = 0.09 the steep (y) direction "
           "is overshot by more each step: the loss and gradient norm grow instead of falling (the run stops when the "
           "point becomes huge). Lower the learning rate "
           "below 0.08 and press Run again.",
           {"SGD + momentum": {"momentum": 0.0}}, lr=0.09, iters=100),
    Lesson("Minibatch noise", "Least squares (data, minibatch)",
           "Solid: plain SGD with minibatches of only m = 2 samples, so every gradient is noisy. Dotted: the same run on "
           "the full data (m = 500). Loss and gradient norm are always measured on all N samples: with small minibatches "
           "the iterate keeps wandering around the minimum and the gradient norm stays well above zero, while the "
           "full-batch run settles.",
           {"SGD + momentum": {"momentum": 0.0}}, lr=0.1, iters=300, n=500, m=2, ghost={"SGD + momentum": {}},
           ghost_label="full batch (m = 500): ", ghost_m=500),
    Lesson("Several minima", "Himmelblau (four minima)",
           "Himmelblau has four equal minima. Where each optimizer ends depends on its path: momentum overshoots, adaptive "
           "methods take different first steps. From (-4, 0) plain momentum SGD and the adaptive methods end in different "
           "minima. Move the start point (or click the contour) and press Run to see how the basins are divided.",
           ALL, lr=0.01, own_lr={"AdaGrad": 0.5, "RMSProp": 0.05, "Adam": 0.1}, iters=200, start=(-4.0, 0.0)),
]
