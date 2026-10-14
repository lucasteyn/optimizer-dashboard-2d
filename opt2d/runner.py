"""Run several optimizers side by side on one objective (no plotting here, so it is easy to test)."""

import numpy as np
from autograd import grad

from .objectives import subset
from .optimizers import OPTIMIZERS
from .schedules import lr_factor


class Run:
    """State of one comparison run.

    chosen: {optimizer name: {param: value, ...}}. A key "lr" in the params overrides the global ``lr``.
    schedule: "constant", "linear" or "cosine" (see schedules.py); multiplies every optimizer's learning rate.
    All optimizers start from the same w0 and see the *same* minibatch at every iteration. Loss and gradient
    norm are always measured on the FULL data, so the curves are comparable.
    """

    def __init__(self, objective, chosen, w0, lr=0.05, max_iters=200, n_samples=500, batch_size=32, seed=0,
                 schedule="constant"):
        self.objective = objective
        self.chosen = chosen
        self.names = list(chosen)
        self.lr = lr
        self.schedule = schedule
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.data = objective.make_data(n_samples, self.rng) if objective.make_data else None
        self.grad = grad(objective.loss)             # d loss / d w  (first argument)

        self.w0 = np.array(w0, dtype=float)
        self.w = {n: self.w0.copy() for n in self.names}
        self.state = {n: OPTIMIZERS[n].init(self.w0) for n in self.names}
        self.alive = {n: True for n in self.names}
        self.messages = []
        self.iteration = 0
        self.loss0, self.gnorm0 = self.metrics(self.w0)
        self.history = {n: {"w": [self.w0.copy()], "loss": [self.loss0], "gnorm": [self.gnorm0],
                            "lr": [float("nan")],
                            "grad": [np.full(2, np.nan)], "step": [np.full(2, np.nan)], "view": [{}]}
                        for n in self.names}

    def rate(self, name):
        return self.chosen[name].get("lr") or self.lr

    def metrics(self, w):
        """Full-data loss and gradient norm."""
        return float(self.objective.loss(w, self.data)), float(np.linalg.norm(self.grad(w, self.data)))

    def _minibatch(self):
        if self.data is None or self.batch_size >= self.n_samples:
            return self.data
        return subset(self.data, self.rng.choice(self.n_samples, size=self.batch_size, replace=False))

    def step(self):
        """One iteration for every optimizer that is still alive."""
        batch = self._minibatch()                    # shared by all optimizers
        t = self.iteration + 1
        factor = lr_factor(self.schedule, self.iteration, self.max_iters)
        for name in self.names:
            if not self.alive[name]:
                continue
            params = {k: v for k, v in self.chosen[name].items() if k != "lr"}
            g = self.grad(self.w[name], batch)
            lr = self.rate(name) * factor
            w_new = OPTIMIZERS[name].step(self.w[name], g, self.state[name], lr, **params)
            try:
                if not np.all(np.isfinite(w_new)) or np.abs(w_new).max() > 1e8:
                    raise FloatingPointError("the point became non-finite or huge")
                loss, gnorm = self.metrics(w_new)
                if not (np.isfinite(loss) and np.isfinite(gnorm)):
                    raise FloatingPointError("loss or gradient became non-finite")
            except FloatingPointError as e:
                self.alive[name] = False
                self.messages.append(f"{name} stopped at iteration {t}: {e}.")
                continue
            w_old, self.w[name] = self.w[name], w_new
            h = self.history[name]
            h["w"].append(w_new.copy())
            h["loss"].append(loss)
            h["gnorm"].append(gnorm)
            h["lr"].append(lr)
            h["grad"].append(np.array(g, dtype=float))                  # minibatch gradient that produced this iterate
            h["step"].append(w_new - w_old)
            h["view"].append({k: np.array(v, dtype=float) for k, v in self.state[name].get("view", {}).items()})
        self.iteration = t

    def run(self, n=None):
        """Run headlessly for ``n`` iterations (default: until ``max_iters``)."""
        for _ in range(self.max_iters - self.iteration if n is None else n):
            self.step()
        return self.history
