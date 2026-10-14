"""Everything that defines one run, in one plain object (no widgets)."""

import math
from dataclasses import dataclass


def parse_lr(text):
    """Learning-rate text box -> None (blank = inherit the global lr), a float, or nan (not a number)."""
    text = str(text).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return math.nan


@dataclass
class RunConfig:
    objective: str
    start: tuple          # (x0, y0)
    lr: float             # global learning rate
    iters: int
    chosen: dict          # {optimizer: {param: value, ["lr": own rate]}}
    n: int = 500          # samples (data objective)
    m: int = 32           # minibatch size
    seed: int = 0
    code: str = ""        # custom-objective source ("" otherwise)
    schedule: str = "constant"
    window: tuple = (0.0, 0.0, 3.0)    # centre x, centre y, half-width (used by the custom objective)

    def validate(self):
        if not self.chosen:
            raise ValueError("select at least one optimizer")
        if not (math.isfinite(self.lr) and self.lr > 0):
            raise ValueError("the global learning rate must be a positive number")
        for name, params in self.chosen.items():
            lr = params.get("lr")
            if lr is not None and not (math.isfinite(lr) and lr > 0):
                raise ValueError(f"{name}: learning rate must be a positive number (or blank to use the global one)")
