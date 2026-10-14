"""Learning-rate schedules: a factor in [0, 1] that multiplies every optimizer's base learning rate.

Iteration k (0-based) of T total uses ``lr * lr_factor(kind, k, T)``. To add a schedule: add a branch here and a
name in ``SCHEDULES``.
"""

import math

SCHEDULES = [("Constant", "constant"), ("Linear decay to 0", "linear"), ("Cosine decay to 0", "cosine")]


def lr_factor(kind, k, T):
    if kind == "constant" or T <= 1:
        return 1.0
    x = min(max(k / T, 0.0), 1.0)
    if kind == "linear":
        return 1.0 - x
    if kind == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * x))
    raise ValueError(f"unknown schedule {kind!r}")


def sparkline_svg(kind, T, width=230, height=46):
    """Tiny inline-SVG preview of the factor over the run (no plotting library needed)."""
    n = 60
    pts = " ".join(f"{4 + (width - 8) * i / (n - 1):.1f},{4 + (height - 8) * (1 - lr_factor(kind, T * i / (n - 1), T)):.1f}"
                   for i in range(n))
    return (f"<svg width='{width}' height='{height}' style='background:#f7f7f7;border:1px solid #ddd'>"
            f"<polyline points='{pts}' fill='none' stroke='#444' stroke-width='2'/></svg>")
