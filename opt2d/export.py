"""Run data as CSV text."""

import math


def history_csv(run):
    """Long format: one row per optimizer and iteration (lr is the rate used to reach that iterate; blank at 0)."""
    lines = ["optimizer,iteration,lr,x,y,loss,grad_norm"]
    for name in run.names:
        h = run.history[name]
        for k, (w, loss, g) in enumerate(zip(h["w"], h["loss"], h["gnorm"])):
            cells = [h["lr"][k], w[0], w[1], loss, g]
            lines.append(f"{name},{k}," + ",".join("" if math.isnan(x) else f"{x:.10g}" for x in cells))
    return "\n".join(lines) + "\n"
