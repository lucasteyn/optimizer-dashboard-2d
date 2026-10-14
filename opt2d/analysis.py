"""Numbers that summarise a finished run."""

import numpy as np


def best_known_loss(run):
    """Loss at the best known minimiser (full data); if none is known, the lowest loss seen in this run."""
    if run.objective.optima:
        return min(float(run.objective.loss(np.array(o, dtype=float), run.data)) for o in run.objective.optima), True
    return min(min(h["loss"]) for h in run.history.values()), False


def first_below(values, tol):
    """First iteration k with values[k] <= tol, or None."""
    hit = np.flatnonzero(np.asarray(values, dtype=float) <= tol)
    return int(hit[0]) if hit.size else None


def convergence_rows(run, loss_gap, gnorm_tol):
    """One row per optimizer: iterations until loss - best_known_loss <= loss_gap and until gradient norm <= gnorm_tol.

    With a minibatch objective the gradient norm is measured on the full data and may never get small.
    """
    best, known = best_known_loss(run)
    rows = []
    for name in run.names:
        h = run.history[name]
        rows.append({"name": name,
                     "loss_iter": first_below(np.array(h["loss"]) - best, loss_gap),
                     "gnorm_iter": first_below(h["gnorm"], gnorm_tol),
                     "final_gap": h["loss"][-1] - best, "final_gnorm": h["gnorm"][-1],
                     "stopped": not run.alive[name]})
    return rows, best, known
