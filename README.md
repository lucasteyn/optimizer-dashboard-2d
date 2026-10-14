# 2-D optimizer dashboard (Streamlit)

Compare SGD with momentum, AdaGrad, RMSProp and Adam on a function of two variables `w = (x, y)`.
One page, four panels that animate on **fixed** axes:

| | |
|---|---|
| **Trajectory** (contour, paths) | **3-D surface** with the paths lifted onto it |
| **Loss** against iteration | **Gradient norm** against iteration |

Press **Run** (all iterations are computed at once), then **Play** or drag the slider. The browser replays the stored
run, so playing causes no Streamlit reruns and nothing flickers; contours, surface and axes never move.

Above the plots: **Speed**, **Trail** (last k points only), **Mark every k iterations**, and 3-D camera buttons
(Isometric / Top / Side, auto-rotate; drag the surface to rotate, scroll to zoom). Under the plots: a live table and
an **Inside the optimizer** table (gradient, step, step/|g| per coordinate, momentum buffer / G / v / m-hat, v-hat).
The **Convergence** tab counts iterations until the loss gap and the gradient norm fall below a tolerance.

## Run locally

```bash
conda activate ds_346_env_2026
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

To enable the **custom objective** (it executes the code you type, so it is off unless you ask for it):

```bat
set ALLOW_CUSTOM=1
python -m streamlit run streamlit_app.py
```

(PowerShell: `$env:ALLOW_CUSTOM = "1"`.) The custom box takes a function of `w`, with `np` = `autograd.numpy`:

```python
def loss(w):
    x, y = w[0], w[1]
    return (1 - x) ** 2 + 5 * (y - x ** 2) ** 2
```

Use `np.sin`, `np.exp`, `np.sum`, ... (not `math` or plain `numpy`, which autograd cannot differentiate).
Set the plotted window (centre and half-width) under **View**.

## Controls

- **Guided example**: seven what-to-look-for setups (momentum in a valley, adaptive steps, rotation, AdaGrad's
  shrinking steps, too large a learning rate, minibatch noise, several minima). Each loads its settings, explains
  what to watch, and where useful pins a baseline run (dotted) automatically. Changing any control leaves the example.

- **Objective**: Quadratic bowl, ill-conditioned and rotated quadratics, Rosenbrock, Himmelblau, Beale, Rastrigin,
  least squares with minibatches (N, m, seed), or Custom. Choosing one loads its start point, window and learning rates.
- **Start point**: type x and y, or click the contour before a run.
- **Optimizers**: tick the ones to compare; each has a settings box (own learning rate, momentum / rho / betas / eps).
- **Run**: global learning rate, max iterations, schedule (constant / linear / cosine decay).
- **Pin run**: keeps the finished run as a faded dotted ghost so a changed setting can be compared with it.
- **View**: window (fixed or fitted to the paths), grid resolution, log colour scale, log axes.
- **Download CSV** (every iterate, loss, gradient norm) and **Download page** (the player as a standalone HTML file
  that needs only internet access for plotly.js).

Loss and gradient norm are always measured on the full objective (for least squares, on all N samples), while the
update uses the shared minibatch. The 3-D height is the loss at each iterate (log scale if the log option is on).

## Layout

```
streamlit_app.py     the page (sidebar, Run, replay)
opt2d/objectives.py  losses (autograd) and the presets: add yours here
opt2d/optimizers.py  four small step functions
opt2d/runner.py      Run: all optimizers, shared minibatch, full-data metrics
opt2d/figures.py     the 2x2 Plotly figure with fixed axes
opt2d/player.py      browser-side player: Play / slider / speed / trail / camera / tables (Plotly.restyle)
opt2d/lessons.py     guided examples (settings, note, optional baseline)
opt2d/analysis.py    iterations-to-tolerance
opt2d/settings.py    widget values -> RunConfig -> Run
tests/               python -m pip install -r requirements-dev.txt; python -m pytest
```
