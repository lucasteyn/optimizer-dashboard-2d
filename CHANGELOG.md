# Changelog

## 0.2.1
- Player is an explicit light card, so tables stay readable in Streamlit's dark theme (same look as the HTML export).
- Iteration markers are larger (size 11, white outline).

## 0.2.0
- Player: speed, trail, iteration markers and 3-D camera presets / auto-rotate are browser controls (no rebuild);
  larger 3-D panel; standalone HTML download.
- Efficiency: contour/surface grid cached with st.cache_data; 3-D surface drawn on a coarser grid; changing speed, trail,
  markers or camera no longer touches Python.
- Education: seven guided examples with automatic baseline pin; "Inside the optimizer" table (gradient, step,
  step/|g|, buffers); Convergence tab (iterations to loss gap / gradient-norm tolerance).

## 0.1.0
- First version: 2-D vector problem, SGD+momentum / AdaGrad / RMSProp / Adam, 2x2 animated page
  (trajectory, loss, gradient norm, 3-D surface) on fixed axes, presets plus gated custom objective,
  pin run, CSV export, click-to-set-start before a run.
