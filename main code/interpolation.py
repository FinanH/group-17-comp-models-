# interpolation.py

import numpy as np

# ----------------------------
# Utility: custom interpolation
# ----------------------------
# Instead of using scipy.interpolate.interp1d, we roll our own.
# It’s basic piecewise linear interpolation, but it does the job and
# is easy to understand by inspection.

def linear_interp1d(x_data, y_data):
    """
    Simple 1D piecewise linear interpolator.

    Returns a function f such that f(x) linearly interpolates between points.
    Works for scalars or numpy arrays.

    Assumes x is 1D and not all equal.
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    # Sort by x just to be safe
    sort_idx = np.argsort(x_data)
    x_data = x_data[sort_idx]
    y_data = y_data[sort_idx]

    def _interp_scalar(x):
        x = float(x)
        # Clamp on the ends (hold the first/last value)
        if x <= x_data[0]:
            return float(y_data[0])
        if x >= x_data[-1]:
            return float(y_data[-1])
        # Otherwise find the segment we’re in
        for i in range(len(x_data) - 1):
            x0, x1 = x_data[i], x_data[i + 1]
            if x0 <= x <= x1:
                y0, y1 = y_data[i], y_data[i + 1]
                t = (x - x0) / (x1 - x0)
                return float(y0 + t * (y1 - y0))
        # Fallback (shouldn’t really hit this)
        return float(y_data[-1])

    def f(x):
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 0:
            return _interp_scalar(x_arr)
        # Vectorised loop (simple, but clear)
        return np.array([_interp_scalar(xi) for xi in x_arr])

    return f
