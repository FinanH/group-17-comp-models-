# root_finding.py

#IMPORTANR - it is notable this is not the only place i have used root finding i also
# used root finding in the ode part of the code when taking off

import math
from typing import Optional

import numpy as np

from ode import params


# ----------------------------
# Utility: incremental-search root finding + bisection
# ----------------------------
# We don’t use scipy’s root_scalar here on purpose.
# Instead, we:
#   1) Walk along the interval in fixed steps (incremental search).
#   2) Look for a sign change.
#   3) When we find one, we refine with a standard bisection method.

def incremental_bisection_root(f, a, b, step=0.5, tol=1e-3, max_iter=50):
    """
    Find a root of f on [a,b] using:
      - incremental search (step size `step`) to find a bracket,
      - then bisection to polish it.

    Returns:
      root (float) or None if nothing changes sign on [a,b].
    """
    x_left = a
    f_left = f(x_left)

    # If we basically start on a root
    if abs(f_left) < tol:
        return x_left

    x = a + step

    while x <= b:
        f_right = f(x)

        # If the right sample is basically on the root
        if abs(f_right) < tol:
            return x

        # Check for actual sign change between left and right
        if f_left * f_right < 0.0:
            # We have a bracket [x_left, x]; do bisection here
            lo, hi = x_left, x
            f_lo, f_hi = f_left, f_right
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                f_mid = f(mid)
                if abs(f_mid) < tol or abs(hi - lo) < tol:
                    return mid
                if f_lo * f_mid < 0.0:
                    hi, f_hi = mid, f_mid
                else:
                    lo, f_lo = mid, f_mid
            return 0.5 * (lo + hi)

        x_left, f_left = x, f_right
        x += step

    # Final safety: check the right endpoint explicitly
    f_b = f(b)
    if abs(f_b) < tol:
        return b

    # No sign change found
    return None


# ----------------------------
# Utility: simple linear regression
# ----------------------------
# We only need basic y ≈ m*x + b, so we implement OLS by hand.

def linear_regression(x, y):
    """
    Ordinary least squares linear regression.
    Returns slope m and intercept b for y ≈ m*x + b.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        # Degenerate case: not enough data -> 0 slope, mean value
        return 0.0, float(y.mean()) if y.size > 0 else 0.0

    x_mean = x.mean()
    y_mean = y.mean()
    Sxy = np.sum((x - x_mean) * (y - y_mean))
    Sxx = np.sum((x - x_mean) ** 2)
    if Sxx == 0.0:
        # All x are the same -> vertical line; we just return a constant model
        return 0.0, y_mean
    m = Sxy / Sxx
    b = y_mean - m * x_mean
    return m, b


# ----------------------------
# Utility: custom exponential fit
# ----------------------------
# For payload vs range/endurance, we often see a “curve” rather than a straight line.
# Here we fit y ≈ a * exp(b x) by taking logs and then reusing our linear_regression.

def exponential_fit(x, y):
    """
    Fit y ≈ a * exp(b*x) using a log transform and our own linear regression.

    Returns:
      f(x_new)  - callable exponential fit
      a, b      - parameters in y = a * exp(b x)

    Points with y <= 0 are ignored for the log transform (since log(<=0) is invalid).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Only use positive y values for log fitting
    mask = y > 0
    x2 = x[mask]
    y2 = y[mask]

    if x2.size < 2:
        # If we can't sensibly fit, fall back to a constant function
        a = float(np.mean(y2)) if y2.size > 0 else 0.0
        b = 0.0

        def f_const(xnew):
            return np.full_like(np.asarray(xnew, dtype=float), a, dtype=float)

        return f_const, a, b

    ln_y = np.log(y2)
    m, b_lin = linear_regression(x2, ln_y)
    a = math.exp(b_lin)
    b = m

    def f(x_new):
        x_arr = np.asarray(x_new, dtype=float)
        return a * np.exp(b * x_arr)

    return f, a, b


# -----------------------
# Root-finding: crossover speed (power balance)
# -----------------------
# Here we find the speed where induced power = drag power for a given payload.
# That’s a nice numerical example and gives some insight into the power model.

def crossover_speed_ms_for_payload(payload_kg: float) -> Optional[float]:
    """
    Use incremental-search-based root finding to solve P_drag(v) - P_ind(v) = 0
    => finds the speed where drag power equals induced power.
    """
    g_ = params["g"]
    m0 = params["m_dry"]
    vh0 = params["v_hover"]
    k1p = params["k1"]
    k2p = params["k2"]

    m = m0 + payload_kg

    def f(v):
        if v <= 0:
            return 1e6  # just a big positive number to avoid nonsense
        P_ind = k1p * ((m * g_) ** 1.5) / np.sqrt(v ** 2 + vh0 ** 2)
        P_drag = k2p * v ** 3
        return P_drag - P_ind

    root = incremental_bisection_root(f, a=0.1, b=40.0, step=0.5, tol=1e-3, max_iter=60)
    return root


def print_crossover_speeds():
    print("Speed where induced and drag power are equal (incremental-search root-finding):")
    for payload in [0.0, 2.0, 5.0, 10.0]:
        v = crossover_speed_ms_for_payload(payload)
        if v is None:
            print(f"  payload {payload:.1f} kg: no root found in [0.1, 40] m/s")
        else:
            print(f"  payload {payload:.1f} kg: v ≈ {v:.2f} m/s ({v*3.6:.1f} km/h)")
    print()
