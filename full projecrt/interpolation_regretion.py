import math
from typing import Tuple, List, Dict, Optional, Set

import numpy as np


# ----------------------------
# Physical constants & battery
# ----------------------------
# Here we define one specific battery pack and keep everything consistent with it.

Vb = 133                # battery voltage (V)
Cb = 27                 # Battery capacity (Ah)
usable_frac = 0.9       # We don't want to drain it 100% in real life

# Usable energy from the pack:
#   V * Ah = Wh
#   Wh * 3600 = J     (because 1 Wh = 3600 J)
E_avail = usable_frac * Vb * Cb * 3600  # J (this is what the ODEs see)

# Same pack, but in kWh for route planning & summary plots
BATTERY_CAPACITY_FROM_CELLS_KWH = usable_frac * Vb * Cb / 1000.0  # kWh ≈ 0.0799

# A very simple mass breakdown

m_frame = 5.93
m_payload_nom = 6
m_battery = 9.5
m_tot = m_frame + m_payload_nom + m_battery


# Vertical target height (we assume every mission climbs to this, then cruises)
alt_target = 30.0       #

v_t_target = 2

v_l_target = -0.5

vx_target = 18.0  

x_target = 2000        # target horizontal distance (m)
z_target = 0
transition_time = 3    # tilt transition (s)

# Braking tilt
theta_cruise = np.deg2rad(25)  # cruising tilt
theta_brake = np.deg2rad(-20)  # braking tilt (negative)
theta_level = np.deg2rad(0)    # level

# ----------------------------
# Drone + environment parameters
# ----------------------------
# These are “generic multirotor-ish” values, not tied to a real product.

            # avionics + misc power (W), always on

g = 9.81                # gravity (m/s^2)
rho = 1.225             # air density (kg/m^3)
C_dx = 0.8              # Drag coefficient horizontal
C_dz = 1                # Drag coefficient vertical
A_front = 1.154         # Effective front area of drone (m^2)
A_top = 0.175674        # Effective top area of drone (m^2)
A_disk = 1.34           # total rotor disk area (m^2)
eta = 0.75              # overall efficiency (motor * prop)
P_av = 12               # avionics power (W)

vh = np.sqrt((m_tot * g) / (2 * rho * A_disk))  # hover induced velocity (m/s)

# Coefficients for the power model
k1 = 1.1 / eta
k2 = 0.5 * rho * C_dz * A_top
k3 = P_av

# Keep everything in one params dict so we can pass it around cleanly
params = {
    "g": g,
    "rho": rho,
    "C_dz": C_dz,
    "C_dx": C_dx,
    "A_top": A_top,
    "A_disk": A_disk,
    "eta": eta,
    "m_frame": m_frame,
    "m_payload": m_payload_nom,
    "m_battery": m_battery,
    "m_dry": m_frame + m_battery,
    "Vb": Vb,
    "Cb": Cb,
    "P_av": P_av,
    "vh": vh,
    "k1": k1,
    "k2": k2,
    "k3": k3,
    "v_hover": vh,
    "E_avail": E_avail,
}

# For the grid plot (used to convert kWh to battery %)
BATTERY_CAPACITY_KWH_GLOBAL: Optional[float] = None

# Globals for vertical ODEs (we reuse the same dynamics with different masses)
M_TAKEOFF = m_tot
VH_TAKEOFF = vh
M_LANDING = m_tot
VH_LANDING = vh


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
    x = a + step

    while x <= b:
        f_right = f(x)
        # Exact zero at left point
        if f_left == 0.0:
            return x_left
        # Check for sign change
        if f_left * f_right <= 0.0:
            # We have a bracket [x_left, x]; do bisection here
            lo, hi = x_left, x
            f_lo, f_hi = f_left, f_right
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                f_mid = f(mid)
                if abs(f_mid) < tol or abs(hi - lo) < tol:
                    return mid
                if f_lo * f_mid <= 0.0:
                    hi, f_hi = mid, f_mid
                else:
                    lo, f_lo = mid, f_mid
            # If we exit the loop, just return the midpoint we ended up with
            return 0.5 * (lo + hi)
        x_left, f_left = x, f_right
        x += step

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
