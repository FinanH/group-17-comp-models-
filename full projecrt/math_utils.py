# math_utils.py
import math
import numpy as np

def linear_interp1d(x_data, y_data):
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    sort_idx = np.argsort(x_data)
    x_data = x_data[sort_idx]
    y_data = y_data[sort_idx]

    def _interp_scalar(x):
        x = float(x)
        if x <= x_data[0]:
            return float(y_data[0])
        if x >= x_data[-1]:
            return float(y_data[-1])
        for i in range(len(x_data) - 1):
            x0, x1 = x_data[i], x_data[i + 1]
            if x0 <= x <= x1:
                y0, y1 = y_data[i], y_data[i + 1]
                t = (x - x0) / (x1 - x0)
                return float(y0 + t * (y1 - y0))
        return float(y_data[-1])

    def f(x):
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 0:
            return _interp_scalar(x_arr)
        return np.array([_interp_scalar(xi) for xi in x_arr])

    return f


def incremental_bisection_root(f, a, b, step=0.5, tol=1e-3, max_iter=50):
    x_left = a
    f_left = f(x_left)
    x = a + step

    while x <= b:
        f_right = f(x)
        if f_left == 0.0:
            return x_left
        if f_left * f_right <= 0.0:
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
            return 0.5 * (lo + hi)
        x_left, f_left = x, f_right
        x += step

    return None


def linear_regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return 0.0, float(y.mean()) if y.size > 0 else 0.0

    x_mean = x.mean()
    y_mean = y.mean()
    Sxy = np.sum((x - x_mean) * (y - y_mean))
    Sxx = np.sum((x - x_mean) ** 2)
    if Sxx == 0.0:
        return 0.0, y_mean
    m = Sxy / Sxx
    b = y_mean - m * x_mean
    return m, b


def exponential_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = y > 0
    x2 = x[mask]
    y2 = y[mask]

    if x2.size < 2:
        a = float(np.mean(y2)) if y2.size > 0 else 0.0
        b = 0.0

        def f_const(xnew):
            return np.full_like(np.asarray(xnew, dtype=float), a, dtype=float)

        return f_const, a, b

    ln_y = np.log(y2)
    from .math_utils import linear_regression as _linreg  # for self-containment if needed
    m, b_lin = linear_regression(x2, ln_y)
    a = math.exp(b_lin)
    b = m

    def f(x_new):
        x_arr = np.asarray(x_new, dtype=float)
        return a * np.exp(b * x_arr)

    return f, a, b
