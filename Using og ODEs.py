# drone_routes_energy_opt.py
#
# This script is a mini “drone delivery lab”:
# - There’s a drone flying on a 2D grid with obstacles (no-fly squares).
# - It has a physically-based energy model (very simplified, but consistent).
# - Take-off and landing use vertical ODEs.
# - Cruise uses an induced + drag + fixed power model.
# - The drone does multiple trips from a warehouse, recharging fully each time.
# - We track energy usage and battery state along each path.
#LANDING_KWH_BASE

# On top of that, there are some numerical analysis “showpieces”:
# - ODE solving with solve_ivp for vertical motion.
# - Custom incremental-search + bisection root finding (no fancy library solvers).
# - A home-made 1D interpolation function instead of using interp1d.
# - Custom linear and exponential regression (fitting) without sklearn/polyfit.
#
# The idea is: this looks and feels like a student project where someone
# actually thought about the maths and then implemented it in Python.


import random
import math
import heapq
from typing import Tuple, List, Dict, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Rectangle, FancyArrowPatch


# ----------------------------
# Physical constants & battery
# ----------------------------
# Here we define one specific battery pack and keep everything consistent with it.

Vb = 22.2              # battery voltage (V)
Cb = 27                  # Battery capacity (Ah)
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


# ---------------------------------
# Takeoff / landing dynamics (ODE)
# ---------------------------------
# These ODEs are deliberately simple – they’re not meant to match a real autopilot,
# but they do capture “energy is power integrated over time while moving vertically”.

def takeoff_dynamics(t, y):
    z, vz, E = y

    # --- Control law / thrust command ---
    # Simple proportional controller on vertical speed:
    k_p = 6.0         # if vz not euqual to v target increase acceleration by 5
                            
    Thurst = m_tot * (g + k_p * (v_t_target - vz))  # Thurst (N)
    Drag_z = 0.5*rho*C_dz*A_top*vz* abs(vz)        # vertical drag (N)
   
    # --- Power model ---
   
    # Induced velocity (momentum theory)
    vi = np.sqrt(Thurst / (2 * rho * A_disk))
  
    P_ind = Thurst * vi                      # induced power (W)
    P_elec = (P_ind / eta) + P_av            # electrical power 

    # --- Dynamics ---
    dzdt = vz
    dvzdt = (Thurst - (m_tot * g) - Drag_z )/ m_tot
    dEdt = -P_elec

    return [dzdt, dvzdt, dEdt]

def takeoff_energy_kwh_for(payload_kg: float) -> float:
    """
    Takeoff energy for a given payload.
    We linearly scale the baseline takeoff energy by total mass ratio.
    """
    total_mass = m_frame + m_battery + payload_kg
    return TAKEOFF_KWH_BASE * (total_mass / m_tot)


def landing_dynamics(t, y):
    z, vz, E = y

    # --- Control law / thrust command ---
    # Simple proportional controller on vertical speed:
    k_p = 0.5                                        # if vz not euquak to v target increase acceleration by 5
    
    Thurst = max(0.0, m_tot * (g + k_p * (v_l_target - vz)))  # Thurst (N)
   
    Drag_z = 0.5*rho*C_dz*A_top*(vz)* abs(vz)           # vertical drag (N)
    
    # --- Power model ---
    # Induced velocity (momentum theory)
    vi = np.sqrt(Thurst / (2 * rho * A_disk))
    P_ind = Thurst * vi                     # induced power (W)
    P_elec = (P_ind / eta) + P_av           # electrical power 

    # --- Dynamics ---
    dzdt = vz
    dvzdt = (Thurst - (m_tot * g) - Drag_z )/ m_tot
    dEdt = -P_elec

    return [dzdt, dvzdt, dEdt]


# ---------------------------------------
# Forward-flight power
# ---------------------------------------
# This is a simple induced + profile drag + fixed term model.

def power_model(W, v, params):
    """
    Power in forward flight, including induced + drag + fixed components.

    W  = payload mass (kg)
    v  = airspeed (m/s)
    """
    g_ = params["g"]
    m0 = params["m_dry"]
    vh0 = params["v_hover"]
    k1p = params["k1"]
    k2p = params["k2"]
    k3p = params["k3"]

    m = m0 + W
    P_ind = k1p * ((m * g_) ** 1.5) / np.sqrt(v ** 2 + vh0 ** 2)
    P_drag = k2p * v ** 3
    P_fixed = k3p
    return P_ind + P_drag + P_fixed   # Watts


# ---------------------------------------
# Takeoff / landing energy (uses ODEs)
# ---------------------------------------
# We compute a "baseline" takeoff & landing energy with the nominal mass,
# then scale with total mass for other payloads (simple linear scaling assumption).


def compute_landing_energy_kwh():
    """
    Integrate landing_dynamics from z=alt_target down to 0 using solve_ivp
    to get baseline landing energy (for nominal mass).
    """
    global M_LANDING, VH_LANDING
    M_LANDING = m_tot
    VH_LANDING = np.sqrt((M_LANDING * g) / (2 * rho * A_disk))

    # Start at target altitude, zero vertical speed, full usable energy
    y0 = [alt_target, 0.0, E_avail]
    t_span = (0.0, 60.0)

    def event_ground_reached(t, y):
        return y[0]  # z = 0 -> ground

    event_ground_reached.terminal = True
    event_ground_reached.direction = -1

    sol = solve_ivp(
        landing_dynamics,
        t_span,
        y0,
        events=event_ground_reached,
        rtol=1e-6,
        atol=1e-8,
        max_step=0.5
    )

    E_end = sol.y[2][-1]
    E_used_J = E_avail - E_end
    return E_used_J / 3.6e6  # J -> kWh


# Baseline vertical costs (for the nominal mass)
LANDING_KWH_BASE = compute_landing_energy_kwh()
TAKEOFF_KWH_BASE: Optional[float] = None


def landing_energy_kwh_for(payload_kg: float) -> float:
    """
    Landing energy for a given payload.
    Same scaling assumption as takeoff.
    """
    total_mass = m_frame + m_battery + payload_kg
    return LANDING_KWH_BASE * (total_mass / m_tot)

def takeoff_energy_kwh_for(payload_kg: float) -> float:
    """
    Takeoff energy for a given payload.

    Uses the existing takeoff_dynamics ODE once (baseline mass),
    then scales linearly with total mass, same idea as landing_energy_kwh_for.
    The E state is the 3rd component of the ODE state vector.
    """
    global TAKEOFF_KWH_BASE, M_TAKEOFF, VH_TAKEOFF

    # Compute baseline once (for nominal total mass m_tot)
    if TAKEOFF_KWH_BASE is None:
        M_TAKEOFF = m_tot
        VH_TAKEOFF = np.sqrt((M_TAKEOFF * g) / (2 * rho * A_disk))

        # Start at ground, zero vertical speed, full usable energy
        y0 = [0.0, 0.0, E_avail]
        t_span = (0.0, 60.0)

        def event_alt_reached(t, y):
            # stop when altitude reaches alt_target
            return y[0] - alt_target

        event_alt_reached.terminal = True
        event_alt_reached.direction = 1

        sol = solve_ivp(
            takeoff_dynamics,
            t_span,
            y0,
            events=event_alt_reached,
            rtol=1e-6,
            atol=1e-8,
            max_step=0.1
        )

        # y[2] is E(t), which came from integrating dE/dt (the 3rd returned value of takeoff_dynamics)
        E_end = sol.y[2][-1]
        E_used_J = E_avail - E_end
        TAKEOFF_KWH_BASE = E_used_J / 3.6e6  # J -> kWh

    # Scale baseline takeoff energy with total mass
    total_mass = m_frame + m_battery + payload_kg
    return TAKEOFF_KWH_BASE * (total_mass / m_tot)


# ---------------------------------------
# Cruise energy
# ---------------------------------------
# This is “distance times power”, but with the power coming from power_model.

# ----------------------------
# Robust braking distance solver using solve_ivp + event
# ----------------------------
def braking_distance_ivp(v0, theta_b):
    """
    Integrate 1D braking dynamics until v = 0.
    Uses solve_ivp with event (root finding) to stop automatically.
    """
    def dyn(t, y):
        v, x = y
        Drag_x = 0.5 * rho * C_dx * A_front * max(v, 0)**2
        F_thrust_x = -m_tot * g * np.sin((abs(theta_b)))  # opposite to +x
        a = (F_thrust_x - Drag_x) / m_tot
        return [a, v]

    # Stop when v = 0 (root-finding event)
    def stop_v0(t, y):
        return y[0]  # velocity
    stop_v0.terminal = True
    stop_v0.direction = -1  # trigger when v decreases through zero

    sol = solve_ivp(dyn, (0, 100), [v0, 0.0], events=stop_v0, max_step=0.1)
    return sol.y[1, -1]  # return stopping distance


# Compute braking distance and cruise start
d_brake = braking_distance_ivp(vx_target, theta_brake)
x_cruise_start = x_target - d_brake
print(f"Active braking distance ≈ {d_brake:.1f} m | Begin braking at x ≈ {x_cruise_start:.1f} m")


# ----------------------------
# Drone dynamics
# ----------------------------
def move_energy(t, y):
    z, vz, x, vx, E = y

    # Drag forces
    Drag_x = 0.5 * rho * C_dx * A_front * vx**2
    Drag_z = 0.5 * rho * C_dz * A_top * vz**2

    # Determine tilt (cruise → brake → level)
    if x < x_cruise_start:
        theta = theta_cruise
    elif x < x_target:
        theta = theta_brake if vx > 0 else 0.0  # only brake while moving forward
    else:
        theta = theta_level

    # Vertical control
    k_p_z = 10.0
    Thrust = m_tot * (g + k_p_z * (z_target - z)) / np.cos(theta)

    # Power model
    vi = np.sqrt(Thrust / (2 * rho * A_disk))
    P_ind = Thrust * vi
    P_par = abs(Drag_x * vx)
    P_elec = (P_ind + P_par)/eta + P_av

    # Dynamics
    dzdt = vz
    dvzdt = (Thrust*np.cos(theta) - m_tot*g - Drag_z)/m_tot
    dxdt = vx
    dvxdt = (Thrust*np.sin(theta) - Drag_x)/m_tot
    dEdt = -P_elec

    return [dzdt, dvzdt, dxdt, dvxdt, dEdt]

def move_energy_kwh(distance_km: float, payload_kg: float, cruise_speed_kmh: float) -> float:
    """
    Cruise energy for a horizontal grid leg.

    Uses the move_energy ODE:
      state y = [z, vz, x, vx, E]
      dE/dt = -P_elec from your ODE.

    We:
      - Integrate until x reaches distance_km * 1000 m,
      - Take the difference in E to get energy in Joules,
      - Convert to kWh,
      - Scale linearly with total mass for different payloads.
    """
    if distance_km <= 0.0 or cruise_speed_kmh <= 0.0:
        return 0.0

    distance_m = distance_km * 1000.0
    v = cruise_speed_kmh / 3.6  # m/s

    # We will temporarily change some globals used inside move_energy
    # so that the leg is flown at constant cruise tilt and level altitude.
    global x_target, x_cruise_start, z_target

    old_x_target = x_target
    old_x_cruise_start = x_cruise_start
    old_z_target = z_target

    try:
        # Level flight around current cruise altitude
        z_target = alt_target

        # Force "always cruise tilt": x < huge -> theta = theta_cruise
        x_target = 1e9
        x_cruise_start = 1e9

        # Initial state: at cruise altitude, no vertical motion, x=0, vx = cruise speed, full usable energy
        y0 = [alt_target, 0.0, 0.0, v, E_avail]

        def event_reach_distance(t, y):
            # Stop integration when x reaches the leg length
            return y[2] - distance_m

        event_reach_distance.terminal = True
        event_reach_distance.direction = 1

        # Rough upper bound on time: distance / speed * 5 as safety factor
        t_end = distance_m / max(v, 0.1) * 5.0

        sol = solve_ivp(
            move_energy,
            (0.0, t_end),
            y0,
            events=event_reach_distance,
            rtol=1e-6,
            atol=1e-8,
            max_step=0.5
        )

        # If for some reason the event didn't trigger, fall back to the algebraic model
        if sol.t_events[0].size == 0:
            P_W = power_model(payload_kg, v, params)  # Watts
            P_kW = P_W / 1000.0
            time_h = distance_km / cruise_speed_kmh
            return P_kW * time_h

        # E is the 5th state (index 4); E_avail is the initial energy (Joules)
        E_end = sol.y[4, -1]
        E_used_J = E_avail - E_end
        E_base_kwh = max(E_used_J, 0.0) / 3.6e6  # J -> kWh

    finally:
        # Restore globals so demos & other code still behave as before
        x_target = old_x_target
        x_cruise_start = old_x_cruise_start
        z_target = old_z_target

    # Scale with total mass, same idea as takeoff/landing
    total_mass = m_frame + m_battery + payload_kg
    mass_ratio = total_mass / m_tot
    return E_base_kwh * mass_ratio


# -----------------------
# Grid / routing helpers
# -----------------------
# We model the city as a simple 10x10 grid. Each cell can be:
#   - warehouse
#   - delivery location
#   - normal empty cell
#   - no-fly (blocked)
#
# The drone moves in 8 directions (including diagonals). Each step has a distance.

Coord = Tuple[int, int]
Path = List[Coord]


def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols


def neighbors(r: int, c: int, rows: int, cols: int) -> List[Coord]:
    # 8-neighbour connectivity: up, down, left, right, and diagonals
    return [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
        (r - 1, c - 1),
        (r - 1, c + 1),
        (r + 1, c - 1),
        (r + 1, c + 1),
    ]


def step_cost(u: Coord, v: Coord) -> float:
    """
    Distance between two neighboring cells in KM.
    Cardinal step = 0.1 km (100 m)
    Diagonal step = 0.1 * sqrt(2) km
    """
    base = 0.1  # km per cardinal step
    dr = v[0] - u[0]
    dc = v[1] - u[1]
    if dr != 0 and dc != 0:
        return base * math.sqrt(2.0)
    return base


def dijkstra_on_grid(rows: int, cols: int, blocked: Set[Coord], start: Coord):
    """
    Classic Dijkstra shortest paths from 'start' to all cells.
    We ignore energy here; it’s purely geometric distance in km.
    """
    dist: Dict[Coord, float] = {start: 0.0}  # km
    prev: Dict[Coord, Coord] = {}
    pq: List[Tuple[float, Coord]] = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        ur, uc = u
        for vr, vc in neighbors(ur, uc, rows, cols):
            if not in_bounds(vr, vc, rows, cols):
                continue
            if (vr, vc) in blocked:
                continue

            sc = step_cost(u, (vr, vc))  # km
            nd = d + sc
            v = (vr, vc)

            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev


def reconstruct_path(prev: Dict[Coord, Coord], start: Coord, goal: Coord) -> Optional[Path]:
    """
    Rebuild a path from start to goal using the 'prev' map from Dijkstra.
    """
    if goal not in prev and goal != start:
        return None
    if goal == start:
        return [start]
    path = [goal]
    cur = goal
    while cur != start:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def precompute_pairs(rows: int, cols: int, points: List[Coord], blocked: Set[Coord]):
    """
    Precompute shortest distances and paths (in KM) between key points.

    This lets us query "shortest path from A to B" very cheaply during routing.
    """
    global x_target
    dist_km: Dict[Tuple[Coord, Coord], float] = {}
    path_map: Dict[Tuple[Coord, Coord], Path] = {}

    for s in points:
        d, prev = dijkstra_on_grid(rows, cols, blocked, s)
        for t in points:
            if s == t:
                dist_km[(s, t)] = 0.0
                path_map[(s, t)] = [s]
            else:
                if t not in d:
                    dist_km[(s, t)] = float("inf")
                    path_map[(s, t)] = []
                else:
                    dist_km[(s, t)] = d[t]  # already km
                    pt = reconstruct_path(prev, s, t)
                    if pt is None:
                        dist_km[(s, t)] = float("inf")
                        path_map[(s, t)] = []
                    else:
                        path_map[(s, t)] = pt
                        
                        
    return dist_km, path_map


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


# -----------------------
# Trip-by-trip planner (per-trip battery, returns paths + SOC per step)
# -----------------------
# This is where we actually "fly". The drone does several trips until all
# deliveries are serviced. After each trip it magically recharges to full.

def run_all_trips(
    rows: int,
    cols: int,
    warehouse: Coord,
    deliveries: List[Coord],
    demands: Dict[Coord, int],
    blocked: Set[Coord],
    carry_capacity: int = 10,
    battery_capacity_kwh: float = 0.08,
    cruise_speed_kmh: float = 40.0
) -> List[Tuple[Path, List[float]]]:
    """
    Run all trips. Each trip starts with a FULL battery (recharged at warehouse).

    Returns a list of (trip_cells, trip_soc_kwh) where:
      trip_cells[i]   = grid coord of step i
      trip_soc_kwh[i] = remaining battery energy (kWh) after reaching that cell.
    """
    points = [warehouse] + deliveries
    dist_km, path = precompute_pairs(rows, cols, points, blocked)

    remaining: List[Coord] = []
    impossible: List[Coord] = []
    for d in deliveries:
        if demands[d] > carry_capacity:
            # Impossible: demand exceeds total capacity of drone
            impossible.append(d)
        elif dist_km[(warehouse, d)] == float("inf") or dist_km[(d, warehouse)] == float("inf"):
            # Also impossible: blocked by no-fly cells
            impossible.append(d)
        else:
            remaining.append(d)

    if impossible:
        print("These deliveries are impossible (capacity or obstacles):", impossible)

    trip_idx = 0
    total_energy_used = 0.0
    all_trip_infos: List[Tuple[Path, List[float]]] = []

    # Keep looping until we run out of deliveries we can reach
    while remaining:
        trip_idx += 1
        # This is how much we try to carry on this trip (greedy: fill as much as possible)
        carried = min(carry_capacity, sum(demands[d] for d in remaining))

        # New battery (recharged) at the start of each trip
        battery_soc = battery_capacity_kwh
        trip_energy_used = 0.0

        # First sanity: can we even do take-off + minimum landing with this mass?
        takeoff_E = takeoff_energy_kwh_for(carried)
        min_landing_E = landing_energy_kwh_for(0.0)

        if takeoff_E + min_landing_E >= battery_capacity_kwh:
            print(f"\nTrip {trip_idx}: cannot even take off and land with current battery capacity for carried={carried} kg.")
            break

        # Deduct the take-off energy from the SOC
        battery_soc -= takeoff_E
        trip_energy_used += takeoff_E
        total_energy_used += takeoff_E

        current = warehouse
        trip_cells: Path = [warehouse]
        trip_soc_list: List[float] = [battery_soc]

        trip_legs = []
        served_this_trip = []

        # ----- inner routing loop: pick deliveries one by one -----
        while True:
            # We must always leave enough energy to land safely at the end
            landing_E_needed = landing_energy_kwh_for(carried)
            available_for_motion = battery_soc - landing_E_needed

            if available_for_motion <= 0:
                break

            # Scan all remaining deliveries and see which ones we can do
            candidates = []
            for d in remaining:
                dmd = demands[d]
                if dmd > carried:
                    continue
                dist_to_km = dist_km[(current, d)]
                if dist_to_km == float("inf"):
                    continue

                go_e = move_energy_kwh(dist_to_km, carried, cruise_speed_kmh)
                new_load = carried - dmd

                dist_back_km = dist_km[(d, warehouse)]
                if dist_back_km == float("inf"):
                    continue
                back_e = move_energy_kwh(dist_back_km, max(new_load, 0.0), cruise_speed_kmh)

                need = go_e + back_e
                if need <= available_for_motion:
                    candidates.append((dist_to_km, d, go_e, back_e, new_load))

            if not candidates:
                # No new deliveries are safe from here; try to go home if we’re not already there
                if current != warehouse:
                    dist_home_km = dist_km[(current, warehouse)]
                    back_e = move_energy_kwh(dist_home_km, max(carried, 0.0), cruise_speed_kmh)

                    landing_E_needed = landing_energy_kwh_for(carried)
                    available_for_motion = battery_soc - landing_E_needed
                    if back_e > available_for_motion:
                        # If even going home is too expensive, we stop here
                        break

                    p = path[(current, warehouse)]
                    full_segment = p[1:] if p and p[0] == current else p

                    # Spread the energy across the steps, so our battery plot is smooth-ish
                    if full_segment:
                        step_costs = []
                        coords_seq = [current] + full_segment
                        for u, v in zip(coords_seq[:-1], coords_seq[1:]):
                            step_costs.append(step_cost(u, v))
                        total_dist_km = sum(step_costs)

                        for cell, sc in zip(full_segment, step_costs):
                            frac = sc / total_dist_km if total_dist_km > 0 else 0.0
                            dE_cell = back_e * frac
                            battery_soc -= dE_cell
                            trip_energy_used += dE_cell
                            total_energy_used += dE_cell
                            trip_cells.append(cell)
                            trip_soc_list.append(battery_soc)

                    trip_legs.append({
                        'type': 'move',
                        'from': current,
                        'to': warehouse,
                        'distance_km': dist_home_km,
                        'load_before_kg': max(carried, 0.0),
                        'energy_kwh': back_e
                    })
                    current = warehouse

                break

            # Basic routing heuristic: pick the nearest feasible candidate
            candidates.sort(key=lambda x: x[0])
            dist_to_km, target, go_e, back_e, new_load = candidates[0]

            # Move from current -> target
            p = path[(current, target)]
            full_segment = p[1:] if p and p[0] == current else p

            if full_segment:
                step_costs = []
                coords_seq = [current] + full_segment
                for u, v in zip(coords_seq[:-1], coords_seq[1:]):
                    step_costs.append(step_cost(u, v))
                total_dist_km = sum(step_costs)

                for cell, sc in zip(full_segment, step_costs):
                    frac = sc / total_dist_km if total_dist_km > 0 else 0.0
                    dE_cell = go_e * frac
                    battery_soc -= dE_cell
                    trip_energy_used += dE_cell
                    total_energy_used += dE_cell
                    trip_cells.append(cell)
                    trip_soc_list.append(battery_soc)

            trip_legs.append({
                'type': 'move',
                'from': current,
                'to': target,
                'distance_km': dist_to_km,
                'load_before_kg': carried,
                'energy_kwh': go_e
            })
            current = target

            # Drop off the payload at this delivery
            dmd = demands[target]
            carried -= dmd
            served_this_trip.append((target, dmd))
            trip_legs.append({
                'type': 'drop',
                'at': target,
                'demand': dmd,
                'load_after': carried
            })
            remaining.remove(target)

            # If we’re now empty and not at the warehouse, head home empty
            if carried == 0 and current != warehouse:
                dist_home_km = dist_km[(current, warehouse)]
                back_e2 = move_energy_kwh(dist_home_km, 0.0, cruise_speed_kmh)

                landing_E_needed_empty = landing_energy_kwh_for(0.0)
                available_for_motion = battery_soc - landing_E_needed_empty
                if back_e2 > available_for_motion:
                    break

                p = path[(current, warehouse)]
                full_segment = p[1:] if p and p[0] == current else p

                if full_segment:
                    step_costs = []
                    coords_seq = [current] + full_segment
                    for u, v in zip(coords_seq[:-1], coords_seq[1:]):
                        step_costs.append(step_cost(u, v))
                    total_dist_km = sum(step_costs)
                    for cell, sc in zip(full_segment, step_costs):
                        frac = sc / total_dist_km if total_dist_km > 0 else 0.0
                        dE_cell = back_e2 * frac
                        battery_soc -= dE_cell
                        trip_energy_used += dE_cell
                        total_energy_used += dE_cell
                        trip_cells.append(cell)
                        trip_soc_list.append(battery_soc)

                trip_legs.append({
                    'type': 'move',
                    'from': current,
                    'to': warehouse,
                    'distance_km': dist_home_km,
                    'load_before_kg': 0.0,
                    'energy_kwh': back_e2
                })
                current = warehouse
                break

        # End-of-trip landing (with whatever payload is left, usually zero)
        landing_E_final = landing_energy_kwh_for(carried)
        if battery_soc < landing_E_final:
            print(f"Warning: battery_soc < landing_E_final on trip {trip_idx}, clamping at 0.")
            landing_E_final = max(battery_soc, 0.0)
        battery_soc -= landing_E_final
        trip_energy_used += landing_E_final
        total_energy_used += landing_E_final
        trip_legs.append({'type': 'landing', 'energy_kwh': landing_E_final})

        # Some trip summary text so we can see what actually happened
        print(f"\n=== Trip {trip_idx} ===")
        if served_this_trip:
            print("Delivered:", ", ".join(f"{pt} (x{dmd} kg)" for pt, dmd in served_this_trip))
        else:
            print("No deliveries completed on this trip.")

        print("Legs:")
        for i, leg in enumerate(trip_legs, 1):
            if leg['type'] == 'move':
                print(
                    f"  {i:02d}. MOVE {leg['from']} -> {leg['to']}  "
                    f"dist_km={leg['distance_km']:.3f}  "
                    f"load_before_kg={leg['load_before_kg']}  "
                    f"energy_kwh={leg['energy_kwh']:.3f}"
                )
            elif leg['type'] == 'drop':
                print(
                    f"  {i:02d}. DROP at {leg['at']}  "
                    f"demand_kg={leg['demand']}  "
                    f"load_after_kg={leg['load_after']}"
                )
            elif leg['type'] == 'landing':
                print(f"  {i:02d}. LANDING energy_kwh={leg['energy_kwh']:.3f}")

        trip_pct = 100.0 * trip_energy_used / battery_capacity_kwh
        print(f"Trip {trip_idx} energy use: {trip_energy_used:.3f} kWh "
              f"({trip_pct:.1f}% of a {battery_capacity_kwh:.3f} kWh battery)")
        print(f"Battery at end of trip {trip_idx} (before recharge): "
              f"{battery_soc:.3f} kWh ({100.0 * battery_soc / battery_capacity_kwh:.1f}% of pack)")
        print(f"Recharging battery back to {battery_capacity_kwh:.3f} kWh for next trip.\n")

        all_trip_infos.append((trip_cells[:], trip_soc_list[:]))

        if not remaining:
            break

    print("\n=== All trips complete (or as many as feasible) ===")
    print(f"Total energy used across all trips: {total_energy_used:.3f} kWh "
          f"(equivalent to {total_energy_used / battery_capacity_kwh:.2f} full battery cycles)")

    return all_trip_infos


# -----------------------
# Pretty print grid
# -----------------------

def print_grid(rows: int, cols: int, warehouse: Coord, deliveries: List[Coord],
               demands: Dict[Coord, int], blocked: Set[Coord]):
    """
    Print a simple ASCII map of the grid so we can see where everything is.
    """
    grid = [["." for _ in range(cols)] for _ in range(rows)]
    for (r, c) in blocked:
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = "X"
    wr, wc = warehouse
    grid[wr][wc] = "W"
    for d in deliveries:
        if d == warehouse or d in blocked:
            continue
        dr, dc = d
        grid[dr][dc] = str(demands[d])
    print("Grid layout:")
    for r in range(rows):
        print(" ".join(grid[r][c] for c in range(cols)))
    print("\nLegend: W=warehouse | 1–3=delivery demand (kg) | X=no-fly | .=empty\n")


# -----------------------
# Debug: vertical energy table
# -----------------------

def print_vertical_energy_table():
    """
    Just print out how much takeoff/landing energy the ODEs are giving us
    for a few different payloads.
    """
    print("Takeoff / landing energy vs payload mass (from ODEs, scaled):")
    print("  payload_kg | takeoff_kWh | landing_kWh")
    for payload in [0.0, 2.0, 5.0, 10.0]:
        to = takeoff_energy_kwh_for(payload)
        ld = landing_energy_kwh_for(payload)
        print(f"    {payload:7.1f} |    {to:7.4f} |     {ld:7.4f}")
    print()


# -----------------------
# Plot: takeoff profile (with interpolation + root finding on altitude)
# -----------------------

def plot_takeoff_profile():
    """
    Solve the takeoff ODE once, then:
      - Use our home-made linear interpolation to get smooth curves.
      - Use our incremental root finder to find when altitude = 15 m.

    This shows off the ODE + interpolation + root finding combo in one place.
    """
    global M_TAKEOFF, VH_TAKEOFF, g
    M_TAKEOFF = m_tot
    VH_TAKEOFF = np.sqrt((M_TAKEOFF * g) / (2 * rho * A_disk))

    y0 = [0.0, 0.0, E_avail]
    t_span = (0.0, 60.0)

    def event_alt_reached(t, y):
        return y[0] - alt_target

    event_alt_reached.terminal = True
    event_alt_reached.direction = 1

    sol = solve_ivp(
        takeoff_dynamics,
        t_span,
        y0,
        events=event_alt_reached,
        rtol=1e-6,
        atol=1e-8,
        max_step=0.1
    )

    t = sol.t
    z = sol.y[0]
    E = sol.y[2] / 3.6e6  # J -> kWh

    # home-made interpolation
    f_alt = linear_interp1d(t, z)
    f_E = linear_interp1d(t, E)
    t_dense = np.linspace(t[0], t[-1], 300)
    z_dense = f_alt(t_dense)
    E_dense = f_E(t_dense)

    fig, ax1 = plt.subplots()
    ax1.set_title("Takeoff profile (baseline mass, interpolated)")
    ax1.plot(t_dense, z_dense, label="Altitude (m)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Altitude (m)")

    ax2 = ax1.twinx()
    ax2.plot(t_dense, E_dense, linestyle="--", label="Battery (kWh)")
    ax2.set_ylabel("Battery (kWh)")

    # Root-finding on altitude = 15 m (incremental search on interpolated curve)
    target_alt = 15.0

    def g(tval):
        return float(f_alt(tval) - target_alt)

    t_min, t_max = float(t_dense[0]), float(t_dense[-1])
    root_t = incremental_bisection_root(
        g,
        a=t_min,
        b=t_max,
        step=(t_max - t_min) / 50.0,
        tol=1e-3,
        max_iter=60
    )

    if root_t is not None:
        z_hit = float(f_alt(root_t))
        E_hit = float(f_E(root_t))
        print(f"Interpolated altitude {target_alt:.1f} m reached at t ≈ {root_t:.2f} s, "
              f"battery ≈ {E_hit:.4f} kWh")

        ax1.axvline(root_t, color="grey", linestyle=":", linewidth=1)
        ax1.plot(root_t, z_hit, "o", color="red", label=f"{target_alt:.1f} m point")
    else:
        print(f"Could not find a time where altitude reaches {target_alt:.1f} m in the takeoff phase.")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.show()


# -----------------------
# Range & endurance vs payload (with exponential fit)
# -----------------------

def compute_range_and_endurance(payloads, battery_capacity_kwh, speed_kmh):
    """
    For each payload, compute:
      - maximum cruise range (km),
      - endurance in minutes,
    after we pay for takeoff + landing.
    """
    v = speed_kmh / 3.6
    ranges_km = []
    endurances_min = []

    for W in payloads:
        takeoffE = takeoff_energy_kwh_for(W)
        landingE = landing_energy_kwh_for(0.0)   # assume we land empty
        avail_kwh = battery_capacity_kwh - takeoffE - landingE

        if avail_kwh <= 0:
            ranges_km.append(0.0)
            endurances_min.append(0.0)
            continue

        P_W = power_model(W, v, params)
        P_kW = P_W / 1000.0

        endurance_h = avail_kwh / P_kW
        endurance_min = endurance_h * 60.0
        range_km = speed_kmh * endurance_h

        ranges_km.append(range_km)
        endurances_min.append(endurance_min)

    return np.array(ranges_km), np.array(endurances_min)


def plot_range_and_endurance_vs_payload(battery_capacity_kwh, speed_kmh):
    """
    Plot how range and endurance change as we increase payload.
    We overlay an exponential fit for both.
    """
    payloads = np.linspace(0.0, 10.0, 11)
    ranges_km, endurances_min = compute_range_and_endurance(payloads, battery_capacity_kwh, speed_kmh)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Range vs payload + exponential fit
    ax0 = axes[0]
    ax0.plot(payloads, ranges_km, "o", label="Range data")
    f_range, a_r, b_r = exponential_fit(payloads, ranges_km)
    x_fit = np.linspace(payloads[0], payloads[-1], 200)
    y_fit = f_range(x_fit)
    ax0.plot(x_fit, y_fit, "-", label="Exponential fit")
    ax0.set_title("Max range vs payload")
    ax0.set_xlabel("Payload (kg)")
    ax0.set_ylabel("Max range (km)")
    ax0.legend()

    # Endurance vs payload + exponential fit
    ax1 = axes[1]
    ax1.plot(payloads, endurances_min, "o", label="Endurance data")
    f_end, a_e, b_e = exponential_fit(payloads, endurances_min)
    y_fit_e = f_end(x_fit)
    ax1.plot(x_fit, y_fit_e, "-", label="Exponential fit")
    ax1.set_title("Battery endurance vs payload")
    ax1.set_xlabel("Payload (kg)")
    ax1.set_ylabel("Endurance (min)")
    ax1.legend()

    plt.tight_layout()
    plt.show()


# -----------------------
# Plot grid + paths with battery percentage colouring
# -----------------------

def plot_grid_and_paths(
    rows: int,
    cols: int,
    warehouse: Coord,
    deliveries: List[Coord],
    demands: Dict[Coord, int],
    blocked: Set[Coord],
    trip_infos: List[Tuple[Path, List[float]]]
):
    """
    Grid plot:
      - blocked cells, warehouse, deliveries (labelled by demand)
      - each trip path drawn
      - nodes colored by remaining battery (% of full pack)

    This gives a nice visual of “where the energy is going”.
    """
    global BATTERY_CAPACITY_KWH_GLOBAL
    capacity = BATTERY_CAPACITY_KWH_GLOBAL or 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Drone paths on grid (colored by battery %)")

    # Draw the grid lines
    for r in range(rows):
        for c in range(cols):
            rect = Rectangle((c, r), 1, 1, fill=False, edgecolor="lightgrey", linewidth=0.5)
            ax.add_patch(rect)

    # No-fly cells
    for (r, c) in blocked:
        rect = Rectangle((c, r), 1, 1, facecolor="black", alpha=0.3)
        ax.add_patch(rect)

    # Warehouse cell
    wr, wc = warehouse
    ax.add_patch(Rectangle((wc, wr), 1, 1, facecolor="gold", alpha=0.8))
    ax.text(wc + 0.5, wr + 0.5, "W", ha="center", va="center", fontsize=10)

    # Delivery cells, labelled with their payload demand
    for d in deliveries:
        if d == warehouse:
            continue
        dr, dc = d
        label = str(demands.get(d, "?"))
        ax.add_patch(Rectangle((dc, dr), 1, 1, facecolor="skyblue", alpha=0.7))
        ax.text(dc + 0.5, dr + 0.5, label, ha="center", va="center", fontsize=9)

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue", "tab:orange", "tab:green"])

    sc = None  # We’ll use this for the colour bar

    for i, trip in enumerate(trip_infos):
        if not trip:
            continue

        cells, soc_kwh = trip

        if not cells or not soc_kwh:
            continue

        if len(cells) != len(soc_kwh):
            print(f"Warning: trip {i+1} has len(cells)={len(cells)} != len(soc)={len(soc_kwh)}; truncating.")
            n = min(len(cells), len(soc_kwh))
            cells = cells[:n]
            soc_kwh = soc_kwh[:n]

        xs = np.array([c + 0.5 for (r, c) in cells], dtype=float)
        ys = np.array([r + 0.5 for (r, c) in cells], dtype=float)
        soc_kwh = np.maximum(np.array(soc_kwh, dtype=float), 0.0)

        soc_pct = np.clip(100.0 * soc_kwh / capacity, 0.0, 100.0)

        color = colors[i % len(colors)]

        # Path lines
        ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.5, label=f"Trip {i+1}")

        # Scatter points coloured by battery %
        sc = ax.scatter(xs, ys, c=soc_pct, cmap="viridis", s=40, edgecolor="k", vmin=0.0, vmax=100.0)

        # Tiny arrows to show direction of travel
        for (x0, y0, x1, y1) in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color=color,
                alpha=0.8
            )
            ax.add_patch(arrow)

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.grid(False)
    ax.legend(loc="upper right")

    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Battery state of charge (%)")

    plt.tight_layout()
    plt.show()


# -----------------------
# Interpolated continuous trip path
# -----------------------
# This is just for visual flair: we take the discrete grid path and “replay”
# it with time stamps, then interpolate x(t) and y(t) with our own interpolator.

def plot_interpolated_trip_continuous(trip_info: Tuple[Path, List[float]], cruise_speed_kmh: float):
    """
    Take one trip's discrete grid cells and:
      - infer time stamps from step distance and cruise speed
      - build linear-interpolated x(t), y(t) using our own function
      - plot map view + time view
    """
    cells, _ = trip_info
    if len(cells) < 2:
        print("Not enough points in trip to interpolate.")
        return

    v = cruise_speed_kmh / 3.6  # m/s
    xs = np.array([c + 0.5 for (r, c) in cells], dtype=float)
    ys = np.array([r + 0.5 for (r, c) in cells], dtype=float)

    # Build time stamps based on distance (in meters) / speed
    t_nodes = [0.0]
    for (r1, c1), (r2, c2) in zip(cells[:-1], cells[1:]):
        dist_km = step_cost((r1, c1), (r2, c2))
        dist_m = dist_km * 1000.0
        dt = dist_m / v
        t_nodes.append(t_nodes[-1] + dt)
    t_nodes = np.array(t_nodes)

    fx = linear_interp1d(t_nodes, xs)
    fy = linear_interp1d(t_nodes, ys)

    t_dense = np.linspace(t_nodes[0], t_nodes[-1], 200)
    x_dense = fx(t_dense)
    y_dense = fy(t_dense)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax0 = axes[0]
    ax0.set_title("Trip path (discrete vs interpolated)")
    ax0.plot(xs, ys, "o-", label="Grid cells")
    ax0.plot(x_dense, y_dense, "--", label="Interpolated path")
    ax0.set_aspect("equal", adjustable="box")
    ax0.invert_yaxis()
    ax0.set_xlabel("x (grid units)")
    ax0.set_ylabel("y (grid units)")
    ax0.legend()

    ax1 = axes[1]
    ax1.set_title("Continuous position vs time")
    ax1.plot(t_dense, x_dense, label="x(t)")
    ax1.plot(t_dense, y_dense, label="y(t)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (grid units)")
    ax1.legend()

    plt.tight_layout()
    plt.show()


# -----------------------
# Demo / run
# -----------------------
# When you run this file directly, it sets up a random little “city”,
# spawns a random warehouse, two no-fly zones, and some deliveries,
# then flies the drone and shows all the plots.

if __name__ == "__main__":
    rows, cols = 10, 10
    num_deliveries = 7
    carry_capacity = 10       # kg (total payload capacity)

    # Use the actual pack we defined at the top
    battery_capacity = BATTERY_CAPACITY_FROM_CELLS_KWH  # ≈ 0.0799 kWh
    cruise_speed = 40.0       # km/h

    # Set a seed if you want reproducible randomness, or None for different each run
    seed = None
    random.seed(seed)

    BATTERY_CAPACITY_KWH_GLOBAL = battery_capacity

    # --- random warehouse location ---
    warehouse: Coord = (random.randint(0, rows - 1), random.randint(0, cols - 1))

    # --- 2 random no-fly squares (blocked), not overlapping warehouse ---
    blocked: Set[Coord] = set()
    while len(blocked) < 2:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        cell = (r, c)
        if cell == warehouse:
            continue
        blocked.add(cell)

    # --- random deliveries, avoiding warehouse and blocked cells ---
    deliveries: List[Coord] = []
    demands: Dict[Coord, int] = {}
    while len(deliveries) < num_deliveries:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        cell = (r, c)
        if cell == warehouse:
            continue
        if cell in blocked:
            continue
        if cell in deliveries:
            continue
        deliveries.append(cell)
        demands[cell] = random.randint(1, 3)  # payload in kg

    # Show the basic layout
    print_grid(rows, cols, warehouse, deliveries, demands, blocked)

    # ODE-based vertical energy table
    print_vertical_energy_table()

    # Root finding on power balance (incremental search)
    print_crossover_speeds()

    # Takeoff ODE + interpolation + root finding on altitude(t)
    plot_takeoff_profile()

    # Range & endurance vs payload, using pack capacity (with exponential fits)
    plot_range_and_endurance_vs_payload(battery_capacity, cruise_speed)

    # Routing / energy-aware trips with per-trip recharge
    trip_infos = run_all_trips(
        rows=rows,
        cols=cols,
        warehouse=warehouse,
        deliveries=deliveries,
        demands=demands,
        blocked=blocked,
        carry_capacity=carry_capacity,
        battery_capacity_kwh=battery_capacity,
        cruise_speed_kmh=cruise_speed
    )

    # Grid visualization (battery-coloured paths) + interpolated path
    if trip_infos:
        plot_grid_and_paths(rows, cols, warehouse, deliveries, demands, blocked, trip_infos)
        plot_interpolated_trip_continuous(trip_infos[0], cruise_speed)