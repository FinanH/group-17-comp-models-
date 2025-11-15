# dynamics.py
import math
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from config import (
    E_avail, m_tot, m_frame, m_battery, m_payload_nom,
    g, rho, C_dx, C_dz, A_front, A_top, A_disk, eta, P_av, params
)
from math_utils import linear_interp1d, incremental_bisection_root

# Mission / motion targets & angles
alt_target = 30.0       # m
v_t_target = 2.0        # m/s
v_l_target = -0.5       # m/s
vx_target = 18.0        # m/s (~65 km/h)

x_target = 2000.0       # horizontal mission distance (m)
z_target = 0.0
transition_time = 3.0   # s

theta_cruise = np.deg2rad(25)
theta_brake = np.deg2rad(-20)
theta_level = 0.0

# Globals for vertical ODEs
M_TAKEOFF = m_tot
VH_TAKEOFF = math.sqrt((M_TAKEOFF * g) / (2 * rho * A_disk))
M_LANDING = m_tot
VH_LANDING = VH_TAKEOFF

# Baseline energies
LANDING_KWH_BASE: float
TAKEOFF_KWH_BASE: Optional[float] = None

# Globals for braking
d_brake: float
x_cruise_start: float


# -----------------------------
# Vertical ODEs
# -----------------------------
def takeoff_dynamics(t, y):
    z, vz, E = y
    k_p = 6.0
    Thrust = m_tot * (g + k_p * (v_t_target - vz))
    Drag_z = 0.5 * rho * C_dz * A_top * vz * abs(vz)

    vi = math.sqrt(Thrust / (2 * rho * A_disk))
    P_ind = Thrust * vi
    P_elec = (P_ind / eta) + P_av

    dzdt = vz
    dvzdt = (Thrust - (m_tot * g) - Drag_z) / m_tot
    dEdt = -P_elec
    return [dzdt, dvzdt, dEdt]


def landing_dynamics(t, y):
    z, vz, E = y
    k_p = 0.5
    Thrust = max(0.0, m_tot * (g + k_p * (v_l_target - vz)))
    Drag_z = 0.5 * rho * C_dz * A_top * vz * abs(vz)

    vi = math.sqrt(Thrust / (2 * rho * A_disk))
    P_ind = Thrust * vi
    P_elec = (P_ind / eta) + P_av

    dzdt = vz
    dvzdt = (Thrust - (m_tot * g) - Drag_z) / m_tot
    dEdt = -P_elec
    return [dzdt, dvzdt, dEdt]


# -----------------------------
# Power model
# -----------------------------
def power_model(W, v, params_dict):
    g_ = params_dict["g"]
    m0 = params_dict["m_dry"]
    vh0 = params_dict["v_hover"]
    k1p = params_dict["k1"]
    k2p = params_dict["k2"]
    k3p = params_dict["k3"]

    m = m0 + W
    P_ind = k1p * ((m * g_) ** 1.5) / np.sqrt(v ** 2 + vh0 ** 2)
    P_drag = k2p * v ** 3
    P_fixed = k3p
    return P_ind + P_drag + P_fixed


# -----------------------------
# Takeoff / landing energies
# -----------------------------
def compute_landing_energy_kwh():
    global M_LANDING, VH_LANDING
    M_LANDING = m_tot
    VH_LANDING = math.sqrt((M_LANDING * g) / (2 * rho * A_disk))

    y0 = [alt_target, 0.0, E_avail]
    t_span = (0.0, 60.0)

    def event_ground_reached(t, y):
        return y[0]

    event_ground_reached.terminal = True
    event_ground_reached.direction = -1

    sol = solve_ivp(
        landing_dynamics,
        t_span,
        y0,
        events=event_ground_reached,
        rtol=1e-6,
        atol=1e-8,
        max_step=0.5,
    )

    E_end = sol.y[2][-1]
    E_used_J = E_avail - E_end
    return E_used_J / 3.6e6


LANDING_KWH_BASE = compute_landing_energy_kwh()


def landing_energy_kwh_for(payload_kg: float) -> float:
    total_mass = m_frame + m_battery + payload_kg
    return LANDING_KWH_BASE * (total_mass / m_tot)


def takeoff_energy_kwh_for(payload_kg: float) -> float:
    global TAKEOFF_KWH_BASE, M_TAKEOFF, VH_TAKEOFF

    if TAKEOFF_KWH_BASE is None:
        M_TAKEOFF = m_tot
        VH_TAKEOFF = math.sqrt((M_TAKEOFF * g) / (2 * rho * A_disk))

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
            max_step=0.1,
        )

        E_end = sol.y[2][-1]
        E_used_J = E_avail - E_end
        TAKEOFF_KWH_BASE = E_used_J / 3.6e6

    total_mass = m_frame + m_battery + payload_kg
    return TAKEOFF_KWH_BASE * (total_mass / m_tot)


# -----------------------------
# Braking distance (horizontal)
# -----------------------------
def braking_distance_ivp(v0, theta_b):
    def dyn(t, y):
        v, x = y
        Drag_x = 0.5 * rho * C_dx * A_front * max(v, 0) ** 2
        F_thrust_x = -m_tot * g * math.sin(abs(theta_b))
        a = (F_thrust_x - Drag_x) / m_tot
        return [a, v]

    def stop_v0(t, y):
        return y[0]

    stop_v0.terminal = True
    stop_v0.direction = -1

    sol = solve_ivp(dyn, (0, 100), [v0, 0.0], events=stop_v0, max_step=0.1)
    return sol.y[1, -1]


# compute braking distance and cruise start on import
d_brake = braking_distance_ivp(vx_target, theta_brake)
x_cruise_start = x_target - d_brake
print(f"Active braking distance ≈ {d_brake:.1f} m | Begin braking at x ≈ {x_cruise_start:.1f} m")


# -----------------------------
# Drone horizontal ODE
# -----------------------------
def move_energy(t, y):
    global z_target, x_target, x_cruise_start
    z, vz, x, vx, E = y

    Drag_x = 0.5 * rho * C_dx * A_front * vx ** 2
    Drag_z = 0.5 * rho * C_dz * A_top * vz ** 2

    if x < x_cruise_start:
        theta = theta_cruise
    elif x < x_target:
        theta = theta_brake if vx > 0 else 0.0
    else:
        theta = theta_level

    k_p_z = 10.0
    Thrust = m_tot * (g + k_p_z * (z_target - z)) / math.cos(theta)

    vi = math.sqrt(Thrust / (2 * rho * A_disk))
    P_ind = Thrust * vi
    P_par = abs(Drag_x * vx)
    P_elec = (P_ind + P_par) / eta + P_av

    dzdt = vz
    dvzdt = (Thrust * math.cos(theta) - m_tot * g - Drag_z) / m_tot
    dxdt = vx
    dvxdt = (Thrust * math.sin(theta) - Drag_x) / m_tot
    dEdt = -P_elec
    return [dzdt, dvzdt, dxdt, dvxdt, dEdt]


def move_energy_kwh(distance_km: float, payload_kg: float, cruise_speed_kmh: float) -> float:
    global x_target, x_cruise_start, z_target

    if distance_km <= 0.0 or cruise_speed_kmh <= 0.0:
        return 0.0

    distance_m = distance_km * 1000.0
    v = cruise_speed_kmh / 3.6

    old_x_target = x_target
    old_x_cruise_start = x_cruise_start
    old_z_target = z_target

    try:
        z_target = alt_target
        x_target = 1e9
        x_cruise_start = 1e9

        y0 = [alt_target, 0.0, 0.0, v, E_avail]

        def event_reach_distance(t, y):
            return y[2] - distance_m

        event_reach_distance.terminal = True
        event_reach_distance.direction = 1

        t_end = distance_m / max(v, 0.1) * 5.0

        sol = solve_ivp(
            move_energy,
            (0.0, t_end),
            y0,
            events=event_reach_distance,
            rtol=1e-6,
            atol=1e-8,
            max_step=0.5,
        )

        if sol.t_events[0].size == 0:
            P_W = power_model(payload_kg, v, params)
            P_kW = P_W / 1000.0
            time_h = distance_km / cruise_speed_kmh
            return P_kW * time_h

        E_end = sol.y[4, -1]
        E_used_J = E_avail - E_end
        E_base_kwh = max(E_used_J, 0.0) / 3.6e6
    finally:
        x_target = old_x_target
        x_cruise_start = old_x_cruise_start
        z_target = old_z_target

    total_mass = m_frame + m_battery + payload_kg
    mass_ratio = total_mass / m_tot
    return E_base_kwh * mass_ratio


# -----------------------------
# Power crossover speed
# -----------------------------
def crossover_speed_ms_for_payload(payload_kg: float) -> Optional[float]:
    g_ = params["g"]
    m0 = params["m_dry"]
    vh0 = params["v_hover"]
    k1p = params["k1"]
    k2p = params["k2"]

    m = m0 + payload_kg

    def f(v):
        if v <= 0:
            return 1e6
        P_ind = k1p * ((m * g_) ** 1.5) / np.sqrt(v ** 2 + vh0 ** 2)
        P_drag = k2p * v ** 3
        return P_drag - P_ind

    return incremental_bisection_root(f, a=0.1, b=40.0, step=0.5, tol=1e-3, max_iter=60)


def print_crossover_speeds():
    print("Speed where induced and drag power are equal (incremental-search root-finding):")
    for payload in [0.0, 2.0, 5.0, 10.0]:
        v = crossover_speed_ms_for_payload(payload)
        if v is None:
            print(f"  payload {payload:.1f} kg: no root found in [0.1, 40] m/s")
        else:
            print(f"  payload {payload:.1f} kg: v ≈ {v:.2f} m/s ({v*3.6:.1f} km/h)")
    print()


# -----------------------------
# Vertical energy table
# -----------------------------
def print_vertical_energy_table():
    print("Takeoff / landing energy vs payload mass (from ODEs, scaled):")
    print("  payload_kg | takeoff_kWh | landing_kWh")
    for payload in [0.0, 2.0, 5.0, 10.0]:
        to = takeoff_energy_kwh_for(payload)
        ld = landing_energy_kwh_for(payload)
        print(f"    {payload:7.1f} |    {to:7.4f} |     {ld:7.4f}")
    print()


# -----------------------------
# Range / endurance helper
# -----------------------------
def compute_range_and_endurance(payloads, battery_capacity_kwh, speed_kmh):
    v = speed_kmh / 3.6
    ranges_km = []
    endurances_min = []

    for W in payloads:
        takeoffE = takeoff_energy_kwh_for(W)
        landingE = landing_energy_kwh_for(0.0)
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
