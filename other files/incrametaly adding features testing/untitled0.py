# drone_routes_energy_opt.py
#
# What this adds:
# - enumerate_feasible_routes: generate all feasible subsets & orders for a trip given capacity/battery
# - plot_routes_energy: matplotlib scatter of "candidate index" vs "energy (kWh)", minimum annotated
# - run_optimized_trips_with_plots: replaces the greedy per-trip selection with an optimal per-trip choice
#
# Usage:
#   python drone_routes_energy_opt.py
#
# Requires: matplotlib

import random
import math
import heapq
from typing import Tuple, List, Dict, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.integrate import solve_ivp

# ----------------------------
# Physical constants & simple takeoff model
# ----------------------------

# Battery parameters (example)
Vb = 22.2               # battery voltage (V)
Cb = 4.5                # battery capacity (Ah)
usable_frac = 0.8       # usable fraction of battery
E_avail = usable_frac * Vb * Cb * 3600  # convert Wh → J

m_frame = 1.5           # frame mass (kg)
m_payload = 0.5         # payload mass (kg)
m_battery = 0.5         # battery mass (kg)
m_tot = m_frame + m_payload + m_battery  # total mass (kg)

# Desired takeoff climb speed target
v_target = 2.0          # m/s (steady climb)
alt_target = 30.0       # m target altitude (stop integration here)

# ----------------------------
# Drone + environment parameters
# ----------------------------
g = 9.81                # gravity (m/s^2)
rho = 1.225             # air density (kg/m^3)
C_d = 1                 # Drag coefficient
A_top = 0.175674        # Effective top area of drone (m^2)
A_disk = 1.34           # total rotor disk area (m^2) ~ six 0.223 m2 rotors
eta = 0.75              # overall efficiency (motor * prop)
P_av = 12               # Avionics power (W)

# Hover velocity approximation
vh = np.sqrt((m_tot * g) / (2 * rho * A_disk))

# Induced power coefficient
k1 = 1.1 / eta
# Parasitic power coefficient (approx)
k2 = 0.5 * rho * C_d * A_top
# Fixed power (avionics etc.)
k3 = P_av

# Pack parameters in a dictionary
params = {
    "g": g,
    "rho": rho,
    "C_d": C_d,
    "A_top": A_top,
    "A_disk": A_disk,
    "eta": eta,
    "m_frame": m_frame,
    "m_payload": m_payload,
    "m_battery": m_battery,
    "m_dry": m_frame + m_battery,  # Dry mass (frame + battery)
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


# ---------------------------------
# Takeoff dynamics
# ---------------------------------
def takeoff_dynamics(t, y):
    """
    Simple vertical takeoff ODE system.
    y[0] = z (altitude)
    y[1] = vz (vertical velocity)
    y[2] = E (remaining battery energy in J)
    """
    z, vz, E = y
    m = m_tot
    v = abs(vz)  # treat upward speed as positive
    # Compute thrust approximately equal to weight in hover
    T = m * g

    # Induced power:
    P_ind = k1 * ((m * g) ** 1.5) / np.sqrt(v ** 2 + vh ** 2)
    # Parasitic / drag power:
    P_drag = k2 * v ** 3
    # Fixed power:
    P_fixed = k3

    P_total = P_ind + P_drag + P_fixed  # total electric power (W)
    # Limit by battery availability:
    if E <= 0:
        P_total = 0.0

    # Simple vertical motion: T - mg = m * dv/dt
    # For a naive model, keep thrust roughly equal to weight plus a small surplus:
    surplus = 0.1 * m * g
    F_net = surplus
    az = F_net / m  # vertical acceleration

    dzdt = vz
    dvzdt = az
    dEdt = -P_total  # Joule per second

    return [dzdt, dvzdt, dEdt]


# ---------------------------------
# Landing dynamics (example)
# ---------------------------------
def landing_dynamics(t, y):
    """
    Simple vertical landing ODE system,
    mirroring takeoff but descending.
    y[0] = z (altitude)
    y[1] = vz (vertical velocity, negative for descent)
    y[2] = E (remaining battery energy in J)
    """
    z, vz, E = y
    m = m_tot
    v = abs(vz)
    T = m * g * 0.9  # slightly less thrust than weight for descent

    P_ind = k1 * ((m * g) ** 1.5) / np.sqrt(v ** 2 + vh ** 2)
    P_drag = k2 * v ** 3
    P_fixed = k3

    P_total = P_ind + P_drag + P_fixed
    if E <= 0:
        P_total = 0.0

    F_net = -0.1 * m * g
    az = F_net / m

    dzdt = vz
    dvzdt = az
    dEdt = -P_total

    return [dzdt, dvzdt, dEdt]


# ---------------------------------------
# Power model for forward flight
# ---------------------------------------
def power_model(W, v, params):
    """
    Compute instantaneous electrical power P(W, v) in watts.
    W : current payload mass (kg)
    v : flight speed (m/s)
    params : dictionary of physical constants
    """
    g   = params["g"]
    m0  = params["m_dry"]     # drone dry mass (frame + battery)
    vh  = params["v_hover"]   # induced velocity in hover
    k1  = params["k1"]        # induced power coefficient
    k2  = params["k2"]        # parasitic drag coefficient
    k3  = params["k3"]        # fixed power load (electronics)

    m = m0 + W
    P_ind   = k1 * ((m * g)**1.5) / np.sqrt(v**2 + vh**2)  # induced power
    P_drag  = k2 * v**3                                    # parasitic/drag power
    P_fixed = k3                                           # fixed onboard power
    return P_ind + P_drag + P_fixed


# ---------------------------------------
# Takeoff / landing energy (kWh)
# ---------------------------------------
def compute_takeoff_energy_kwh():
    """
    Integrate takeoff_dynamics until alt_target and return energy used in kWh.
    Uses the same E_avail and takeoff_dynamics as your plotting code,
    but without plotting.
    """
    y0 = [0.0, 0.0, E_avail]          # z, vz, E
    t_span = (0.0, 60.0)              # generous upper bound

    def event_alt_reached(t, y):
        # Stop when z reaches alt_target
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
        max_step=0.5
    )

    E_end = sol.y[2][-1]
    E_used_J = E_avail - E_end  # Joules
    return E_used_J / 3.6e6     # J → kWh


# Precompute once for the whole script
TAKEOFF_KWH = compute_takeoff_energy_kwh()
# Assume landing is roughly similar energy cost to takeoff
LANDING_KWH = TAKEOFF_KWH


# ---------------------------------------
# Main simulation: energy vs distance
# ---------------------------------------
def simulate_energy_along_path(payload_drops, E0, speed, params):
    """
    Simulate drone battery energy as a function of distance
    while payload mass decreases after each delivery segment.
    This demonstrates that as the drone gets lighter,
    power decreases and the energy curve flattens.
    """
    W = np.sum(payload_drops)   # start with full payload (heaviest)
    E = E0
    s_trace, E_trace, W_trace, P_trace = [0], [E0], [W], []

    # Each flight leg has fixed distance (e.g., 500 m)
    leg_distance = 500
    N_steps = 100
    ds = leg_distance / N_steps
    total_distance = 0

    for drop in payload_drops:
        for _ in range(N_steps):
            v = speed
            P = power_model(W, v, params)
            dE = P * (ds / v)
            E -= dE
            total_distance += ds
            s_trace.append(total_distance)
            E_trace.append(E)
            W_trace.append(W)
            P_trace.append(P)
        W -= drop

    return np.array(s_trace), np.array(E_trace), np.array(W_trace), np.array(P_trace)


# ---------------------------------------
# Simple linearized power model (still used by old bits)
# ---------------------------------------
def power_kw(load_kg: float, base_kw: float = 0.3, alpha_kw_per_kg: float = 0.02, quad_kw_per_kg2: float = 0.0) -> float:
    """
    Base hover/flight power plus linear (and optional quadratic) term in payload mass.
    (Kept for reference; not used by move_energy_kwh anymore.)
    """
    return base_kw + alpha_kw_per_kg * max(load_kg, 0.0) + quad_kw_per_kg2 * max(load_kg, 0.0)**2


def move_energy_kwh(distance_km: float, load_kg: float, speed_kmh: float) -> float:
    """Cruise energy using the physics-based power_model(W, v, params).

    distance_km : path length between two grid points
    load_kg     : payload mass carried on that leg
    speed_kmh   : cruise speed
    """
    if distance_km == float("inf"):
        return float("inf")
    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be > 0")

    v = speed_kmh / 3.6          # m/s
    distance_m = distance_km * 1000.0
    time_s = distance_m / v      # s of cruise

    # power_model already includes drone dry mass, payload, and velocity
    P = power_model(load_kg, v, params)   # watts

    E_J = P * time_s
    return E_J / 3.6e6           # kWh


# -----------------------
# Grid / routing helpers
# -----------------------
Coord = Tuple[int, int]
Path = List[Coord]


def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols


def neighbors(r: int, c: int, rows: int, cols: int) -> List[Coord]:
    return [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    ]


def dijkstra_on_grid(rows: int, cols: int, blocked: Set[Coord], start: Coord) -> Tuple[Dict[Coord, float], Dict[Coord, Coord]]:
    """
    Classic Dijkstra from 'start' to every reachable cell.
    Cost of each move is 1 (then converted to distance in km).
    """
    dist = {start: 0.0}
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
            nd = d + 1.0
            v = (vr, vc)
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev


def reconstruct_path(prev: Dict[Coord, Coord], start: Coord, goal: Coord) -> Optional[Path]:
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


def precompute_pairs(rows: int, cols: int, points: List[Coord], blocked: Set[Coord]) -> Tuple[Dict[Tuple[Coord, Coord], float], Dict[Tuple[Coord, Coord], Path]]:
    """
    Precompute shortest path distances and actual paths among all points in 'points'.
    Distances stored in km.
    """
    dist_km: Dict[Tuple[Coord, Coord], float] = {}
    path_map: Dict[Tuple[Coord, Coord], Path] = {}

    for i, s in enumerate(points):
        d, prev = dijkstra_on_grid(rows, cols, blocked, s)
        for j, t in enumerate(points):
            if s == t:
                dist_km[(s, t)] = 0.0
                path_map[(s, t)] = [s]
            else:
                if t not in d:
                    dist_km[(s, t)] = float("inf")
                    path_map[(s, t)] = []
                else:
                    dist_km[(s, t)] = d[t] / 1000.0
                    pt = reconstruct_path(prev, s, t)
                    if pt is None:
                        dist_km[(s, t)] = float("inf")
                        path_map[(s, t)] = []
                    else:
                        path_map[(s, t)] = pt

    return dist_km, path_map


# -----------------------
# Trip-by-trip planner (updated with takeoff/landing ODE energy)
# -----------------------
def run_all_trips(
    rows: int,
    cols: int,
    warehouse: Coord,
    deliveries: List[Coord],
    demands: Dict[Coord, int],   # kg per delivery
    blocked: Set[Coord],
    carry_capacity: int = 10,    # kg
    battery_capacity_kwh: float = 5.0,
    cruise_speed_kmh: float = 40.0
):
    points = [warehouse] + deliveries
    dist_km, path = precompute_pairs(rows, cols, points, blocked)

    remaining = []
    impossible = []
    for d in deliveries:
        if demands[d] > carry_capacity:
            impossible.append(d)
        elif dist_km[(warehouse, d)] == float("inf") or dist_km[(d, warehouse)] == float("inf"):
            impossible.append(d)
        else:
            remaining.append(d)

    if impossible:
        print("These deliveries are impossible (capacity or obstacles):", impossible)

    trip_idx = 0
    grand_total_energy = 0.0

    while remaining:
        trip_idx += 1

        # Per-trip takeoff / landing overhead from the ODE model
        takeoff_E = TAKEOFF_KWH
        landing_E = LANDING_KWH

        if takeoff_E + landing_E >= battery_capacity_kwh:
            print(f"\nTrip {trip_idx}: cannot even take off and land with current battery capacity.")
            break

        # Energy available for cruise + return (landing energy reserved logically)
        energy = battery_capacity_kwh - takeoff_E
        grand_total_energy += takeoff_E

        carried = min(carry_capacity, sum(demands[d] for d in remaining))
        current = warehouse
        trip_path: Path = [warehouse]
        trip_legs = []

        # Record takeoff as the first leg
        trip_legs.append({'type': 'takeoff', 'energy_kwh': takeoff_E})

        served_this_trip = []

        while True:
            # Always reserve landing_E so we can land safely at the end
            reserved_for_landing = landing_E
            available_for_motion = energy - reserved_for_landing

            if available_for_motion <= 0:
                # No energy left for any more movement; end trip
                break

            candidates = []
            for d in remaining:
                dmd = demands[d]
                if dmd > carried:
                    continue
                dist_to_km = dist_km[(current, d)]
                if dist_to_km == float("inf"):
                    continue

                # Outbound leg with current carried load
                go_e = move_energy_kwh(dist_to_km, carried, cruise_speed_kmh)

                # After delivery at d, new load
                new_load = carried - dmd

                # Energy to go home from d with lighter load
                dist_back_km = dist_km[(d, warehouse)]
                if dist_back_km == float("inf"):
                    continue
                back_e = move_energy_kwh(dist_back_km, max(new_load, 0.0), cruise_speed_kmh)

                # We only require enough for cruise; landing_E is reserved separately
                need = go_e + back_e
                if need <= available_for_motion:
                    candidates.append((dist_to_km, d, go_e, back_e, new_load))

            if not candidates:
                # Try to go home directly if we're not already there
                if current != warehouse:
                    dist_home_km = dist_km[(current, warehouse)]
                    back_e = move_energy_kwh(dist_home_km, max(carried, 0.0), cruise_speed_kmh)

                    reserved_for_landing = landing_E
                    available_for_motion = energy - reserved_for_landing
                    if back_e > available_for_motion:
                        # Can't safely return; trip ends wherever we are
                        break

                    p = path[(current, warehouse)]
                    trip_path += (p[1:] if p and p[0] == current else p)
                    trip_legs.append({
                        'type': 'move',
                        'from': current,
                        'to': warehouse,
                        'distance_km': dist_home_km,
                        'load_before_kg': max(carried, 0.0),
                        'energy_kwh': back_e
                    })
                    energy -= back_e
                    grand_total_energy += back_e
                    current = warehouse

                # end of this trip
                break

            # Pick nearest feasible candidate
            candidates.sort(key=lambda x: x[0])
            dist_to_km, target, go_e, back_e, new_load = candidates[0]

            # Fly from current -> target
            p = path[(current, target)]
            trip_path += (p[1:] if p and p[0] == current else p)

            trip_legs.append({
                'type': 'move',
                'from': current,
                'to': target,
                'distance_km': dist_to_km,
                'load_before_kg': carried,
                'energy_kwh': go_e
            })
            energy -= go_e
            grand_total_energy += go_e
            current = target

            # Deliver payload at target
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

            # If empty and not at warehouse, go home empty (still respecting landing reserve)
            if carried == 0 and current != warehouse:
                dist_home_km = dist_km[(current, warehouse)]
                back_e2 = move_energy_kwh(dist_home_km, 0.0, cruise_speed_kmh)

                reserved_for_landing = landing_E
                available_for_motion = energy - reserved_for_landing
                if back_e2 > available_for_motion:
                    # Can't safely return; end trip here
                    break

                p = path[(current, warehouse)]
                trip_path += (p[1:] if p and p[0] == current else p)
                trip_legs.append({
                    'type': 'move',
                    'from': current,
                    'to': warehouse,
                    'distance_km': dist_home_km,
                    'load_before_kg': 0.0,
                    'energy_kwh': back_e2
                })
                energy -= back_e2
                grand_total_energy += back_e2
                current = warehouse
                break

        # End-of-trip landing
        trip_legs.append({'type': 'landing', 'energy_kwh': landing_E})
        grand_total_energy += landing_E
        energy -= landing_E

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
            elif leg['type'] == 'takeoff':
                print(f"  {i:02d}. TAKEOFF energy_kwh={leg['energy_kwh']:.3f}")
            elif leg['type'] == 'landing':
                print(f"  {i:02d}. LANDING energy_kwh={leg['energy_kwh']:.3f}")

        print(f"Trip {trip_idx} ended at: {current}")
        print("Remaining deliveries:", remaining if remaining else "None")

        # Check if any future trip is feasible with remaining deliveries
        if remaining:
            can_start = False
            start_load = min(carry_capacity, sum(demands[x] for x in remaining))
            max_motion_energy = max(
                0.0, battery_capacity_kwh - (TAKEOFF_KWH + LANDING_KWH)
            )

            for d in remaining:
                dmd = demands[d]
                if dmd > carry_capacity:
                    continue
                dist_to_km = dist_km[(warehouse, d)]
                dist_back_km = dist_km[(d, warehouse)]
                if dist_to_km == float("inf") or dist_back_km == float("inf"):
                    continue

                go_e = move_energy_kwh(dist_to_km, start_load, cruise_speed_kmh)
                back_e = move_energy_kwh(dist_back_km, max(start_load - dmd, 0.0), cruise_speed_kmh)

                if go_e + back_e <= max_motion_energy:
                    can_start = True
                    break

            if not can_start:
                print("\nSome deliveries are impossible with current obstacles/capacity/battery:", remaining)
                break

    print("\n=== All trips complete (or as many as feasible) ===")
    print(f"Grand total energy (kWh) including takeoff/landing: {grand_total_energy:.3f}")


# -----------------------
# Pretty print grid
# -----------------------
def print_grid(rows: int, cols: int, warehouse: Coord, deliveries: List[Coord],
               demands: Dict[Coord, int], blocked: Set[Coord]):
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
# Demo / run
# -----------------------
if __name__ == "__main__":
    rows, cols = 10, 10
    num_deliveries = 7
    carry_capacity = 10       # kg
    battery_capacity = 5.0    # kWh
    cruise_speed = 40.0       # km/h
    seed = None

    blocked = set()
    random.seed(seed)

    warehouse = (rows // 2, cols // 2)
    deliveries = []
    demands: Dict[Coord, int] = {}
    while len(deliveries) < num_deliveries:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if (r, c) == warehouse:
            continue
        if (r, c) in blocked:
            continue
        if (r, c) in deliveries:
            continue
        deliveries.append((r, c))
        demands[(r, c)] = random.randint(1, 3)

    print_grid(rows, cols, warehouse, deliveries, demands, blocked)

    run_all_trips(
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
