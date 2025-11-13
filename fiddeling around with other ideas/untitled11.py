# drone_routes_energy_opt.py
#
# Drone grid-delivery simulation with:
# - Physics-based power model for cruise (power_model)
# - Takeoff/landing energy from ODE, scaled with payload mass
# - Battery-aware routing with obstacles
# - Diagonal grid movement (8 directions)
# - Computational methods:
#     * ODE integration (solve_ivp)
#     * Root finding (root_scalar) to find crossover speed
#     * Root finding on interpolated altitude to find time for given height
#     * Interpolation (interp1d) for smooth takeoff curves
#     * Interpolation (interp1d) for continuous trip path (position vs time)
# - Plots:
#     * Takeoff profile (altitude & battery vs time, interpolated)
#     * Max range vs payload
#     * Battery endurance vs payload
#     * Grid map showing paths, nodes colored by remaining energy (%)
#     * Interpolated continuous trip path (x,y vs time)
#
# Copy-paste this as a single file.

import random
import math
import heapq
from typing import Tuple, List, Dict, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from matplotlib.patches import Rectangle, FancyArrowPatch

# ----------------------------
# Physical constants & battery
# ----------------------------

Vb = 22.2               # battery voltage (V)
Cb = 4.5                # battery capacity (Ah)
usable_frac = 0.8       # usable fraction of battery
E_avail = usable_frac * Vb * Cb * 3600  # Wh -> J

# Masses
m_frame = 1.5           # frame mass (kg)
m_payload_nom = 0.5     # nominal payload for baseline takeoff (kg)
m_battery = 0.5         # battery mass (kg)
m_tot = m_frame + m_payload_nom + m_battery

# Vertical target
v_target = 2.0          # m/s
alt_target = 30.0       # m

# ----------------------------
# Drone + environment parameters
# ----------------------------
g = 9.81
rho = 1.225
C_d = 1.0
A_top = 0.175674
A_disk = 1.34
eta = 0.75
P_av = 12.0

vh = np.sqrt((m_tot * g) / (2 * rho * A_disk))  # hover induced velocity

k1 = 1.1 / eta
k2 = 0.5 * rho * C_d * A_top
k3 = P_av

params = {
    "g": g,
    "rho": rho,
    "C_d": C_d,
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

# Globals for vertical ODE
M_TAKEOFF = m_tot
VH_TAKEOFF = vh
M_LANDING = m_tot
VH_LANDING = vh


# ---------------------------------
# Takeoff / landing dynamics (ODE)
# ---------------------------------
def takeoff_dynamics(t, y):
    """ODE for vertical takeoff: y = [z, vz, E]."""
    z, vz, E = y
    m = M_TAKEOFF
    v = abs(vz)

    P_ind = k1 * ((m * g) ** 1.5) / np.sqrt(v ** 2 + VH_TAKEOFF ** 2)
    P_drag = k2 * v ** 3
    P_fixed = k3
    P_total = P_ind + P_drag + P_fixed
    if E <= 0:
        P_total = 0.0

    surplus = 0.1 * m * g
    az = surplus / m

    dzdt = vz
    dvzdt = az
    dEdt = -P_total
    return [dzdt, dvzdt, dEdt]


def landing_dynamics(t, y):
    """ODE for vertical landing (mirror of takeoff)."""
    z, vz, E = y
    m = M_LANDING
    v = abs(vz)

    P_ind = k1 * ((m * g) ** 1.5) / np.sqrt(v ** 2 + VH_LANDING ** 2)
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
# Forward-flight power
# ---------------------------------------
def power_model(W, v, params):
    """Power in forward flight, including induced + drag + fixed."""
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
    return P_ind + P_drag + P_fixed


# ---------------------------------------
# Takeoff / landing energy (uses ODE)
# ---------------------------------------
def compute_takeoff_energy_kwh():
    """
    Integrate takeoff_dynamics from z=0 to alt_target using solve_ivp
    to get baseline takeoff energy (ODE example).
    """
    global M_TAKEOFF, VH_TAKEOFF
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
        max_step=0.5
    )

    E_end = sol.y[2][-1]
    E_used_J = E_avail - E_end
    return E_used_J / 3.6e6  # J -> kWh


TAKEOFF_KWH_BASE = compute_takeoff_energy_kwh()


def takeoff_energy_kwh_for(payload_kg: float) -> float:
    total_mass = m_frame + m_battery + payload_kg
    return TAKEOFF_KWH_BASE * (total_mass / m_tot)


def landing_energy_kwh_for(payload_kg: float) -> float:
    return takeoff_energy_kwh_for(payload_kg)


# ---------------------------------------
# Cruise energy
# ---------------------------------------
def move_energy_kwh(distance_km: float, load_kg: float, speed_kmh: float) -> float:
    if distance_km == float("inf"):
        return float("inf")
    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be > 0")

    v = speed_kmh / 3.6
    distance_m = distance_km * 1000.0
    time_s = distance_m / v

    P = power_model(load_kg, v, params)
    E_J = P * time_s
    return E_J / 3.6e6


# -----------------------
# Grid / routing helpers
# -----------------------
Coord = Tuple[int, int]
Path = List[Coord]


def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols


def neighbors(r: int, c: int, rows: int, cols: int) -> List[Coord]:
    # 8-neighbour (diagonal allowed)
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
    dr = v[0] - u[0]
    dc = v[1] - u[1]
    if dr != 0 and dc != 0:
        return math.sqrt(2.0)
    return 1.0


def dijkstra_on_grid(rows: int, cols: int, blocked: Set[Coord], start: Coord):
    dist: Dict[Coord, float] = {start: 0.0}
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

            sc = step_cost(u, (vr, vc))
            nd = d + sc
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


def precompute_pairs(rows: int, cols: int, points: List[Coord], blocked: Set[Coord]):
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
                    dist_km[(s, t)] = d[t] / 1000.0  # grid units -> "m" -> km
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
def crossover_speed_ms_for_payload(payload_kg: float) -> Optional[float]:
    """
    Use root finding to solve P_drag(v) - P_ind(v) = 0
    => finds the speed where drag power equals induced power.
    This uses root_scalar (Brent) = ROOT FINDING example.
    """
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

    try:
        sol = root_scalar(f, bracket=[0.1, 40.0], method="brentq")
        return sol.root if sol.converged else None
    except ValueError:
        return None


def print_crossover_speeds():
    """Print crossover speeds found via root finding for various payloads."""
    print("Speed where induced and drag power are equal (root-finding):")
    for payload in [0.0, 2.0, 5.0, 10.0]:
        v = crossover_speed_ms_for_payload(payload)
        if v is None:
            print(f"  payload {payload:.1f} kg: no root found")
        else:
            print(f"  payload {payload:.1f} kg: v ≈ {v:.2f} m/s ({v*3.6:.1f} km/h)")
    print()


# -----------------------
# Trip-by-trip planner (returns paths + energy per step)
# -----------------------
def run_all_trips(
    rows: int,
    cols: int,
    warehouse: Coord,
    deliveries: List[Coord],
    demands: Dict[Coord, int],
    blocked: Set[Coord],
    carry_capacity: int = 10,
    battery_capacity_kwh: float = 5.0,
    cruise_speed_kmh: float = 40.0
) -> List[Tuple[Path, List[float]]]:
    """
    Run all trips.
    Returns a list of (trip_cells, trip_energy_kwh) where:
      trip_cells[i]       = grid coord of step i
      trip_energy_kwh[i]  = remaining motion energy after reaching that cell (kWh)
    """
    points = [warehouse] + deliveries
    dist_km, path = precompute_pairs(rows, cols, points, blocked)

    remaining: List[Coord] = []
    impossible: List[Coord] = []
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
    all_trip_infos: List[Tuple[Path, List[float]]] = []

    while remaining:
        trip_idx += 1
        carried = min(carry_capacity, sum(demands[d] for d in remaining))

        takeoff_E = takeoff_energy_kwh_for(carried)
        landing_reserve_max = landing_energy_kwh_for(carried)

        if takeoff_E + landing_reserve_max >= battery_capacity_kwh:
            print(f"\nTrip {trip_idx}: cannot even take off and land with current battery capacity for carried={carried} kg.")
            break

        # Motion energy after takeoff (for coloring / tracking)
        energy = battery_capacity_kwh - takeoff_E
        grand_total_energy += takeoff_E

        current = warehouse
        trip_cells: Path = [warehouse]
        trip_energy: List[float] = [energy]

        trip_legs = []
        served_this_trip = []

        # ----- inner routing loop -----
        while True:
            landing_E_needed = landing_energy_kwh_for(carried)
            reserved_for_landing = landing_E_needed
            available_for_motion = energy - reserved_for_landing

            if available_for_motion <= 0:
                break

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
                # Try to go home directly
                if current != warehouse:
                    dist_home_km = dist_km[(current, warehouse)]
                    back_e = move_energy_kwh(dist_home_km, max(carried, 0.0), cruise_speed_kmh)

                    landing_E_needed = landing_energy_kwh_for(carried)
                    reserved_for_landing = landing_E_needed
                    available_for_motion = energy - reserved_for_landing
                    if back_e > available_for_motion:
                        break

                    p = path[(current, warehouse)]
                    full_segment = p[1:] if p and p[0] == current else p

                    if full_segment:
                        step_costs = []
                        coords_seq = [current] + full_segment
                        for u, v in zip(coords_seq[:-1], coords_seq[1:]):
                            step_costs.append(step_cost(u, v))
                        total_dist = sum(step_costs)

                        for cell, sc in zip(full_segment, step_costs):
                            frac = sc / total_dist if total_dist > 0 else 0.0
                            dE_cell = back_e * frac
                            energy -= dE_cell
                            trip_cells.append(cell)
                            trip_energy.append(energy)

                    trip_legs.append({
                        'type': 'move',
                        'from': current,
                        'to': warehouse,
                        'distance_km': dist_home_km,
                        'load_before_kg': max(carried, 0.0),
                        'energy_kwh': back_e
                    })
                    grand_total_energy += back_e
                    current = warehouse

                break

            # Pick nearest feasible candidate
            candidates.sort(key=lambda x: x[0])
            dist_to_km, target, go_e, back_e, new_load = candidates[0]

            # Path current -> target
            p = path[(current, target)]
            full_segment = p[1:] if p and p[0] == current else p

            if full_segment:
                step_costs = []
                coords_seq = [current] + full_segment
                for u, v in zip(coords_seq[:-1], coords_seq[1:]):
                    step_costs.append(step_cost(u, v))
                total_dist = sum(step_costs)
                for cell, sc in zip(full_segment, step_costs):
                    frac = sc / total_dist if total_dist > 0 else 0.0
                    dE_cell = go_e * frac
                    energy -= dE_cell
                    trip_cells.append(cell)
                    trip_energy.append(energy)

            trip_legs.append({
                'type': 'move',
                'from': current,
                'to': target,
                'distance_km': dist_to_km,
                'load_before_kg': carried,
                'energy_kwh': go_e
            })
            grand_total_energy += go_e
            current = target

            # Deliver payload
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

            # If empty payload and not at warehouse, go home
            if carried == 0 and current != warehouse:
                dist_home_km = dist_km[(current, warehouse)]
                back_e2 = move_energy_kwh(dist_home_km, 0.0, cruise_speed_kmh)

                landing_E_needed_empty = landing_energy_kwh_for(0.0)
                reserved_for_landing = landing_E_needed_empty
                available_for_motion = energy - reserved_for_landing
                if back_e2 > available_for_motion:
                    break

                p = path[(current, warehouse)]
                full_segment = p[1:] if p and p[0] == current else p

                if full_segment:
                    step_costs = []
                    coords_seq = [current] + full_segment
                    for u, v in zip(coords_seq[:-1], coords_seq[1:]):
                        step_costs.append(step_cost(u, v))
                    total_dist = sum(step_costs)
                    for cell, sc in zip(full_segment, step_costs):
                        frac = sc / total_dist if total_dist > 0 else 0.0
                        dE_cell = back_e2 * frac
                        energy -= dE_cell
                        trip_cells.append(cell)
                        trip_energy.append(energy)

                trip_legs.append({
                    'type': 'move',
                    'from': current,
                    'to': warehouse,
                    'distance_km': dist_home_km,
                    'load_before_kg': 0.0,
                    'energy_kwh': back_e2
                })
                grand_total_energy += back_e2
                current = warehouse
                break

        # Landing energy (not mapped to a specific cell)
        landing_E_final = landing_energy_kwh_for(carried)
        trip_legs.append({'type': 'landing', 'energy_kwh': landing_E_final})
        grand_total_energy += landing_E_final
        energy -= landing_E_final

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

        print(f"Trip {trip_idx} ended at: {current}")
        print("Remaining deliveries:", remaining if remaining else "None")

        all_trip_infos.append((trip_cells[:], trip_energy[:]))

        # Check possible future trips
        if remaining:
            can_start = False
            start_load = min(carry_capacity, sum(demands[x] for x in remaining))
            takeoff_next = takeoff_energy_kwh_for(start_load)
            landing_next = landing_energy_kwh_for(0.0)
            max_motion_energy = max(0.0, battery_capacity_kwh - (takeoff_next + landing_next))

            if max_motion_energy <= 0:
                print("\nNo energy left for motion after takeoff/landing on any future trip.")
                break

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

    return all_trip_infos


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
# Debug: vertical energy table
# -----------------------
def print_vertical_energy_table():
    print("Takeoff / landing energy vs payload mass:")
    print("  payload_kg | takeoff_kWh | landing_kWh")
    for payload in [0.0, 2.0, 5.0, 10.0]:
        to = takeoff_energy_kwh_for(payload)
        ld = landing_energy_kwh_for(payload)
        print(f"    {payload:7.1f} |    {to:7.3f} |     {ld:7.3f}")
    print()


# -----------------------
# Plot: takeoff profile (with interpolation + root finding on altitude)
# -----------------------
def plot_takeoff_profile():
    """
    Solve ODE for takeoff (solve_ivp), then use interp1d to
    reconstruct smoother altitude & battery curves.

    Also uses root finding on the interpolated altitude(t) to find
    the time when altitude = 15 m.
    """
    global M_TAKEOFF, VH_TAKEOFF
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
    E = sol.y[2] / 3.6e6

    # --- interpolation / recreation of a smooth curve from discrete ODE output ---
    f_alt = interp1d(t, z, kind="cubic")
    f_E = interp1d(t, E, kind="cubic")
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

    # Root-finding on interpolated altitude: find t where altitude = 15 m
    target_alt = 15.0

    def g(tval):
        return float(f_alt(tval) - target_alt)

    try:
        sol_root = root_scalar(g, bracket=[t_dense[0], t_dense[-1]], method="brentq")
        if sol_root.converged:
            t_hit = sol_root.root
            z_hit = float(f_alt(t_hit))
            E_hit = float(f_E(t_hit))
            print(f"Interpolated altitude {target_alt:.1f} m reached at t ≈ {t_hit:.2f} s, battery ≈ {E_hit:.3f} kWh")

            # Mark this point on the plot
            ax1.axvline(t_hit, color="grey", linestyle=":", linewidth=1)
            ax1.plot(t_hit, z_hit, "o", color="red", label=f"{target_alt:.1f} m point")
    except ValueError:
        print(f"Could not find a time where altitude reaches {target_alt:.1f} m in the takeoff phase.")

    # Update legend including the special point
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.show()


# -----------------------
# Range & endurance vs payload
# -----------------------
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


def plot_range_and_endurance_vs_payload(battery_capacity_kwh, speed_kmh):
    payloads = np.linspace(0.0, 10.0, 11)
    ranges_km, endurances_min = compute_range_and_endurance(payloads, battery_capacity_kwh, speed_kmh)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(payloads, ranges_km, marker="o")
    axes[0].set_title("Max range vs payload")
    axes[0].set_xlabel("Payload (kg)")
    axes[0].set_ylabel("Max range (km)")

    axes[1].plot(payloads, endurances_min, marker="o")
    axes[1].set_title("Battery endurance vs payload")
    axes[1].set_xlabel("Payload (kg)")
    axes[1].set_ylabel("Endurance (min)")

    plt.tight_layout()
    plt.show()


# -----------------------
# Plot grid + paths with energy percentage coloring
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
      - nodes colored by remaining motion energy (% of trip's initial motion energy)
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Drone paths on grid (colored by remaining energy %)")

    # background grid
    for r in range(rows):
        for c in range(cols):
            rect = Rectangle((c, r), 1, 1, fill=False, edgecolor="lightgrey", linewidth=0.5)
            ax.add_patch(rect)

    # blocked
    for (r, c) in blocked:
        rect = Rectangle((c, r), 1, 1, facecolor="black", alpha=0.3)
        ax.add_patch(rect)

    # warehouse
    wr, wc = warehouse
    ax.add_patch(Rectangle((wc, wr), 1, 1, facecolor="gold", alpha=0.8))
    ax.text(wc + 0.5, wr + 0.5, "W", ha="center", va="center", fontsize=10)

    # deliveries with demand
    for d in deliveries:
        if d == warehouse:
            continue
        dr, dc = d
        label = str(demands.get(d, "?"))
        ax.add_patch(Rectangle((dc, dr), 1, 1, facecolor="skyblue", alpha=0.7))
        ax.text(dc + 0.5, dr + 0.5, label, ha="center", va="center", fontsize=9)

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue", "tab:orange", "tab:green"])

    sc = None  # for colorbar

    for i, (cells, energies_kwh) in enumerate(trip_infos):
        if not cells:
            continue
        xs = [c + 0.5 for (r, c) in cells]
        ys = [r + 0.5 for (r, c) in cells]
        color = colors[i % len(colors)]

        initial = energies_kwh[0] if energies_kwh and energies_kwh[0] > 0 else 1.0
        energies_pct = [max(0.0, 100.0 * E / initial) for E in energies_kwh]

        # path line
        ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.5, label=f"Trip {i+1}")

        # scatter colored by percentage
        sc = ax.scatter(xs, ys, c=energies_pct, cmap="viridis", s=40, edgecolor="k", vmin=0.0, vmax=100.0)

        # arrows to show direction
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

    # colorbar for energy percentage
    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Remaining motion energy (%)")

    plt.tight_layout()
    plt.show()


# -----------------------
# NEW: interpolate one trip path (continuous x(t), y(t))
# -----------------------
def plot_interpolated_trip_continuous(trip_info: Tuple[Path, List[float]], cruise_speed_kmh: float):
    """
    Take one trip's discrete grid cells and:
      - infer time stamps from step distance and cruise speed
      - build interp1d(x(t)), interp1d(y(t)) for continuous position
      - plot:
          * map view: discrete vs interpolated path
          * time view: x(t) and y(t)
    This demonstrates interpolation / reconstruction on the grid path.
    """
    cells, energies = trip_info
    if len(cells) < 2:
        print("Not enough points in trip to interpolate.")
        return

    v = cruise_speed_kmh / 3.6  # m/s
    # grid cell center coordinates
    xs = np.array([c + 0.5 for (r, c) in cells], dtype=float)
    ys = np.array([r + 0.5 for (r, c) in cells], dtype=float)

    # Build time stamps based on step distance / speed
    t_nodes = [0.0]
    for (r1, c1), (r2, c2) in zip(cells[:-1], cells[1:]):
        dist_m = step_cost((r1, c1), (r2, c2))  # interpreted as metres
        dt = dist_m / v
        t_nodes.append(t_nodes[-1] + dt)
    t_nodes = np.array(t_nodes)

    # Interpolate x(t), y(t)
    fx = interp1d(t_nodes, xs, kind="linear")
    fy = interp1d(t_nodes, ys, kind="linear")

    t_dense = np.linspace(t_nodes[0], t_nodes[-1], 200)
    x_dense = fx(t_dense)
    y_dense = fy(t_dense)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: map view (discrete vs interpolated)
    ax0 = axes[0]
    ax0.set_title("Trip path (discrete vs interpolated)")
    ax0.plot(xs, ys, "o-", label="Grid cells")
    ax0.plot(x_dense, y_dense, "--", label="Interpolated path")
    ax0.set_aspect("equal", adjustable="box")
    ax0.invert_yaxis()
    ax0.set_xlabel("x (grid units)")
    ax0.set_ylabel("y (grid units)")
    ax0.legend()

    # Right: x(t) and y(t)
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
if __name__ == "__main__":
    rows, cols = 10, 10
    num_deliveries = 7
    carry_capacity = 10       # kg
    battery_capacity = 5.0    # kWh
    cruise_speed = 40.0       # km/h
    seed = None

    blocked: Set[Coord] = set()
    random.seed(seed)

    warehouse = (rows // 2, cols // 2)
    deliveries: List[Coord] = []
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

    # ODE-based vertical energy
    print_vertical_energy_table()

    # ROOT FINDING on power balance: where induced & drag power are equal
    print_crossover_speeds()

    # ODE + INTERPOLATION + ROOT FINDING on altitude(t)
    plot_takeoff_profile()

    # Range & endurance vs payload
    plot_range_and_endurance_vs_payload(battery_capacity, cruise_speed)

    # Routing / energy-aware trips
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

    # Grid visualization (energy-coloured paths)
    if trip_infos:
        plot_grid_and_paths(rows, cols, warehouse, deliveries, demands, blocked, trip_infos)

        # Interpolated continuous path for the first trip
        plot_interpolated_trip_continuous(trip_infos[0], cruise_speed)
