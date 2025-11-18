# plotting.py

from typing import Tuple, List, Dict, Optional, Set

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

import ode
from ode import (
    alt_target,
    E_avail,
    rho,
    A_disk,
    m_tot,
    landing_energy_kwh_for,
    takeoff_energy_kwh_for,
    move_energy_kwh,
    power_model,
    params,
)

from interpolation import linear_interp1d
from root_finding import incremental_bisection_root, exponential_fit


Coord = Tuple[int, int]
Path = List[Coord]


# step_cost 
def step_cost(u: Coord, v: Coord) -> float:
    """
    Distance between two neighboring cells in KM.
    Cardinal step = 0.1 km (100 m)
    Diagonal step = 0.1 * sqrt(2) km
    """
    base = 1  # km per cardinal step
    dr = v[0] - u[0]
    dc = v[1] - u[1]
    if dr != 0 and dc != 0:
        return base * np.sqrt(2.0)
    return base


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
    global m_tot, rho, A_disk, E_avail
    from ode import takeoff_dynamics  

    g = 9.8
    M_TAKEOFF = m_tot
    VH_TAKEOFF = np.sqrt((M_TAKEOFF * g) / (2 * rho * A_disk))

    y0 = [0.0, 0.0, E_avail]
    t_span = (0.0, 60.0)

    def event_alt_reached(t, y):
        return y[0] - alt_target

    event_alt_reached.terminal = True
    event_alt_reached.direction = 1

    from scipy.integrate import solve_ivp

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
    target_alt = 30.0

    def gfun(tval):
        return float(f_alt(tval) - target_alt)

    t_min, t_max = float(t_dense[0]), float(t_dense[-1])
    root_t = incremental_bisection_root(
        gfun,
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
    payloads = np.linspace(0.0, 6.0, 11)
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


def plot_payload_vs_total_delivered(
    trip_distance_km: float,
    cruise_speed_kmh: float,
    battery_capacity_kwh: float = ode.BATTERY_CAPACITY_FROM_CELLS_KWH,
    structural_limit_kg: float = 30.0,   # <- big default, change as needed
    num_points: int = 61
):
    """
    Payload vs total delivered mass (multiple identical trips per battery).
    Goes up to the structural payload limit instead of stopping at 10 kg.
    """

    payloads = np.linspace(0.0, structural_limit_kg, num_points)
    total_delivered = []

    for W in payloads:
        if W <= 0:
            total_delivered.append(0.0)
            continue

        # Energy per trip
        E_takeoff = takeoff_energy_kwh_for(W)
        E_cruise = move_energy_kwh(trip_distance_km, W, cruise_speed_kmh)
        E_landing = landing_energy_kwh_for(W)

        E_trip = E_takeoff + E_cruise + E_landing

        if E_trip >= battery_capacity_kwh:
            total_delivered.append(0.0)
            continue

        n_trips = int(battery_capacity_kwh // E_trip)
        total_delivered.append(n_trips * W)

    plt.figure()
    plt.plot(payloads, total_delivered, marker="o")
    plt.xlabel("Payload per trip (kg)")
    plt.ylabel("Total delivered per battery (kg)")
    plt.title(f"Total delivered vs payload ({trip_distance_km:.1f} km trips)")
    plt.grid(True)
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
    capacity = ode.BATTERY_CAPACITY_KWH_GLOBAL or 1.0

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
