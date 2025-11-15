# main.py

import random
import math
import heapq
from typing import Tuple, List, Dict, Optional, Set

import numpy as np

import ode
from ode import (
    BATTERY_CAPACITY_FROM_CELLS_KWH,
    print_vertical_energy_table,
    takeoff_energy_kwh_for,
    landing_energy_kwh_for,
    move_energy_kwh,
)
from root_finding import print_crossover_speeds
from plotting import (
    plot_takeoff_profile,
    plot_range_and_endurance_vs_payload,
    plot_payload_vs_total_delivered,
    plot_grid_and_paths,
    plot_interpolated_trip_continuous,
    Coord,
    Path,
    step_cost,
)


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
# Demo / run
# -----------------------

if __name__ == "__main__":
    rows, cols = 10, 10
    num_deliveries = 7
    carry_capacity = 10       # kg (total payload capacity)

    # Use the actual pack we defined at the top
    battery_capacity = BATTERY_CAPACITY_FROM_CELLS_KWH  # ≈ 0.0799 kWh
    cruise_speed = 40.0       # km/h
    
    # -----------------------
    # Payload vs total delivered mass plot
    # -----------------------
    # Example: 2 km one-way trip distance
    trip_distance_km = 2.0
    
    plot_payload_vs_total_delivered(
        trip_distance_km=trip_distance_km,
        cruise_speed_kmh=cruise_speed,
        battery_capacity_kwh=battery_capacity,  # use your drone's payload capacity
        num_points=21
    )

    # Set a seed if you want reproducible randomness, or None for different each run
    seed = None
    random.seed(seed)

  
    ode.BATTERY_CAPACITY_KWH_GLOBAL = battery_capacity

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
