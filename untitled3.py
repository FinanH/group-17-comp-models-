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

import matplotlib.pyplot as plt

Coord = Tuple[int, int]
Path = List[Coord]

# -----------------------
# No-fly zones (obstacles)
# -----------------------
def make_no_fly_zones(rows: int, cols: int, count: int = 2, seed: Optional[int] = None,
                      max_w: int = 1, max_h: int = 1) -> Set[Coord]:
    if seed is not None:
        random.seed(seed + 1001)
    blocked: Set[Coord] = set()
    attempts = 0
    while count > 0 and attempts < 200:
        attempts += 1
        w = 1
        h = 1
        r0 = random.randint(0, rows - h)
        c0 = random.randint(0, cols - w)
        rect = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
        if not rect.issubset(blocked):
            blocked |= rect
            count -= 1
    return blocked

# -----------------------
# Grid & utilities
# -----------------------
def random_warehouse(rows: int, cols: int, blocked: Set[Coord], seed: Optional[int] = None) -> Coord:
    if seed is not None:
        random.seed(seed + 2002)
    candidates = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in blocked]
    if not candidates:
        raise ValueError("No space for warehouse (everything blocked).")
    return random.choice(candidates)

def sample_delivery_points(rows: int, cols: int, k: int, avoid: Coord, blocked: Set[Coord],
                           seed: Optional[int] = None) -> List[Coord]:
    if seed is not None:
        random.seed(seed + 3003)
    cells = [(r, c) for r in range(rows) for c in range(cols)
             if (r, c) != avoid and (r, c) not in blocked]
    random.shuffle(cells)
    return cells[:k]

def assign_demands(deliveries: List[Coord], low: int = 1, high: int = 3, seed: Optional[int] = None) -> Dict[Coord, int]:
    if seed is not None:
        random.seed(seed + 4004)
    return {d: random.randint(low, high) for d in deliveries}

# -----------------------
# Dijkstra shortest path (km) with obstacles
# -----------------------
def dijkstra_shortest_path_km(rows: int, cols: int, start: Coord, end: Coord, blocked: Set[Coord]) -> Tuple[float, Path]:
    if start == end:
        return 0.0, [start]
    if start in blocked or end in blocked:
        return float("inf"), []

    DIRS = [
        ( 1,  0, 1.0), (-1,  0, 1.0), (0,  1, 1.0), (0, -1, 1.0),
        ( 1,  1, math.sqrt(2)), ( 1, -1, math.sqrt(2)),
        (-1,  1, math.sqrt(2)), (-1, -1, math.sqrt(2))
    ]

    dist = {start: 0.0}
    parent: Dict[Coord, Coord] = {}
    pq = [(0.0, start)]

    while pq:
        d, (r, c) = heapq.heappop(pq)
        if (r, c) == end:
            path: Path = []
            cur = end
            while cur != start:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            path.reverse()
            return d, path
        if d > dist[(r, c)]:
            continue
        for dr, dc, w in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in blocked:
                nd = d + w
                if nd < dist.get((nr, nc), float("inf")):
                    dist[(nr, nc)] = nd
                    parent[(nr, nc)] = (r, c)
                    heapq.heappush(pq, (nd, (nr, nc)))
    return float("inf"), []

# -----------------------
# Precompute distances/paths between key points
# -----------------------
def precompute_pairs(rows: int, cols: int, points: List[Coord], blocked: Set[Coord]):
    dist_km: Dict[Tuple[Coord, Coord], float] = {}
    path: Dict[Tuple[Coord, Coord], Path] = {}
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if i == j:
                dist_km[(a, b)] = 0.0
                path[(a, b)] = [a]
            elif (a, b) not in dist_km:
                d, p = dijkstra_shortest_path_km(rows, cols, a, b, blocked)
                dist_km[(a, b)] = d
                path[(a, b)] = p
                dist_km[(b, a)] = d
                path[(b, a)] = list(reversed(p))
    return dist_km, path

# -----------------------
# Energy & Power Model (kWh)
# -----------------------
def power_kw(load_kg: float, base_kw: float = 0.3, alpha_kw_per_kg: float = 0.02, quad_kw_per_kg2: float = 0.0) -> float:
    """
    Base hover/flight power plus linear (and optional quadratic) term in payload mass.
    """
    return base_kw + alpha_kw_per_kg * max(load_kg, 0.0) + quad_kw_per_kg2 * max(load_kg, 0.0)**2

def move_energy_kwh(distance_km: float, load_kg: float, speed_kmh: float) -> float:
    if distance_km == float("inf"):
        return float("inf")
    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be > 0")
    return power_kw(load_kg) * (distance_km / speed_kmh)

# -----------------------
# Trip optimizer (enumerate feasible routes within one trip)
# -----------------------
def enumerate_feasible_routes(
    warehouse: Coord,
    candidates: List[Coord],
    demands: Dict[Coord, int],
    dist_km: Dict[Tuple[Coord, Coord], float],
    battery_capacity_kwh: float,
    carry_capacity: int,
    cruise_speed_kmh: float
):
    """
    Enumerate all feasible sequences starting at warehouse and returning to warehouse
    such that cumulative energy (including return leg) never exceeds battery_capacity_kwh
    and total delivered weight in the trip does not exceed carry_capacity.
    Returns a list of dicts with keys: 'order', 'energy_kwh'.
    """
    results = []

    from itertools import permutations, combinations

    feasible_points = [p for p in candidates if demands[p] <= carry_capacity]

    # Generate subsets of feasible_points respecting capacity
    for r in range(1, len(feasible_points) + 1):
        for subset in combinations(feasible_points, r):
            total_weight = sum(demands[p] for p in subset)
            if total_weight > carry_capacity:
                continue
            # Try all orders
            for order in permutations(subset):
                energy_used = 0.0
                current = warehouse
                carried = total_weight
                ok = True

                # Move through each stop
                for stop in order:
                    d = dist_km.get((current, stop), float("inf"))
                    leg_e = move_energy_kwh(d, carried, cruise_speed_kmh)
                    energy_used += leg_e
                    if energy_used > battery_capacity_kwh:
                        ok = False
                        break
                    carried -= demands[stop]
                    current = stop

                if not ok:
                    continue

                # Return to warehouse
                d_back = dist_km.get((current, warehouse), float("inf"))
                return_e = move_energy_kwh(d_back, max(carried, 0.0), cruise_speed_kmh)
                energy_used += return_e
                if energy_used <= battery_capacity_kwh and math.isfinite(energy_used):
                    results.append({
                        "order": list(order),
                        "energy_kwh": energy_used
                    })

    # If nothing feasible, we still consider the empty route (do nothing this trip)
    if not results:
        results.append({"order": [], "energy_kwh": float("inf")})

    # Sort by energy
    results.sort(key=lambda x: x["energy_kwh"])
    return results

def plot_routes_energy(routes, trip_number: int):
    energies = [r["energy_kwh"] for r in routes]
    xs = list(range(1, len(routes) + 1))

    plt.figure()
    plt.scatter(xs, energies, marker='o')
    if energies and math.isfinite(energies[0]):
        # routes are sorted ascending; index 1 is the minimum
        plt.scatter([1], [energies[0]], marker='x', s=100)
        plt.annotate("min", (1, energies[0]))
    plt.xlabel("Candidate route index (sorted by energy)")
    plt.ylabel("Total energy (kWh)")
    plt.title(f"Trip {trip_number}: candidate routes vs energy")
    plt.show()

# -----------------------
# Planner that uses per-trip optimizer + plotting
# -----------------------
def run_optimized_trips_with_plots(
    rows: int,
    cols: int,
    warehouse: Coord,
    deliveries: List[Coord],
    demands: Dict[Coord, int],
    blocked: Set[Coord],
    carry_capacity: int = 10,
    battery_capacity_kwh: float = 5.0,
    cruise_speed_kmh: float = 40.0,
    max_trips: int = 100
):
    points = [warehouse] + deliveries
    dist_km, path = precompute_pairs(rows, cols, points, blocked)

    remaining = deliveries[:]
    trip_idx = 0
    grand_total_energy = 0.0

    while remaining and trip_idx < max_trips:
        trip_idx += 1

        # Enumerate all feasible routes this trip
        routes = enumerate_feasible_routes(
            warehouse=warehouse,
            candidates=remaining,
            demands=demands,
            dist_km=dist_km,
            battery_capacity_kwh=battery_capacity_kwh,
            carry_capacity=carry_capacity,
            cruise_speed_kmh=cruise_speed_kmh
        )

        # Plot candidates and highlight min
        plot_routes_energy(routes, trip_idx)

        # Pick the minimum-energy route
        best = routes[0]
        if not math.isfinite(best["energy_kwh"]) or not best["order"]:
            print("\nNo feasible route for remaining deliveries within battery/capacity constraints:")
            print(remaining)
            break

        selected = best["order"]
        print(f"\n=== Trip {trip_idx} (optimal among {len(routes)} candidates) ===")
        print("Will visit, in order:", selected)
        print(f"Trip energy (kWh): {best['energy_kwh']:.3f}")

        # Execute: remove delivered points, accumulate energy
        grand_total_energy += best["energy_kwh"]
        for p in selected:
            if p in remaining:
                remaining.remove(p)

    print("\n=== All trips complete (or as many as feasible) ===")
    print(f"Grand total movement energy (kWh): {grand_total_energy:.3f}")
    if remaining:
        print("Undelivered:", remaining)
    else:
        print("All deliveries served.")

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
        r, c = d
        grid[r][c] = str(demands[d])
    print("Grid layout:")
    for r in range(rows):
        print(" ".join(f"{grid[r][c]:>2}" for c in range(cols)))
    print("\nLegend: W=warehouse | 1–3=delivery demand (kg) | X=no-fly | .=empty\n")

# -----------------------
# Demo / run
# -----------------------
if __name__ == "__main__":
    # You can tweak these or wire in your own scenario
    rows, cols = 8, 8
    num_deliveries = 5
    carry_capacity = 7        # kg
    battery_capacity = 4.0    # kWh
    cruise_speed = 40.0       # km/h
    seed = 42                 # fixed for reproducibility

    blocked = make_no_fly_zones(rows, cols, count=2, seed=seed, max_w=1, max_h=1)
    warehouse = random_warehouse(rows, cols, blocked, seed=seed)
    deliveries = sample_delivery_points(rows, cols, num_deliveries, avoid=warehouse, blocked=blocked, seed=seed)
    demands = assign_demands(deliveries, low=1, high=3, seed=seed)

    print_grid(rows, cols, warehouse, deliveries, demands, blocked)

    print("Warehouse:", warehouse)
    print("Deliveries & demands:", demands)
    print("\nOptimizing trips…")

    run_optimized_trips_with_plots(
        rows, cols,
        warehouse=warehouse,
        deliveries=deliveries,
        demands=demands,
        blocked=blocked,
        carry_capacity=carry_capacity,
        battery_capacity_kwh=battery_capacity,
        cruise_speed_kmh=cruise_speed
    )
