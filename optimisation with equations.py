import random
import math
import heapq
from typing import Tuple, List, Dict, Optional, Set

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
# Trip-by-trip planner
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

    trip_idx = 0
    grand_total_energy = 0.0
    while remaining:
        trip_idx += 1
        energy = battery_capacity_kwh
        carried = min(carry_capacity, sum(demands[d] for d in remaining))
        current = warehouse
        trip_path: Path = [warehouse]
        trip_legs = []
        served_this_trip = []

        while True:
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
                if need <= energy:
                    candidates.append((dist_to_km, d, go_e, back_e, new_load))
            if not candidates:
                if current != warehouse:
                    dist_home_km = dist_km[(current, warehouse)]
                    back_e = move_energy_kwh(dist_home_km, max(carried, 0.0), cruise_speed_kmh)
                    if back_e > energy:
                        break
                    p = path[(current, warehouse)]
                    trip_path += (p[1:] if p and p[0] == current else p)
                    trip_legs.append({
                        'type': 'move', 'from': current, 'to': warehouse,
                        'distance_km': dist_home_km, 'load_before_kg': max(carried, 0.0), 'energy_kwh': back_e
                    })
                    energy -= back_e
                    grand_total_energy += back_e
                    current = warehouse
                break

            candidates.sort(key=lambda x: x[0])
            dist_to_km, target, go_e, back_e, new_load = candidates[0]
            p = path[(current, target)]
            trip_path += (p[1:] if p and p[0] == current else p)
            trip_legs.append({
                'type': 'move', 'from': current, 'to': target,
                'distance_km': dist_to_km, 'load_before_kg': carried, 'energy_kwh': go_e
            })
            energy -= go_e
            grand_total_energy += go_e
            current = target

            dmd = demands[target]
            carried -= dmd
            served_this_trip.append((target, dmd))
            trip_legs.append({'type': 'drop', 'at': target, 'demand': dmd, 'load_after': carried})
            remaining.remove(target)

            if carried == 0 and current != warehouse:
                dist_home_km = dist_km[(current, warehouse)]
                back_e2 = move_energy_kwh(dist_home_km, 0.0, cruise_speed_kmh)
                if back_e2 > energy:
                    break
                p = path[(current, warehouse)]
                trip_path += (p[1:] if p and p[0] == current else p)
                trip_legs.append({
                    'type': 'move', 'from': current, 'to': warehouse,
                    'distance_km': dist_home_km, 'load_before_kg': 0.0, 'energy_kwh': back_e2
                })
                energy -= back_e2
                grand_total_energy += back_e2
                current = warehouse
                break

        print(f"\n=== Trip {trip_idx} ===")
        if served_this_trip:
            print("Delivered:", ", ".join(f"{pt} (x{dmd} kg)" for pt, dmd in served_this_trip))
        else:
            print("No deliveries completed on this trip.")
        print("Legs:")
        for i, leg in enumerate(trip_legs, 1):
            if leg['type'] == 'move':
                print(f"  {i:02d}. MOVE {leg['from']} -> {leg['to']}  "
                      f"dist_km={leg['distance_km']:.3f}  load_before_kg={leg['load_before_kg']}  "
                      f"energy_kwh={leg['energy_kwh']:.3f}")
            else:
                print(f"  {i:02d}. DROP at {leg['at']}  demand_kg={leg['demand']}  load_after_kg={leg['load_after']}")
        print(f"Trip {trip_idx} ended at: {current}")
        print("Remaining deliveries:", remaining if remaining else "None")

        if remaining:
            can_start = False
            start_load = min(carry_capacity, sum(demands[x] for x in remaining))
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
                if go_e + back_e <= battery_capacity_kwh:
                    can_start = True
                    break
            if not can_start:
                print("\nSome deliveries are impossible with current obstacles/capacity:", remaining)
                break

    print("\n=== All trips complete (or as many as feasible) ===")
    print(f"Grand total movement energy (kWh): {grand_total_energy:.3f}")

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
    rows, cols = 10, 10
    num_deliveries = 7
    carry_capacity = 10       # kg
    battery_capacity = 5.0    # kWh
    cruise_speed = 40.0       # km/h
    seed = None

    blocked = make_no_fly_zones(rows, cols, count=2, seed=seed, max_w=3, max_h=3)
    warehouse = random_warehouse(rows, cols, blocked, seed=seed)
    deliveries = sample_delivery_points(rows, cols, num_deliveries, avoid=warehouse, blocked=blocked, seed=seed)
    demands = assign_demands(deliveries, low=1, high=3, seed=seed)

    print_grid(rows, cols, warehouse, deliveries, demands, blocked)

    print("Warehouse:", warehouse)
    print("Deliveries & demands:", demands)
    print("\nStarting trips…")

    run_all_trips(
        rows, cols,
        warehouse=warehouse,
        deliveries=deliveries,
        demands=demands,
        blocked=blocked,
        carry_capacity=carry_capacity,
        battery_capacity_kwh=battery_capacity,
        cruise_speed_kmh=cruise_speed
    )
