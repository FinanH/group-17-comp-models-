import random
from collections import deque
from typing import Tuple, List, Dict, Optional, Set
import math

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
        #setting the no fly box width and height
        attempts += 1
        w = 1
        h = 1
        #setting the no cly box location 
        r0 = random.randint(0, rows - h) 
        c0 = random.randint(0, cols - w)
        # setting where the values of the grid are blocked and saving it for the creation of the marix
        rect = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
        #setting rect to a subset of blocked and setting it in binary to save opperating space
        if not rect.issubset(blocked):
            blocked |= rect
            count -= 1
    return blocked

# -----------------------
# Grid & utilities
# -----------------------
def random_warehouse(rows: int, cols: int, blocked: Set[Coord], seed: Optional[int] = None) -> Coord:
    # if there is a seed then useing that rather than random values
    if seed is not None:
        random.seed(seed + 2002)
    
    # making a list of avable options
    candidates = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in blocked]
    #if there is no space then print an error
    if not candidates:
        raise ValueError("No space for warehouse (everything blocked).")
    #selecting the location of w from the options
    return random.choice(candidates)


def sample_delivery_points(rows: int, cols: int, k: int, avoid: Coord, blocked: Set[Coord],
                           seed: Optional[int] = None) -> List[Coord]:
    #setting a seed 
    if seed is not None:
        random.seed(seed + 3003)
    # setting the 
    cells = [(r, c) for r in range(rows) for c in range(cols)
             if (r, c) != avoid and (r, c) not in blocked]
    #randomising cells
    random.shuffle(cells)
    #returns the number of deleveris of the cells
    return cells[:k]

def assign_demands(deliveries: List[Coord], low: int = 1, high: int = 3, seed: Optional[int] = None) -> Dict[Coord, int]:
    if seed is not None:
        random.seed(seed + 4004)
    #setting the amount of weight that each delivary has 
    return {d: random.randint(low, high) for d in deliveries}

# -----------------------
# BFS shortest path (steps) with obstacles
# -----------------------
def bfs_shortest_path(rows: int, cols: int, start: Coord, end: Coord, blocked: Set[Coord]) -> Tuple[int, Path]:
    # making sure that you can and need to move
    if start == end:
        return 0, [start]
    if start in blocked or end in blocked:
        return float("inf"), []
    
    sr, sc = start
    er, ec = end
    q = deque([(sr, sc)])
    parent: Dict[Coord, Coord] = {}
    seen = [[False]*cols for _ in range(rows)]
    seen[sr][sc] = True
    # setting the avalable move actions
    dirs = [(1,0), (-1,0), (0,1), (0,-1), #horezontals
            (1, 1), (1, -1), (-1, 1), (-1, -1)]  # diagonals
    
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr][nc] and (nr, nc) not in blocked:
                seen[nr][nc] = True
                parent[(nr, nc)] = (r, c)
                if (nr, nc) == (er, ec):
                    path = [(er, ec)]
                    cur = (er, ec)
                    while cur != (sr, sc):
                        cur = parent[cur]
                        path.append(cur)
                    path.reverse()
                    return len(path)-1, path
                q.append((nr, nc))
    return float("inf"), []

# -----------------------
# Precompute distances/paths between key points
# -----------------------
def precompute_pairs(rows: int, cols: int, points: List[Coord], blocked: Set[Coord]):
    dist: Dict[Tuple[Coord, Coord], int] = {}
    path: Dict[Tuple[Coord, Coord], Path] = {}
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if i == j:
                dist[(a, b)] = 0
                path[(a, b)] = [a]
            elif (a, b) not in dist:
                steps, p = bfs_shortest_path(rows, cols, a, b, blocked)
                dist[(a, b)] = steps
                path[(a, b)] = p
                dist[(b, a)] = steps
                path[(b, a)] = list(reversed(p))
    return dist, path

# -----------------------
# Energy helpers
# -----------------------
def move_energy(steps: int, load: float) -> float:
    rate = load if load > 0 else 0.5
    return steps * rate

# -----------------------
# Trip-by-trip planner (prints as it goes)
# -----------------------
def run_all_trips(
    rows: int,
    cols: int,
    warehouse: Coord,
    deliveries: List[Coord],
    demands: Dict[Coord, int],
    blocked: Set[Coord],
    carry_capacity: int = 10,
    battery_capacity: float = 100.0
):
    points = [warehouse] + deliveries
    dist, path = precompute_pairs(rows, cols, points, blocked)

    # Filter unreachable deliveries beforehand
    remaining = []
    impossible = []
    # 
    for d in deliveries:
        if demands[d] > carry_capacity:
            impossible.append(d)
        elif dist[(warehouse, d)] == float("inf") or dist[(d, warehouse)] == float("inf"):
            impossible.append(d)
        else:
            remaining.append(d)

    trip_idx = 0
    grand_total_energy = 0.0
    while remaining:
        trip_idx += 1
        energy = battery_capacity
        carried = min(carry_capacity, sum(demands[d] for d in remaining))
        current = warehouse
        trip_path: Path = [warehouse]
        trip_legs = []
        served_this_trip = []

        # Keep serving until no next delivery fits with safe return
        while True:
            candidates = []
            for d in remaining:
                dmd = demands[d]
                if dmd > carried:
                    continue
                steps_to = dist[(current, d)]
                if steps_to == float("inf"):
                    continue
                go_e = move_energy(steps_to, carried if carried > 0 else 0.5)
                new_load = carried - dmd
                steps_back = dist[(d, warehouse)]
                if steps_back == float("inf"):
                    continue
                back_e = move_energy(steps_back, new_load if new_load > 0 else 0.5)
                need = go_e + back_e
                if need <= energy:
                    candidates.append((steps_to, d, go_e, back_e, new_load))
            if not candidates:
                # No further deliveries this trip; return to W if not there already
                if current != warehouse:
                    steps_home = dist[(current, warehouse)]
                    if steps_home == float("inf"):
                        break
                    back_e = move_energy(steps_home, carried if carried > 0 else 0.5)
                    if back_e > energy:
                        break
                    p = path[(current, warehouse)]
                    trip_path += (p[1:] if p and p[0] == current else p)
                    trip_legs.append({
                        'type': 'move', 'from': current, 'to': warehouse,
                        'steps': steps_home, 'load_before': carried if carried > 0 else 0,
                        'energy': back_e
                    })
                    energy -= back_e
                    grand_total_energy += back_e
                    current = warehouse
                break

            # Pick nearest-by-steps
            candidates.sort(key=lambda x: x[0])
            steps_to, target, go_e, back_e, new_load = candidates[0]

            # Move current -> target
            p = path[(current, target)]
            trip_path += (p[1:] if p and p[0] == current else p)
            trip_legs.append({
                'type': 'move', 'from': current, 'to': target,
                'steps': steps_to, 'load_before': carried, 'energy': go_e
            })
            energy -= go_e
            grand_total_energy += go_e
            current = target

            # Drop
            dmd = demands[target]
            carried -= dmd
            served_this_trip.append((target, dmd))
            trip_legs.append({'type': 'drop', 'at': target, 'demand': dmd, 'load_after': carried})
            remaining.remove(target)

            # If empty, head home immediately (if not already there)
            if carried == 0 and current != warehouse:
                steps_home = dist[(current, warehouse)]
                if steps_home == float("inf"):
                    break
                back_e2 = move_energy(steps_home, 0.5)
                if back_e2 > energy:
                    break
                p = path[(current, warehouse)]
                trip_path += (p[1:] if p and p[0] == current else p)
                trip_legs.append({
                    'type': 'move', 'from': current, 'to': warehouse,
                    'steps': steps_home, 'load_before': 0, 'energy': back_e2
                })
                energy -= back_e2
                grand_total_energy += back_e2
                current = warehouse
                break  # end this trip (we’ll reload next loop)

        # Print this trip immediately
        print(f"\n=== Trip {trip_idx} ===")
        if served_this_trip:
            print("Delivered:", ", ".join(f"{pt} (x{dmd})" for pt, dmd in served_this_trip))
        else:
            print("No deliveries completed on this trip.")
        print("Legs:")
        for i, leg in enumerate(trip_legs, 1):
            if leg['type'] == 'move':
                print(f"  {i:02d}. MOVE {leg['from']} -> {leg['to']}  "
                      f"steps={leg['steps']}  load_before={leg['load_before']}  "
                      f"energy={leg['energy']:.2f}")
            else:
                print(f"  {i:02d}. DROP at {leg['at']}  demand={leg['demand']}  "
                      f"load_after={leg['load_after']}")
        print(f"Trip {trip_idx} ended at: {current}")
        print("Remaining deliveries:", remaining if remaining else "None")

        # If we’re stuck at warehouse and still can’t serve any remaining with fresh energy, stop
        if remaining:
            # Fresh battery/load feasibility check for at least one remaining
            can_start = False
            for d in remaining:
                dmd = demands[d]
                if dmd > carry_capacity:
                    continue
                steps_to = dist[(warehouse, d)]
                steps_back = dist[(d, warehouse)]
                if steps_to == float("inf") or steps_back == float("inf"):
                    continue
                go_e = move_energy(steps_to, min(carry_capacity, sum(demands[x] for x in remaining)))
                back_e = move_energy(steps_back, max(min(carry_capacity, sum(demands[x] for x in remaining)) - dmd, 0) or 0.5)
                if go_e + back_e <= battery_capacity:
                    can_start = True
                    break
            if not can_start:
                # Remaining are impossible
                print("\nSome deliveries are impossible with current obstacles/capacity:", remaining)
                break

    print("\n=== All trips complete (or as many as feasible) ===")
    print(f"Grand total movement energy: {grand_total_energy:.2f}")

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
    print("\nLegend: W=warehouse | 1–3=delivery demand | X=no-fly | .=empty\n")

# -----------------------
# Demo / run
# -----------------------
if __name__ == "__main__":
    rows, cols = 10, 10
    num_deliveries = 7
    carry_capacity = 10
    battery_capacity = 75.0
    seed = None  # change/None for different random worlds

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
        battery_capacity=battery_capacity
    )
    