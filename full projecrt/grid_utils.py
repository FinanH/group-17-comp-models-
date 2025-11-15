# grid_utils.py
import math
import heapq
from typing import Tuple, List, Dict, Optional, Set

Coord = Tuple[int, int]
Path = List[Coord]


def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols


def neighbors(r: int, c: int, rows: int, cols: int):
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
    base = 1.0  # km per cardinal step
    dr = v[0] - u[0]
    dc = v[1] - u[1]
    if dr != 0 and dc != 0:
        return base * math.sqrt(2.0)
    return base


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


def precompute_pairs(rows: int, cols: int, points, blocked: Set[Coord]):
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
                    dist_km[(s, t)] = d[t]
                    pt = reconstruct_path(prev, s, t)
                    if pt is None:
                        dist_km[(s, t)] = float("inf")
                        path_map[(s, t)] = []
                    else:
                        path_map[(s, t)] = pt

    return dist_km, path_map

