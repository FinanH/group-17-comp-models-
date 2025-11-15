# main.py
import random
import numpy as np

from grid_utils import Coord
from config import BATTERY_CAPACITY_FROM_CELLS_KWH, BATTERY_CAPACITY_KWH_GLOBAL
from dynamics import print_vertical_energy_table, print_crossover_speeds
from routing import run_all_trips
from plotting_helpers import (
    print_grid,
    plot_takeoff_profile,
    plot_grid_and_paths,
    plot_interpolated_trip_continuous,
    plot_payload_vs_total_delivered,
)

if __name__ == "__main__":
    rows, cols = 10, 10
    num_deliveries = 7
    carry_capacity = 10

    battery_capacity = BATTERY_CAPACITY_FROM_CELLS_KWH
    cruise_speed = 40.0

    trip_distance_km = 2.0
    plot_payload_vs_total_delivered(
        trip_distance_km=trip_distance_km,
        cruise_speed_kmh=cruise_speed,
        battery_capacity_kwh=battery_capacity,
        num_points=21,
    )

    seed = None
    random.seed(seed)

    BATTERY_CAPACITY_KWH_GLOBAL = battery_capacity

    warehouse: Coord = (random.randint(0, rows - 1), random.randint(0, cols - 1))

    blocked = set()
    while len(blocked) < 2:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        cell = (r, c)
        if cell == warehouse:
            continue
        blocked.add(cell)

    deliveries = []
    demands = {}
    while len(deliveries) < num_deliveries:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        cell = (r, c)
        if cell == warehouse or cell in blocked or cell in deliveries:
            continue
        deliveries.append(cell)
        demands[cell] = random.randint(1, 3)

    print_grid(rows, cols, warehouse, deliveries, demands, blocked)
    print_vertical_energy_table()
    print_crossover_speeds()
    plot_takeoff_profile()

    trip_infos = run_all_trips(
        rows=rows,
        cols=cols,
        warehouse=warehouse,
        deliveries=deliveries,
        demands=demands,
        blocked=blocked,
        carry_capacity=carry_capacity,
        battery_capacity_kwh=battery_capacity,
        cruise_speed_kmh=cruise_speed,
    )

    if trip_infos:
        plot_grid_and_paths(rows, cols, warehouse, deliveries, demands, blocked, trip_infos)
        plot_interpolated_trip_continuous(trip_infos[0], cruise_speed)
