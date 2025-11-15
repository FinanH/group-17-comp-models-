# routing.py
from typing import List, Dict, Set, Tuple

from config import BATTERY_CAPACITY_FROM_CELLS_KWH
from dynamics import move_energy_kwh, landing_energy_kwh_for, takeoff_energy_kwh_for
from grid_utils import Coord, Path, precompute_pairs, step_cost


def run_all_trips(
    rows: int,
    cols: int,
    warehouse: Coord,
    deliveries: List[Coord],
    demands: Dict[Coord, int],
    blocked: Set[Coord],
    carry_capacity: int = 10,
    battery_capacity_kwh: float = BATTERY_CAPACITY_FROM_CELLS_KWH,
    cruise_speed_kmh: float = 40.0,
) -> List[Tuple[Path, List[float]]]:

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
    total_energy_used = 0.0
    all_trip_infos: List[Tuple[Path, List[float]]] = []

    while remaining:
        trip_idx += 1
        carried = min(carry_capacity, sum(demands[d] for d in remaining))

        battery_soc = battery_capacity_kwh
        trip_energy_used = 0.0

        takeoff_E = takeoff_energy_kwh_for(carried)
        min_landing_E = landing_energy_kwh_for(0.0)

        if takeoff_E + min_landing_E >= battery_capacity_kwh:
            print(f"\nTrip {trip_idx}: cannot even take off and land with current battery capacity for carried={carried} kg.")
            break

        battery_soc -= takeoff_E
        trip_energy_used += takeoff_E
        total_energy_used += takeoff_E

        current = warehouse
        trip_cells: Path = [warehouse]
        trip_soc_list: List[float] = [battery_soc]

        trip_legs = []
        served_this_trip = []

        while True:
            landing_E_needed = landing_energy_kwh_for(carried)
            available_for_motion = battery_soc - landing_E_needed

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
                if current != warehouse:
                    dist_home_km = dist_km[(current, warehouse)]
                    back_e = move_energy_kwh(dist_home_km, max(carried, 0.0), cruise_speed_kmh)

                    landing_E_needed = landing_energy_kwh_for(carried)
                    available_for_motion = battery_soc - landing_E_needed
                    if back_e > available_for_motion:
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

            candidates.sort(key=lambda x: x[0])
            dist_to_km, target, go_e, back_e, new_load = candidates[0]

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

        landing_E_final = landing_energy_kwh_for(carried)
        if battery_soc < landing_E_final:
            print(f"Warning: battery_soc < landing_E_final on trip {trip_idx}, clamping at 0.")
            landing_E_final = max(battery_soc, 0.0)
        battery_soc -= landing_E_final
        trip_energy_used += landing_E_final
        total_energy_used += landing_E_final
        trip_legs.append({'type': 'landing', 'energy_kwh': landing_E_final})

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

