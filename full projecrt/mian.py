from interpolation_regretion import linear_interp1d, _interp_scalar, incremental_bisection_root, linear_regression, exponential_fit
from ODEs_called import 


def move_energy_kwh(distance_km: float, payload_kg: float, cruise_speed_kmh: float) -> float:
    """
    Cruise energy for a horizontal grid leg.

    Uses the move_energy ODE:
      state y = [z, vz, x, vx, E]
      dE/dt = -P_elec from your ODE.

    We:
      - Integrate until x reaches distance_km * 1000 m,
      - Take the difference in E to get energy in Joules,
      - Convert to kWh,
      - Scale linearly with total mass for different payloads.
    """
    if distance_km <= 0.0 or cruise_speed_kmh <= 0.0:
        return 0.0

    distance_m = distance_km * 1000.0
    v = cruise_speed_kmh / 3.6  # m/s

    # We will temporarily change some globals used inside move_energy
    # so that the leg is flown at constant cruise tilt and level altitude.
    global x_target, x_cruise_start, z_target

    old_x_target = x_target
    old_x_cruise_start = x_cruise_start
    old_z_target = z_target

    try:
        # Level flight around current cruise altitude
        z_target = alt_target

        # Force "always cruise tilt": x < huge -> theta = theta_cruise
        x_target = 1e9
        x_cruise_start = 1e9

        # Initial state: at cruise altitude, no vertical motion, x=0, vx = cruise speed, full usable energy
        y0 = [alt_target, 0.0, 0.0, v, E_avail]

        def event_reach_distance(t, y):
            # Stop integration when x reaches the leg length
            return y[2] - distance_m

        event_reach_distance.terminal = True
        event_reach_distance.direction = 1

        # Rough upper bound on time: distance / speed * 5 as safety factor
        t_end = distance_m / max(v, 0.1) * 5.0

        sol = solve_ivp(
            move_energy,
            (0.0, t_end),
            y0,
            events=event_reach_distance,
            rtol=1e-6,
            atol=1e-8,
            max_step=0.5
        )

        # If for some reason the event didn't trigger, fall back to the algebraic model
        if sol.t_events[0].size == 0:
            P_W = power_model(payload_kg, v, params)  # Watts
            P_kW = P_W / 1000.0
            time_h = distance_km / cruise_speed_kmh
            return P_kW * time_h

        # E is the 5th state (index 4); E_avail is the initial energy (Joules)
        E_end = sol.y[4, -1]
        E_used_J = E_avail - E_end
        E_base_kwh = max(E_used_J, 0.0) / 3.6e6  # J -> kWh

    finally:
        # Restore globals so demos & other code still behave as before
        x_target = old_x_target
        x_cruise_start = old_x_cruise_start
        z_target = old_z_target

    # Scale with total mass, same idea as takeoff/landing
    total_mass = m_frame + m_battery + payload_kg
    mass_ratio = total_mass / m_tot
    return E_base_kwh * mass_ratio


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

Coord = Tuple[int, int]
Path = List[Coord]

