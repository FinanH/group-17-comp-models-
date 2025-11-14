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
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp




# ----------------------------
# Drone + environment parameters
# ----------------------------
g = 9.81                # gravity (m/s^2)
rho = 1.225             # air density (kg/m^3)
C_dx = 0.8              # Drag coefficient horizontal
C_dz = 0.9              # Drag coefficient vertical
A_front = 1.154         # Effective front area of drone (m^2)
A_top = 0.175674        # Effective top area of drone (m^2)
A_disk = 1.34           # total rotor disk area (m^2)
eta = 0.75              # overall efficiency (motor * prop)
Vb = 133.2              # battery voltage (V)
Cb = 27                 # battery capacity (Ah)
P_av = 24               # avionics power (W)
usable_frac = 0.9
E_avail = usable_frac * Vb * Cb * 3600  # convert Wh → J

m_frame = 5.93
m_payload = 6
m_battery = 3.57
m_tot = m_frame + m_payload + m_battery


# Desired takeoff climb speed target
v_t_target = 2.0          # m/s (steady climb)
alt_target = 30.0       # m target altitude (stop integration here)





# Desired landing  speed target
v_l_target = -0.50          # m/s (steady decline)




# Desired takeoff climb speed target
vx_target = 15.0            # foward cruise (m/s)
x_target = 2000
vz_target = 0
z_target = 0
transition_time = 3        #  tilt time (s)


# ----------------------------
# Compute drag-braking distance for 15 → ~0.5 m/s
# ----------------------------
v_end = 0.5  # near rest
d_brake = (2*m_tot/(rho*C_dx*A_front)) * np.log(vx_target/v_end)
x_coast_start = x_target - d_brake
print(f"Braking distance ≈ {d_brake:.1f} m | Begin coasting at x ≈ {x_coast_start:.1f} m")


# ----------------------------
# ODEs for takeoff phase
# state vector y = [z, vz, E]
# ----------------------------

def drone_dynamics(t, y):
    z, vz, x, vx, E = y
   
    Drag_target = 0.5 * rho * C_dx * A_front * (vx_target)**2  # Drag at target speed
    Drag_z = 0.5 * rho * C_dz * A_top * vz**2                  # vertical drag (N)
    Drag_x = 0.5 * rho * C_dx * A_front * vx**2                # horizontal drag (N)
    
    
    theta_req =  np.arctan(Drag_target / (m_tot * g))  # radians
    theta = theta_req * min(t / transition_time, 1.0)
    
    # --- Control law / thrust command ---
    # Simple proportional controller on vertical speed:
        
    
  
    k_p_x = 2.0        # if vx not euquak to vx target increase acceleration by 1
    k_p_z = 10.0
    
    theta_brake = np.deg2rad(0)
    
    if x >= x_coast_start:
        # level out, no horizontal thrust or speed control
        theta = theta_brake
        u_x = 0
    else:
        # normal forward acceleration control
        u_x = k_p_x * (vx_target - vx)
    
    
    
    Thrust = (m_tot * (g + (k_p_z * (z_target - z)))) / np.cos(theta)    # Thurst (N)
    
    
    # --- Power model ---
    # Induced velocity (momentum theory)
    vi = np.sqrt(Thrust / (2 * rho * A_disk))
    P_ind = Thrust * vi                     # induced power (W)
    P_par = abs(Drag_x * vx)
    P_elec = ((P_ind + P_par) / eta) + P_av      # electrical power 

    # --- Dynamics ---
    dzdt = vz
    dvzdt = (Thrust*np.cos(theta) - (m_tot * g) - Drag_z )/ m_tot
    dxdt = vx
    dvxdt = ((Thrust * np.sin(theta) + u_x) - Drag_x) / m_tot
    dEdt = -P_elec

    return [dzdt, dvzdt, dxdt, dvxdt, dEdt]

def plotting_drone_moving():
    # ----------------------------
    # Integration setup
    # ----------------------------
    y0 = [0, 0, 0, 0, E_avail]                    # initial altitude, vertical speed, energy
    t_span = (0, 1000)                              # simulate up to 50 s (should reach speed)
    t_eval = np.linspace(t_span[0], t_span[1], 3000)
    
    sol = solve_ivp(drone_dynamics, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-8)
    
    # ----------------------------
    # Extract results
    # ----------------------------
    z = sol.y[0]
    vz = sol.y[1]
    x = sol.y[2]
    vx = sol.y[3]
    E = sol.y[4]
    t = sol.t
    P_loss = (E_avail - E) / 3600  # convert J to Wh
    
    # Stop when target distance reached
    idx_arr = np.where(x >= x_target)[0]
    if len(idx_arr) > 0:
        idx = idx_arr[0]
        z = z[:idx]
        vz = vz[:idx]
        x = x[:idx]
        vx = vx[:idx]
        E = E[:idx]
        t = t[:idx]
        P_loss = P_loss[:idx]
    else:
        print("Warning: x never reached x_target during simulation.")
    
    # ----------------------------
    # Plot results
    # ----------------------------
    plt.figure(figsize=(10,6))
    
    plt.subplot(3,1,1)
    plt.plot(t, x, label='Displacement x(t)')
    plt.ylabel('Displacement (m)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(t, vx, label='Horizontal Speed')
    plt.axhline(vx_target, color='r', linestyle='--', label='Target speed')
    plt.ylabel('v_x (m/s)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(t, P_loss, label='Energy used')
    plt.ylabel('Energy used (Wh)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Reached velocity: {vx[-1]:.2f} m in {t[-1]:.2f} s")
    print(f"Traveled distance: {x[-1]:.2f} m in {t[-1]:.2f} s")
    print(f"Altitude gained: {z[-1]:.2f} m in {t[-1]:.2f} s")
    print(f"Energy used: {(E_avail - E[-1])/3600:.2f} Wh ({100*(E_avail-E[-1])/E_avail:.2f} % of battery)")

# ----------------------------
# ODEs for takeoff phase
# state vector y = [z, vz, E]
# ----------------------------

def takeoff_dynamics(t, y):
    z, vz, E = y

    # --- Control law / thrust command ---
    # Simple proportional controller on vertical speed:
    k_p = 6.0         # if vz not euqual to v target increase acceleration by 5
                            
    Thurst = m_tot * (g + k_p * (v_t_target - vz))  # Thurst (N)
    Drag_z = 0.5*rho*C_dz*A_top*vz* abs(vz)        # vertical drag (N)
   
    # --- Power model ---
   
    # Induced velocity (momentum theory)
    vi = np.sqrt(Thurst / (2 * rho * A_disk))
  
    P_ind = Thurst * vi                      # induced power (W)
    P_elec = (P_ind / eta) + P_av            # electrical power 

    # --- Dynamics ---
    dzdt = vz
    dvzdt = (Thurst - (m_tot * g) - Drag_z )/ m_tot
    dEdt = -P_elec

    return [dzdt, dvzdt, dEdt]

# ----------------------------
# Integration setup
# ----------------------------
def plotting_take_off():
    y0 = [0.0, 0.0, E_avail]                      # initial altitude, vertical speed, energy
    t_span = (0, 30)                              # simulate up to 30 s (should reach ~10 m)
    t_eval = np.linspace(t_span[0], t_span[1], 300)
    
    sol = solve_ivp(takeoff_dynamics, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-8)
    
    # ----------------------------
    # Extract results
    # ----------------------------
    z = sol.y[0]
    vz = sol.y[1]
    E = sol.y[2]
    t = sol.t
    P_loss = (E_avail - E) / 3600  # convert J to Wh
    
    # Stop when target altitude reached
    idx = np.where(z >= alt_target)[0][0]
    z = z[:idx]
    vz = vz[:idx]
    E = E[:idx]
    t = t[:idx]
    P_loss = P_loss[:idx]
    
    # ----------------------------
    # Plot results
    # ----------------------------
    
    plt.figure(figsize=(10,6))
    
    plt.subplot(3,1,1)
    plt.plot(t, z, label='Altitude z(t)')
    plt.ylabel('Altitude (m)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(t, vz, label='Vertical Speed')
    plt.axhline(v_t_target, color='r', linestyle='--', label='Target speed')
    plt.ylabel('v_z (m/s)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(t, P_loss, label='Energy used')
    plt.ylabel('Energy used (Wh)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Reached altitude: {z[-1]:.2f} m in {t[-1]:.2f} s")
    print(f"Energy used: {(E_avail - E[-1])/3600:.2f} Wh ({100*(E_avail-E[-1])/E_avail:.2f} % of battery)")


# ----------------------------
# ODEs for takeoff phase
# state vector y = [z, vz, E]
# ----------------------------
def landing_dynamics(t, y):
    z, vz, E = y

    # --- Control law / thrust command ---
    # Simple proportional controller on vertical speed:
    k_p = 6.0         # if vz not euqual to v target increase acceleration by 5
                            
    Thurst = m_tot * (g + k_p * (v_l_target - vz))  # Thurst (N)
    Drag_z = 0.5*rho*C_dz*A_top*vz* abs(vz)        # vertical drag (N)
   
    # --- Power model ---
   
    # Induced velocity (momentum theory)
    vi = np.sqrt(Thurst / (2 * rho * A_disk))
  
    P_ind = Thurst * vi                      # induced power (W)
    P_elec = (P_ind / eta) + P_av            # electrical power 

    # --- Dynamics ---
    dzdt = vz
    dvzdt = (Thurst - (m_tot * g) - Drag_z )/ m_tot
    dEdt = -P_elec

    return [dzdt, dvzdt, dEdt]

# ----------------------------
# Integration setup
# ----------------------------
def plotting_landing():
    y0 = [0.0, 0.0, E_avail]                      # initial altitude, vertical speed, energy
    t_span = (0, 30)                              # simulate up to 30 s (should reach ~10 m)
    t_eval = np.linspace(t_span[0], t_span[1], 300)
    
    sol = solve_ivp(landing_dynamics, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-8)
    
    # ----------------------------
    # Extract results
    # ----------------------------
    z = sol.y[0]
    vz = sol.y[1]
    E = sol.y[2]
    t = sol.t
    P_loss = (E_avail - E) / 3600  # convert J to Wh
    
    # Stop when target altitude reached
    idx = np.where(z >= alt_target)[0][0]
    z = z[:idx]
    vz = vz[:idx]
    E = E[:idx]
    t = t[:idx]
    P_loss = P_loss[:idx]
    
    # ----------------------------
    # Plot results
    # ----------------------------
    
    plt.figure(figsize=(10,6))
    
    plt.subplot(3,1,1)
    plt.plot(t, z, label='Altitude z(t)')
    plt.ylabel('Altitude (m)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(t, vz, label='Vertical Speed')
    plt.axhline(v_l_target, color='r', linestyle='--', label='Target speed')
    plt.ylabel('v_z (m/s)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3,1,3)
    plt.plot(t, P_loss, label='Energy used')
    plt.ylabel('Energy used (Wh)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Reached altitude: {z[-1]:.2f} m in {t[-1]:.2f} s")
    print(f"Energy used: {(E_avail - E[-1])/3600:.2f} Wh ({100*(E_avail-E[-1])/E_avail:.2f} % of battery)")

Coord = Tuple[int, int]
Path = List[Coord]
# ---------------------------------------
# Parameter setup (values chosen for clarity)
# ---------------------------------------
params = {
    "g": 9.81,
    "m_dry": 2.0,      # drone + battery = 2 kg
    "v_hover": 3.0,
    "k1": 0.5,         # much stronger induced power term
    "k2": 0.02,
    "k3": 5            # much smaller fixed power
}

# Very heavy payload initially, drops sharply
payload_drops = [8.0, 5.0, 3.0]  # start 16 kg → end 0 kg

Vb, Cb, usable_frac = 11.1, 3.0, 0.9
E0 = usable_frac * Vb * Cb * 3600
speed = 8.0


# ---------------------------------------
# Physics-based power model
# ---------------------------------------


def power_model(W, v, params):
    """
    Compute instantaneous electrical power P(W, v) in watts.
    W : current payload mass (kg)
    v : flight speed (m/s)
    params : dictionary of physical constants
    """
    g   = params["g"]
    m0  = params["m_dry"]     # drone dry mass (frame + battery)
    vh  = params["v_hover"]   # induced velocity in hover
    k1  = params["k1"]        # induced power coefficient
    k2  = params["k2"]        # parasitic drag coefficient
    k3  = params["k3"]        # fixed power load (electronics)

    m = m0 + W
    P_ind   = k1 * ((m * g)**1.5) / np.sqrt(v**2 + vh**2)  # induced power
    P_drag  = k2 * v**3                                    # parasitic/drag power
    P_fixed = k3                                            # fixed onboard power
    return P_ind + P_drag + P_fixed


# ---------------------------------------
<<<<<<< HEAD
=======
# Takeoff / landing energy (kWh)
# ---------------------------------------
def compute_takeoff_energy_kwh():
    """
    Integrate takeoff_dynamics until alt_target and return energy used in kWh.
    Uses the same E_avail and takeoff_dynamics as your plotting code,
    but without plotting.
    """
    y0 = [0.0, 0.0, E_avail]          # z, vz, E
    t_span = (0.0, 60.0)              # generous upper bound

    def event_alt_reached(t, y):
        # Stop when z reaches alt_target
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
    E_used_J = E_avail - E_end  # Joules
    return E_used_J / 3.6e6     # J → kWh


# Precompute once for the whole script
TAKEOFF_KWH = compute_takeoff_energy_kwh()
# Assume landing is roughly similar energy cost to takeoff
LANDING_KWH = TAKEOFF_KWH


# ---------------------------------------
>>>>>>> 59e2c9104cddb354c1322830da7a087801ccb849
# Main simulation: energy vs distance
# ---------------------------------------
def simulate_energy_along_path(payload_drops, E0, speed, params):
    """
    Simulate drone battery energy as a function of distance
    while payload mass decreases after each delivery segment.
    This demonstrates that as the drone gets lighter,
    power decreases and the energy curve flattens.
    """
    W = np.sum(payload_drops)   # start with full payload (heaviest)
    E = E0
    s_trace, E_trace, W_trace, P_trace = [0], [E0], [W], []

    # Each flight leg has fixed distance (e.g., 500 m)
    leg_distance = 500
    N_steps = 100
    ds = leg_distance / N_steps
    total_distance = 0

    for leg_index, drop in enumerate(payload_drops):
        for _ in range(N_steps):
            P = power_model(W, speed, params)
            P_trace.append(P)
            dE = -(P / speed) * ds     # ODE: dE/ds = -P/v
            E += dE
            total_distance += ds
            s_trace.append(total_distance)
            E_trace.append(E)
            W_trace.append(W)
            if E <= 0:
                break
        # At the end of each leg, deliver some payload → lighter
        W -= drop
        if W < 0: W = 0

    return np.array(s_trace), np.array(E_trace), np.array(W_trace), np.array(P_trace)

def polt_energy():
    # ---------------------------------------
    # Run simulation
    # ---------------------------------------
    s_trace, E_trace, W_trace, P_trace = simulate_energy_along_path(payload_drops, E0, speed, params)
    
    
    # ---------------------------------------
    # Plot: Battery energy vs distance
    # ---------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(s_trace, E_trace/3600, color='royalblue', label='Remaining energy')
    plt.xlabel('Distance flown (m)')
    plt.ylabel('Remaining battery energy (Wh)')
    plt.title('Energy vs Distance — Heavier → Faster drain → Flattening curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    # ---------------------------------------
    # Plot: Instantaneous power vs distance
    # ---------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(s_trace[:len(P_trace)], P_trace, color='orange', label='Instantaneous power (W)')
    plt.xlabel('Distance flown (m)')
    plt.ylabel('Power (W)')
    plt.title('Power decreases as payload decreases')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    # ---------------------------------------
    # Plot: Payload mass vs distance
    # ---------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(s_trace, W_trace, color='green', label='Payload on board (kg)')
    plt.xlabel('Distance flown (m)')
    plt.ylabel('Payload mass (kg)')
    plt.title('Payload mass decreases after each delivery')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    # ---------------------------------------
    # Print results summary
    # ---------------------------------------
    print("Initial payload: {:.1f} kg".format(sum(payload_drops)))
    print("Final payload: {:.1f} kg".format(W_trace[-1]))
    print("Total energy used: {:.2f} Wh".format((E0 - E_trace[-1]) / 3600))
    print("Power range: {:.1f} W → {:.1f} W".format(max(P_trace), min(P_trace)))

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
<<<<<<< HEAD
=======
    """
    Cruise energy using the physics-based power_model(W, v, params).

    distance_km : path length between two grid points
    load_kg     : payload mass carried on that leg
    speed_kmh   : cruise speed
    """
>>>>>>> 59e2c9104cddb354c1322830da7a087801ccb849
    if distance_km == float("inf"):
        return float("inf")
    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be > 0")
<<<<<<< HEAD
    return power_kw(load_kg) * (distance_km / speed_kmh)
=======

    v = speed_kmh / 3.6          # m/s
    distance_m = distance_km * 1000.0
    time_s = distance_m / v      # s of cruise

    # power_model already includes drone dry mass, payload, and velocity
    P = power_model(load_kg, v, params)   # watts

    E_J = P * time_s
    return E_J / 3.6e6           # kWh

>>>>>>> 59e2c9104cddb354c1322830da7a087801ccb849

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

<<<<<<< HEAD
    trip_idx = 0
    grand_total_energy = 0.0
    while remaining:
        trip_idx += 1
        energy = battery_capacity_kwh
=======
    if impossible:
        print("These deliveries are impossible (capacity or obstacles):", impossible)

    trip_idx = 0
    grand_total_energy = 0.0

    while remaining:
        trip_idx += 1

        # Per-trip takeoff / landing overhead from the ODE model
        takeoff_E = TAKEOFF_KWH
        landing_E = LANDING_KWH

        if takeoff_E + landing_E >= battery_capacity_kwh:
            print(f"\nTrip {trip_idx}: cannot even take off and land with current battery capacity.")
            break

        # Energy available for cruise + return (landing energy reserved logically)
        energy = battery_capacity_kwh - takeoff_E
        grand_total_energy += takeoff_E

>>>>>>> 59e2c9104cddb354c1322830da7a087801ccb849
        carried = min(carry_capacity, sum(demands[d] for d in remaining))
        current = warehouse
        trip_path: Path = [warehouse]
        trip_legs = []
<<<<<<< HEAD
        served_this_trip = []

        while True:
=======

        # Record takeoff as the first leg
        trip_legs.append({'type': 'takeoff', 'energy_kwh': takeoff_E})

        served_this_trip = []

        while True:
            # Always reserve landing_E so we can land safely at the end
            reserved_for_landing = landing_E
            available_for_motion = energy - reserved_for_landing

            if available_for_motion <= 0:
                # No energy left for any more movement; end trip
                break

>>>>>>> 59e2c9104cddb354c1322830da7a087801ccb849
            candidates = []
            for d in remaining:
                dmd = demands[d]
                if dmd > carried:
                    continue
                dist_to_km = dist_km[(current, d)]
                if dist_to_km == float("inf"):
                    continue
<<<<<<< HEAD
                go_e = move_energy_kwh(dist_to_km, carried, cruise_speed_kmh)
                new_load = carried - dmd
=======

                # Outbound leg with current carried load
                go_e = move_energy_kwh(dist_to_km, carried, cruise_speed_kmh)

                # After delivery at d, new load
                new_load = carried - dmd

                # Energy to go home from d with lighter load
>>>>>>> 59e2c9104cddb354c1322830da7a087801ccb849
                dist_back_km = dist_km[(d, warehouse)]
                if dist_back_km == float("inf"):
                    continue
                back_e = move_energy_kwh(dist_back_km, max(new_load, 0.0), cruise_speed_kmh)
<<<<<<< HEAD
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
=======

                # We only require enough for cruise; landing_E is reserved separately
                need = go_e + back_e
                if need <= available_for_motion:
                    candidates.append((dist_to_km, d, go_e, back_e, new_load))

            if not candidates:
                # Try to go home directly if we're not already there
                if current != warehouse:
                    dist_home_km = dist_km[(current, warehouse)]
                    back_e = move_energy_kwh(dist_home_km, max(carried, 0.0), cruise_speed_kmh)

                    reserved_for_landing = landing_E
                    available_for_motion = energy - reserved_for_landing
                    if back_e > available_for_motion:
                        # Can't safely return; trip ends wherever we are
                        break

                    p = path[(current, warehouse)]
                    trip_path += (p[1:] if p and p[0] == current else p)
                    trip_legs.append({
                        'type': 'move',
                        'from': current,
                        'to': warehouse,
                        'distance_km': dist_home_km,
                        'load_before_kg': max(carried, 0.0),
                        'energy_kwh': back_e
>>>>>>> 59e2c9104cddb354c1322830da7a087801ccb849
                    })
                    energy -= back_e
                    grand_total_energy += back_e
                    current = warehouse
<<<<<<< HEAD
                break

            candidates.sort(key=lambda x: x[0])
            dist_to_km, target, go_e, back_e, new_load = candidates[0]
            p = path[(current, target)]
            trip_path += (p[1:] if p and p[0] == current else p)
            trip_legs.append({
                'type': 'move', 'from': current, 'to': target,
                'distance_km': dist_to_km, 'load_before_kg': carried, 'energy_kwh': go_e
=======

                # end of this trip
                break

            # Pick nearest feasible candidate
            candidates.sort(key=lambda x: x[0])
            dist_to_km, target, go_e, back_e, new_load = candidates[0]

            # Fly from current -> target
            p = path[(current, target)]
            trip_path += (p[1:] if p and p[0] == current else p)

            trip_legs.append({
                'type': 'move',
                'from': current,
                'to': target,
                'distance_km': dist_to_km,
                'load_before_kg': carried,
                'energy_kwh': go_e
>>>>>>> 59e2c9104cddb354c1322830da7a087801ccb849
            })
            energy -= go_e
            grand_total_energy += go_e
            current = target

<<<<<<< HEAD
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
=======
            # Deliver payload at target
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

            # If empty and not at warehouse, go home empty (still respecting landing reserve)
            if carried == 0 and current != warehouse:
                dist_home_km = dist_km[(current, warehouse)]
                back_e2 = move_energy_kwh(dist_home_km, 0.0, cruise_speed_kmh)

                reserved_for_landing = landing_E
                available_for_motion = energy - reserved_for_landing
                if back_e2 > available_for_motion:
                    # Can't safely return; end trip here
                    break

                p = path[(current, warehouse)]
                trip_path += (p[1:] if p and p[0] == current else p)
                trip_legs.append({
                    'type': 'move',
                    'from': current,
                    'to': warehouse,
                    'distance_km': dist_home_km,
                    'load_before_kg': 0.0,
                    'energy_kwh': back_e2
>>>>>>> 59e2c9104cddb354c1322830da7a087801ccb849
                })
                energy -= back_e2
                grand_total_energy += back_e2
                current = warehouse
                break

<<<<<<< HEAD
=======
        # End-of-trip landing
        trip_legs.append({'type': 'landing', 'energy_kwh': landing_E})
        grand_total_energy += landing_E
        energy -= landing_E

>>>>>>> 59e2c9104cddb354c1322830da7a087801ccb849
        print(f"\n=== Trip {trip_idx} ===")
        if served_this_trip:
            print("Delivered:", ", ".join(f"{pt} (x{dmd} kg)" for pt, dmd in served_this_trip))
        else:
            print("No deliveries completed on this trip.")
<<<<<<< HEAD
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
=======

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
            elif leg['type'] == 'takeoff':
                print(f"  {i:02d}. TAKEOFF energy_kwh={leg['energy_kwh']:.3f}")
            elif leg['type'] == 'landing':
                print(f"  {i:02d}. LANDING energy_kwh={leg['energy_kwh']:.3f}")

        print(f"Trip {trip_idx} ended at: {current}")
        print("Remaining deliveries:", remaining if remaining else "None")

        # Check if any future trip is feasible with remaining deliveries
        if remaining:
            can_start = False
            start_load = min(carry_capacity, sum(demands[x] for x in remaining))
            max_motion_energy = max(
                0.0, battery_capacity_kwh - (TAKEOFF_KWH + LANDING_KWH)
            )

>>>>>>> 59e2c9104cddb354c1322830da7a087801ccb849
            for d in remaining:
                dmd = demands[d]
                if dmd > carry_capacity:
                    continue
                dist_to_km = dist_km[(warehouse, d)]
                dist_back_km = dist_km[(d, warehouse)]
                if dist_to_km == float("inf") or dist_back_km == float("inf"):
                    continue
<<<<<<< HEAD
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
=======

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

>>>>>>> 59e2c9104cddb354c1322830da7a087801ccb849

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

    polt_energy()
    plotting_take_off()
    plotting_landing() 
    plotting_drone_moving()
