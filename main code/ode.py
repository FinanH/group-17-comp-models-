# ode.py

from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp


# Physical constants & battery

# Here we define one specific battery pack and keep everything consistent with it.

Vb = 55                # battery voltage (V)
Cb = 27                 # Battery capacity (Ah)
usable_frac = 0.9       # We don't want to drain it 100% in real life

# Usable energy from the pack:
#   V * Ah = Wh
#   Wh * 3600 = J     (because 1 Wh = 3600 J)
E_avail = usable_frac * Vb * Cb * 3600  # J (this is what the ODEs see)

# Same pack, but in kWh for route planning & summary plots
BATTERY_CAPACITY_FROM_CELLS_KWH = usable_frac * Vb * Cb / 1000.0  # kWh ≈ 0.0799

# A very simple mass breakdown

m_frame = 5.93
m_payload_nom = 6
m_battery = 3.57
m_tot = m_frame + m_payload_nom + m_battery


# Vertical target height/distances and velocaties
alt_target = 30.0       # target altatude

v_t_target = 2          # target velocaty take of target 

v_l_target = -0.5       # velocaty landing traget 

vx_target = 11.0        # target cruse velocity

x_target = 2000        # target horizontal distance (m)
z_target = 0
transition_time = 3    # tilt transition (s)

# Braking tilt
theta_cruise = np.deg2rad(25)  # cruising tilt
theta_brake = np.deg2rad(-20)  # braking tilt (negative)
theta_level = np.deg2rad(0)    # level


# Drone + environment parameters


g = 9.81                # gravity (m/s^2)
rho = 1.225             # air density (kg/m^3)
C_dx = 0.8              # Drag coefficient horizontal
C_dz = 1                # Drag coefficient vertical
A_front = 1.154         # Effective front area of drone (m^2)
A_top = 0.175674        # Effective top area of drone (m^2)
A_disk = 1.34           # total rotor disk area (m^2)
eta = 0.75              # overall efficiency (motor * prop)
P_av = 12               # avionics power (W), always on

vh = np.sqrt((m_tot * g) / (2 * rho * A_disk))  # hover induced velocity (m/s)

# Coefficients for the power model
k1 = 1.1 / eta
k2 = 0.5 * rho * C_dz * A_top
k3 = P_av

# Keep everything in one params dict so we can pass it around cleanly
params = {
    "g": g,
    "rho": rho,
    "C_dz": C_dz,
    "C_dx": C_dx,
    "A_top": A_top,
    "A_disk": A_disk,
    "eta": eta,
    "m_frame": m_frame,
    "m_payload": m_payload_nom,
    "m_battery": m_battery,
    "m_dry": m_frame + m_battery,
    "Vb": Vb,
    "Cb": Cb,
    "P_av": P_av,
    "vh": vh,
    "k1": k1,
    "k2": k2,
    "k3": k3,
    "v_hover": vh,
    "E_avail": E_avail,
}

# For the grid plot (used to convert kWh to battery %)
BATTERY_CAPACITY_KWH_GLOBAL: Optional[float] = None

# Globals for vertical ODEs (we reuse the same dynamics with different masses)
M_TAKEOFF = m_tot
VH_TAKEOFF = vh
M_LANDING = m_tot
VH_LANDING = vh


# Takeoff / landing dynamics (ODE)


def takeoff_dynamics(t, y):
    z, vz, E = y

    #  Control law / thrust command 
    # Simple proportional controller on vertical speed:
    k_p = 6.0         # if vz not euqual to v target increase acceleration by 5
                            
    Thurst = m_tot * (g + k_p * (v_t_target - vz))  # Thurst (N)
    Drag_z = 0.5*rho*C_dz*A_top*vz* abs(vz)        # vertical drag (N)
   
    #  Power model 
   
    # Induced velocity 
    vi = np.sqrt(Thurst / (2 * rho * A_disk))
  
    P_ind = Thurst * vi                      # induced power (W)
    P_elec = (P_ind / eta) + P_av            # electrical power 

    #  Dynamics 
    dzdt = vz
    dvzdt = (Thurst - (m_tot * g) - Drag_z )/ m_tot
    dEdt = -P_elec

    return [dzdt, dvzdt, dEdt]


def landing_dynamics(t, y):
    z, vz, E = y

    #  Control law / thrust command 
    # Proportional controller on vertical speed:
    k_p = 0.5  # if vz not euquak to v target increase acceleration by 0.5
    
    Thurst = max(0.0, m_tot * (g + k_p * (v_l_target - vz)))  # Thurst (N)
   
    Drag_z = 0.5*rho*C_dz*A_top*(vz)* abs(vz)           # vertical drag (N)
    
    #  Power model 
    #  Induced velocity 
    vi = np.sqrt(Thurst / (2 * rho * A_disk))
    P_ind = Thurst * vi                     # induced power (W)
    P_elec = (P_ind / eta) + P_av           # electrical power 

    #  Dynamics 
    dzdt = vz
    dvzdt = (Thurst - (m_tot * g) - Drag_z )/ m_tot
    dEdt = -P_elec

    return [dzdt, dvzdt, dEdt]



# Forward-flight power

# This is a simple induced + profile drag + fixed term model.

def power_model(W, v, params):
    """
    Power in forward flight, including induced + drag + fixed components.

    W  = payload mass (kg)
    v  = airspeed (m/s)
    """
    g_ = params["g"]
    m0 = params["m_dry"]
    vh0 = params["v_hover"]
    k1p = params["k1"]
    k2p = params["k2"]
    k3p = params["k3"]

    m = m0 + W
    P_ind = k1p * ((m * g_) ** 1.5) / np.sqrt(v ** 2 + vh0 ** 2)
    P_drag = k2p * v ** 3
    P_fixed = k3p
    return P_ind + P_drag + P_fixed   # Watts


#
# Takeoff / landing energy 
#
# We compute a "baseline" takeoff & landing energy with the nominal mass,
# then scale with total mass for other payloads (simple linear scaling assumption).


def compute_landing_energy_kwh():
    """
    Integrate landing_dynamics from z=alt_target down to 0 using solve_ivp
    to get baseline landing energy (for nominal mass).
    """
    global M_LANDING, VH_LANDING
    M_LANDING = m_tot
    VH_LANDING = np.sqrt((M_LANDING * g) / (2 * rho * A_disk))

    # Start at target altitude, zero vertical speed, full usable energy
    y0 = [alt_target, 0.0, E_avail]
    t_span = (0.0, 60.0)

    def event_ground_reached(t, y):
        return y[0]  # z = 0 to ground

    event_ground_reached.terminal = True
    event_ground_reached.direction = -1

    sol = solve_ivp(
        landing_dynamics,
        t_span,
        y0,
        events=event_ground_reached,
        rtol=1e-6,
        atol=1e-8,
        max_step=0.5
    )

    E_end = sol.y[2][-1]
    E_used_J = E_avail - E_end
    return E_used_J / 3.6e6  # J to kWh


# Baseline vertical costs (for the nominal mass)
LANDING_KWH_BASE = compute_landing_energy_kwh()
TAKEOFF_KWH_BASE: Optional[float] = None


def landing_energy_kwh_for(payload_kg: float) -> float:
    """
    Landing energy for a given payload.
    Same scaling assumption as takeoff.
    """
    total_mass = m_frame + m_battery + payload_kg
    return LANDING_KWH_BASE * (total_mass / m_tot)


def takeoff_energy_kwh_for(payload_kg: float) -> float:
    """
    Takeoff energy for a given payload.

    Uses the existing takeoff_dynamics ODE once (baseline mass),
    then scales linearly with total mass, same idea as landing_energy_kwh_for.
    The E state is the 3rd component of the ODE state vector.
    """
    global TAKEOFF_KWH_BASE, M_TAKEOFF, VH_TAKEOFF

    # Compute baseline once (for nominal total mass m_tot)
    if TAKEOFF_KWH_BASE is None:
        M_TAKEOFF = m_tot
        VH_TAKEOFF = np.sqrt((M_TAKEOFF * g) / (2 * rho * A_disk))

        # Start at ground, zero vertical speed, full usable energy
        y0 = [0.0, 0.0, E_avail]
        t_span = (0.0, 60.0)

        def event_alt_reached(t, y):
            # stop when altitude reaches alt_target
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
            max_step=0.1
        )

        # y[2] is E(t), which came from integrating dE/dt (the 3rd returned value of takeoff_dynamics)
        E_end = sol.y[2][-1]
        E_used_J = E_avail - E_end
        TAKEOFF_KWH_BASE = E_used_J / 3.6e6  # J -> kWh

    # Scale baseline takeoff energy with total mass
    total_mass = m_frame + m_battery + payload_kg
    return TAKEOFF_KWH_BASE * (total_mass / m_tot)


# Cruise energy
# This is “distance times power”, but with the power coming from power_model.
# Braking distance solver using solve_ivp + event

def braking_distance_ivp(v0, theta_b):
    """
    Integrate 1D braking dynamics until v = 0.
    Uses solve_ivp with event (root finding) to stop automatically.
    """
    def dyn(t, y):
        v, x = y
        Drag_x = 0.5 * rho * C_dx * A_front * max(v, 0)**2
        F_thrust_x = -m_tot * g * np.sin((abs(theta_b)))  # opposite to +x
        dvxdt = (F_thrust_x - Drag_x) / m_tot
        return [dvxdt, v]

    # Stop when v = 0 (root-finding event)
    def stop_v0(t, y):
        return y[0]  # velocity
    stop_v0.terminal = True
    stop_v0.direction = -1  # trigger when v decreases through zero

    sol = solve_ivp(dyn, (0, 100), [v0, 0.0], events=stop_v0, max_step=0.1)
    return sol.y[1, -1]  # return stopping distance


# Compute braking distance and cruise start
d_brake = braking_distance_ivp(vx_target, theta_brake)
x_cruise_start = x_target - d_brake
print(f"Active braking distance ≈ {d_brake:.1f} m | Begin braking at x ≈ {x_cruise_start:.1f} m")



# Drone dynamics

def move_energy(t, y):
    z, vz, x, vx, E = y

    # Drag forces
    Drag_x = 0.5 * rho * C_dx * A_front * vx**2
    Drag_z = 0.5 * rho * C_dz * A_top * vz**2

    # Determine tilt (cruise to brake to level)
    if x < x_cruise_start:
        theta = theta_cruise
    elif x < x_target:
        theta = theta_brake if vx > 0 else 0.0  # only brake while moving forward
    else:
        theta = theta_level

    # Vertical control
    k_p_z = 10.0
    Thrust = m_tot * (g + k_p_z * (z_target - z)) / np.cos(theta)

    # Power model
    vi = np.sqrt(Thrust / (2 * rho * A_disk))
    P_ind = Thrust * vi
    P_par = abs(Drag_x * vx)
    P_elec = (P_ind + P_par)/eta + P_av

    # Dynamics
    dzdt = vz
    dvzdt = (Thrust*np.cos(theta) - m_tot*g - Drag_z)/m_tot
    dxdt = vx
    dvxdt = (Thrust*np.sin(theta) - Drag_x)/m_tot
    dEdt = -P_elec

    return [dzdt, dvzdt, dxdt, dvxdt, dEdt]


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



# Debug: vertical energy table


def print_vertical_energy_table():
    """
    Just print out how much takeoff/landing energy the ODEs are giving us
    for a few different payloads.
    """
    print("Takeoff / landing energy vs payload mass (from ODEs, scaled):")
    print("  payload_kg | takeoff_kWh | landing_kWh")
    for payload in [0.0, 2.0, 5.0, 10.0]:
        to = takeoff_energy_kwh_for(payload)
        ld = landing_energy_kwh_for(payload)
        print(f"    {payload:7.1f} |    {to:7.4f} |     {ld:7.4f}")
    print()
