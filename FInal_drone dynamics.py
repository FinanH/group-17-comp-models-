import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ----------------------------
# Drone + environment parameters
# ----------------------------
g = 9.81                # gravity (m/s^2)
rho = 1.225             # air density (kg/m^3)
C_dx = 0.8              # Drag coefficient horizontal
C_dz = 1                # Drag coefficient vertical
A_front = 1.154         # Effective front area of drone (m^2)
A_top = 0.175674        # Effective top area of drone (m^2)
A_disk = 1.34           # total rotor disk area (m^2)
eta = 0.75              # overall efficiency (motor * prop)
Vb = 133.2              # battery voltage (V)
Cb = 27                 # battery capacity (Ah)
P_av = 12               # avionics power (W)
usable_frac = 0.9
E_avail = usable_frac * Vb * Cb * 3600  # convert Wh → J

m_frame = 5.93
m_payload = 6
m_battery = 9.5
m_tot = m_frame + m_payload + m_battery

# ----------------------------
# Flight targets
# ----------------------------
vx_target = 18.0       # forward cruise (m/s)
x_target = 2000        # target horizontal distance (m)
z_target = 0
transition_time = 3    # tilt transition (s)

# Braking tilt
theta_cruise = np.deg2rad(25)  # cruising tilt
theta_brake = np.deg2rad(-20)  # braking tilt (negative)
theta_level = np.deg2rad(0)    # level


# ----------------------------
# Robust braking distance solver using solve_ivp + event
# ----------------------------
def braking_distance_ivp(v0, theta_b):
    """
    Integrate 1D braking dynamics until v = 0.
    Uses solve_ivp with event (root finding) to stop automatically.
    """
    def dyn(t, y):
        v, x = y
        Drag_x = 0.5 * rho * C_dx * A_front * max(v, 0)**2
        F_thrust_x = -m_tot * g * np.sin((abs(theta_b)))  # opposite to +x
        a = (F_thrust_x - Drag_x) / m_tot
        return [a, v]

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


# ----------------------------
# Drone dynamics
# ----------------------------
def drone_dynamics(t, y):
    z, vz, x, vx, E = y

    # Drag forces
    Drag_x = 0.5 * rho * C_dx * A_front * vx**2
    Drag_z = 0.5 * rho * C_dz * A_top * vz**2

    # Determine tilt (cruise → brake → level)
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


# ----------------------------
# Event functions for root finding
# ----------------------------

def reach_target(t, y):
    """Stop when drone reaches target distance."""
    return y[2] - x_target
reach_target.terminal = True
reach_target.direction = 1

def stop_at_zero_vx(t, y):
    """Stop when horizontal velocity crosses zero (from positive)."""
    return y[3]
stop_at_zero_vx.terminal = True
stop_at_zero_vx.direction = -1


# ----------------------------
# Integration setup
# ----------------------------
y0 = [0, 0, 0, 0, E_avail]
t_span = (0, 2200)
t_eval = np.linspace(*t_span, 5000)

# root finding occurs automatically via event detection:
sol = solve_ivp(
    drone_dynamics, t_span, y0, t_eval=t_eval,
    rtol=1e-6, atol=1e-8,
    events=[reach_target, stop_at_zero_vx]
)

# ----------------------------
# Extract results
# ----------------------------
z = sol.y[0]
vz = sol.y[1]
x = sol.y[2]
vx = sol.y[3]
E = sol.y[4]
t = sol.t
P_loss = (E_avail - E)/3600  # Wh

print("\nEvent times:", sol.t_events)
print("Reached final x =", x[-1], "m; final vx =", vx[-1], "m/s\n")

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

print(f"Reached velocity: {vx[-1]:.2f} m/s in {t[-1]:.2f} s")
print(f"Traveled distance: {x[-1]:.2f} m in {t[-1]:.2f} s")
print(f"Energy used: {(E_avail - E[-1])/3600:.2f} Wh "
      f"({100*(E_avail-E[-1])/E_avail:.2f}% of battery)")