import numpy as np
import matplotlib.pyplot as plt

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
