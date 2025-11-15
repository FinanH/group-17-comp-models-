# config.py
import numpy as np
from typing import Optional

# ----------------------------
# Physical constants & battery
# ----------------------------

Vb = 133                # battery voltage (V)
Cb = 27                 # capacity (Ah)
usable_frac = 0.9       # usable fraction

# Usable energy in Joules
E_avail = usable_frac * Vb * Cb * 3600.0

# Same pack, in kWh
BATTERY_CAPACITY_FROM_CELLS_KWH = usable_frac * Vb * Cb / 1000.0  # ≈ 0.0799 kWh

# Masses
m_frame = 5.93
m_payload_nom = 6.0
m_battery = 9.5
m_tot = m_frame + m_payload_nom + m_battery

# ----------------------------
# Drone + environment parameters
# ----------------------------

g = 9.81
rho = 1.225
C_dx = 0.8
C_dz = 1.0
A_front = 1.154
A_top = 0.175674
A_disk = 1.34
eta = 0.75
P_av = 12.0

vh = np.sqrt((m_tot * g) / (2 * rho * A_disk))  # hover induced velocity

# Power model coefficients
k1 = 1.1 / eta
k2 = 0.5 * rho * C_dz * A_top
k3 = P_av

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

# For plotting (global capacity for % SOC)
BATTERY_CAPACITY_KWH_GLOBAL: Optional[float] = None
