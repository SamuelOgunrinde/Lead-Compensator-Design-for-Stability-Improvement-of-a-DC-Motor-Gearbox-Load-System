# Compensated DC motor–gearbox–load system 
import numpy as np
import matplotlib.pyplot as plt
import control as ct

#######################################################################
# Plant model
#######################################################################

# model parameters
V = 12; I = 1.5; R = 2.6; L = 0.18; J_m= 3.9e-7; 
J_o = 2.2e-3; n_m = 14; n_l = 70; B_eq = 1e-6; 
k_t = 0.0077; k_e = 0.0077; n = n_l/n_m; 

a = n*k_t
b = L*(J_o + n**2*J_m)
c = R*(J_o + n**2*J_m) + L*B_eq
d = R*B_eq + n**2*k_e*k_t

# plant transfer function
num_g = np.array ([a])
den_g = np.array ([b, c, d, 0])
G = ct.tf(num_g, den_g)
# print('G(S) =', G)

#######################################################################
# Uncompensated control system
#######################################################################

K = 1.0 # system gain

# uncompensated open-loop control system
L = ct.series(K, G)

# uncompensated feedback control system
T = ct.feedback(L, 1)

# stability margins
gm, pm, wcg, wcp = ct.margin(L)

#######################################################################
# Phase lead compensation
#######################################################################

mag, phase, omega = ct.frequency_response(L) # Bode data

phase_deg = np.degrees(phase) # Convert phase from radians to degrees

mag_db = 20 * np.log10(mag) # Convert magnitude to dB

pm_desired = 45 # desired phase margin

pm_current = pm # current phase margin

pm_tolerance = 10 # Safety allowance

# additional phase margin required
phi = pm_desired - pm_current +  pm_tolerance
# print("Additional phase margin =", phi, "degrees")

# calculate alpha for lead compensator
alpha = (1 + np.sin(np.radians(phi))) / (1 - np.sin(np.radians(phi)))
# print("Alpha =", alpha)

# target magnitude from uncompensated system's bode plot
target_mag_db = -10 * np.log10(alpha)

# find frequency where magnitude is closest to target magnitude
index = np.argmin(np.abs(mag_db - target_mag_db))

wm = omega[index] # frequency at the target magnitude

# calculte time constant for lead compensator
tau = 1 / (wm * np.sqrt(alpha))
# print('Time constant = ',tau)

# phase lead compensator transfer function
num_C = np.array([alpha * tau, 1])
den_C = np.array([tau, 1])
C = ct.tf(num_C, den_C)
print('C(S) =', C)

#######################################################################
# Compensated control system
#######################################################################

K_c = 0.8 # lead compensator gain

## compensated open-loop control system
L_new = ct.series(K, G, K_c, C)
# print('L(S) =', L_new)

# compensated feedback control system
T_new = ct.feedback(L_new, 1)
# print('T(S) =', T_new)

#######################################################################
# Frequency-domain analysis
#######################################################################

# bode plot of compensated system
plt.figure(4)
ct.bode_plot(L_new, dB=True,
                  title='Bode plot of compensated system', 
                  display_margins=True)

# stability margins of compensated system
gm_new, pm_new, wcg_new, wcp_new = ct.margin(L_new)

print(f"Gain Margin (GM):       {gm_new:.4g}")
print(f"Phase Margin (PM):      {pm_new:.4f} deg")
print(f"Gain crossover freq:    {wcg_new:.4f} rad/s")
print(f"Phase crossover freq:   {wcp_new:.4f} rad/s")

#######################################################################
# Time-domain analysis
#######################################################################

# step response of compensated system
t, y = ct.step_response(T_new)
plt.figure(5)
plt.plot(t, y)
plt.title('Step response of compensated system')

# performance metrics
info = ct.step_info(T_new)
print('Performance metrics')
for key, value in info.items():
    print(f'{key}: {value}')

plt.show()