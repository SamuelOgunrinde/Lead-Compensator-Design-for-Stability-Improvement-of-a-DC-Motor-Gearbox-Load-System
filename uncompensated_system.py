# Uncompensated DC motor–gearbox–load system  
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
print('G(S) =', G)

#######################################################################
# Uncompensated control system
#######################################################################

K = 1.0 # system gain

# uncompensated open-loop control system
L = ct.series(K, G)
# print('L(S) =', L)

# uncompensated feedback control system
T = ct.feedback(L, 1)
# print('T(S) =', T)

# system's type
ct.pzmap(L, grid=True)

# system's stability
ct.pzmap(T, grid=True)

#######################################################################
# Frequency-domain analysis
#######################################################################

# bode plot of uncompensated system
plt.figure(2)
ct.bode_plot(L, dB=True, 
                  title='Bode plot of uncompensated control system', 
                  display_margins=True)

# stability margins of uncompensated system
gm, pm, wcg, wcp = ct.margin(L)

print(f"Gain Margin (GM):       {gm:.4g}")
print(f"Phase Margin (PM):      {pm:.4f} deg")
print(f"Gain crossover freq:    {wcg:.4f} rad/s")
print(f"Phase crossover freq:   {wcp:.4f} rad/s")

#######################################################################
# Time-domain analysis
#######################################################################

# step response of the uncompensated system
t, y = ct.step_response(T)
plt.figure(3)
plt.plot(t, y)
plt.title('Step response of uncompensated system')

# performance metrics
info = ct.step_info(T)
print('Performance metrics')
for key, value in info.items():
    print(f'{key}: {value}')

plt.show()