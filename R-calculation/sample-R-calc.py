import numpy as np


Ts0 = -1.56281e+07
Td0 = 3.28291e+06
Pn0 = -4.8295e+07
T_s = -0.00937244
T_d = 0.0171799
P_n = -0.00619755
Mud = 0.6

mu_st = 0.6
mu_dy = 0.2
cohesion = -0.5e6

R = (np.sqrt(np.power(Ts0, 2) + np.power(Td0, 2)) - mu_dy * Pn0) / (
    (mu_st - mu_dy)* Pn0 + cohesion)

print(R)