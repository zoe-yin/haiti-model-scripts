# Compute dCFS with shear traction in the rake direction of the final slip
# (0.6 being mu_s in the last line)
# ignore "the begin_2nd_event" it was for the Turkish doublet, 
# we wanted to compute the Coulomb stress change due to the first event on the second one, 
# and both events where simulated in the same run
# anyway you want to use the last time step

begin_2nd_event = compute_time_indices(sx, [100])[0]
print(begin_2nd_event)
T_s = sx.ReadData("T_s", begin_2nd_event)
T_d = sx.ReadData("T_d", begin_2nd_event)
P_n = sx.ReadData("P_n", begin_2nd_event)
Ts0 = sx.ReadData("Ts0", 0)
Td0 = sx.ReadData("Td0", 0)
rake = np.arctan2(Td0, Ts0)
Shear_in_slip_direction = np.cos(rake) * T_s + np.sin(rake) * T_d
myData = Shear_in_slip_direction + 0.6 * P_n