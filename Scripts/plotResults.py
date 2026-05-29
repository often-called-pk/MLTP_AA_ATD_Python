import numpy as np
import scipy.io as sio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import casadi as ca
import sys
import os

# Ensure the correct paths are loaded to fetch parameters and models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Parameters')))
from userOpts import pt, vp, AeroConfig, ATD_config
import vehModel as vm

# Load solution data from the solver
try:
    sol_data = sio.loadmat('Results/solution.mat', squeeze_me=True)
except FileNotFoundError:
    print("Run MLTP.py first to generate Results/solution.mat")
    sys.exit()

x_opt = sol_data['x_opt']
u_opt = sol_data['u_opt']
y_opt = sol_data['y_opt']
s_knot = sol_data['s_knot']

# Isolate lap timestamps using Beacon Markers
beacon_start_idx = 0
beacon_end_idx = len(s_knot) - 1
print(f"Lap isolated using Beacon Markers at index {beacon_start_idx} and {beacon_end_idx}.")

N_points = x_opt.shape[1]

# Reconstruct derived vehicle data using CasADi
# This maps exactly to the SDI post-processing in the MATLAB framework
calc_forces = ca.Function('calc_forces', [vm.x, vm.u, vm.y, vm.pv],
    [vm.fx_fl, vm.fx_fr, vm.fx_rl, vm.fx_rr,
     vm.fy_fl, vm.fy_fr, vm.fy_rl, vm.fy_rr,
     vm.fz_fl, vm.fz_fr, vm.fz_rl, vm.fz_rr,
     vm.f_lift, vm.f_drag, vm.f_lift_fl, vm.f_lift_fr, vm.f_lift_rl, vm.f_lift_rr, vm.f_side,
     vm.mu_fl, vm.mu_fr, vm.mu_rl, vm.mu_rr,
     vm.T_fl, vm.T_fr, vm.T_rl, vm.T_rr, vm.P_motor])

# Reconstruct track curvature interpolation
from userOpts import track
from scipy.interpolate import interp1d
k_interp = interp1d(track['s'], track['k'], kind='linear', fill_value="extrapolate")
pv_opt = k_interp(s_knot).reshape(1, -1)

# Evaluate physics across the lap
forces = np.zeros((28, N_points))
for i in range(N_points):
    res = calc_forces(x_opt[:, i]/vm.x_s, u_opt[:, i]/vm.u_s, y_opt[:, i]/vm.y_s, pv_opt[:, i])
    for j in range(28):
        forces[j, i] = float(res[j])

# Map reconstructed arrays
fx_fl, fx_fr, fx_rl, fx_rr = forces[0,:], forces[1,:], forces[2,:], forces[3,:]
fy_fl, fy_fr, fy_rl, fy_rr = forces[4,:], forces[5,:], forces[6,:], forces[7,:]
fz_fl, fz_fr, fz_rl, fz_rr = forces[8,:], forces[9,:], forces[10,:], forces[11,:]
f_lift, f_drag = forces[12,:], forces[13,:]
f_lift_fl, f_lift_fr, f_lift_rl, f_lift_rr, f_side = forces[14,:], forces[15,:], forces[16,:], forces[17,:], forces[18,:]
mu_fl, mu_fr, mu_rl, mu_rr = forces[19,:], forces[20,:], forces[21,:], forces[22,:]
T_fl, T_fr, T_rl, T_rr = forces[23,:], forces[24,:], forces[25,:], forces[26,:]
P_motor = forces[27,:]

# Map inputs
T_motor = u_opt[0, :]
T_brake = u_opt[1, :]
delta = u_opt[-1, :]

# Setup Plotly Subplots (3 rows, 4 columns)
fig = make_subplots(rows=3, cols=4, subplot_titles=(
    "1. Velocity", "2. Mu Limits", "3. Load Transfer", "4. Fx (Longitudinal)",
    "5. Steering (delta)", "6. Torque Distribution", "7. Aerodynamics", "8. Fy (Lateral)",
    "9. Motor / Brake", "10. Wheel Torques", "11. Aero Forces", "12. Fz (Vertical)"
))

def add_line(fig, row, col, y_data, name):
    fig.add_trace(go.Scatter(x=s_knot, y=y_data, mode='lines', name=name), row=row, col=col)

# 1x1 - Velocity
add_line(fig, 1, 1, x_opt[0, :], 'vx')

# 2x1 - Steering Angle
add_line(fig, 2, 1, delta, 'delta')

# 3x1 - Torque
add_line(fig, 3, 1, T_motor, 'T_motor')
add_line(fig, 3, 1, T_brake, 'T_brake')

# 1x2 - Mu Lim
mu_lim_fl = (mu_fl**2 - (fx_fl**2 + fy_fl**2)/fz_fl**2) / mu_fl**2
mu_lim_fr = (mu_fr**2 - (fx_fr**2 + fy_fr**2)/fz_fr**2) / mu_fr**2
mu_lim_rl = (mu_rl**2 - (fx_rl**2 + fy_rl**2)/fz_rl**2) / mu_rl**2
mu_lim_rr = (mu_rr**2 - (fx_rr**2 + fy_rr**2)/fz_rr**2) / mu_rr**2
add_line(fig, 1, 2, mu_lim_fl, 'mu_lim_fl')
add_line(fig, 1, 2, mu_lim_fr, 'mu_lim_fr')
add_line(fig, 1, 2, mu_lim_rl, 'mu_lim_rl')
add_line(fig, 1, 2, mu_lim_rr, 'mu_lim_rr')

# 2x2 - Torque Distribution
if pt.ATD == 0:
    add_line(fig, 2, 2, P_motor, 'P_motor (kW)')
    add_line(fig, 2, 2, T_motor, 'T_motor')
elif pt.ATD == 1:
    add_line(fig, 2, 2, u_opt[2, :], 'ATD_FL')
    add_line(fig, 2, 2, u_opt[3, :], 'ATD_FR')
    add_line(fig, 2, 2, u_opt[4, :], 'ATD_RL')
    add_line(fig, 2, 2, u_opt[5, :], 'ATD_RR')

# 3x2 - Torque per wheel
add_line(fig, 3, 2, T_fl, 'T_fl')
add_line(fig, 3, 2, T_fr, 'T_fr')
add_line(fig, 3, 2, T_rl, 'T_rl')
add_line(fig, 3, 2, T_rr, 'T_rr')

# 1x3 - Load Transfer
add_line(fig, 1, 3, y_opt[0, :], 'loadTransferX')
add_line(fig, 1, 3, y_opt[1, :], 'loadTransferY')

# 2x3 - Aerodynamics
if vp.ActAero == 0:
    add_line(fig, 2, 3, f_lift, 'f_lift')
    add_line(fig, 2, 3, f_drag, 'f_drag')
elif vp.ActAero == 1:
    idx = 6 if pt.ATD == 1 else 2
    add_line(fig, 2, 3, u_opt[idx, :], 'activeAeroRW')
elif vp.ActAero == 2:
    idx = 6 if pt.ATD == 1 else 2
    add_line(fig, 2, 3, u_opt[idx, :], 'activeAeroFW')
    add_line(fig, 2, 3, u_opt[idx+1, :], 'activeAeroRW')
elif vp.ActAero == 3:
    idx = 6 if pt.ATD == 1 else 2
    add_line(fig, 2, 3, u_opt[idx, :], 'activeAeroFL')
    add_line(fig, 2, 3, u_opt[idx+1, :], 'activeAeroFR')
    add_line(fig, 2, 3, u_opt[idx+2, :], 'activeAeroRW')
    add_line(fig, 2, 3, u_opt[idx+3, :], 'activeAeroTW')

# 3x3 - Aerodynamic forces
add_line(fig, 3, 3, f_lift_fl, 'f_lift_fl')
add_line(fig, 3, 3, f_lift_fr, 'f_lift_fr')
add_line(fig, 3, 3, f_lift_rl, 'f_lift_rl')
add_line(fig, 3, 3, f_lift_rr, 'f_lift_rr')
add_line(fig, 3, 3, f_side, 'f_side')

# 1x4 - Longitudinal tyre forces
add_line(fig, 1, 4, fx_fl, 'fx_fl')
add_line(fig, 1, 4, fx_fr, 'fx_fr')
add_line(fig, 1, 4, fx_rl, 'fx_rl')
add_line(fig, 1, 4, fx_rr, 'fx_rr')

# 2x4 - Lateral tyre forces
add_line(fig, 2, 4, fy_fl, 'fy_fl')
add_line(fig, 2, 4, fy_fr, 'fy_fr')
add_line(fig, 2, 4, fy_rl, 'fy_rl')
add_line(fig, 2, 4, fy_rr, 'fy_rr')

# 3x4 - Vertical tyre forces
add_line(fig, 3, 4, fz_fl, 'fz_fl')
add_line(fig, 3, 4, fz_fr, 'fz_fr')
add_line(fig, 3, 4, fz_rl, 'fz_rl')
add_line(fig, 3, 4, fz_rr, 'fz_rr')

# Format and launch dashboard
fig.update_layout(height=1000, width=1600, title_text="MLTP Telemetry Dashboard", template="plotly_white")
fig.show()