import casadi as ca
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
import time
import os

# Import parameters and models
from userOpts import *
from vehModel import *

# Helper to calculate Collocation Coefficients
def collocation_coeff(tau):
    d = len(tau)
    C = np.zeros((d + 1, d + 1))
    D = np.zeros(d + 1)
    B = np.zeros(d + 1)
    tau_root = np.insert(tau, 0, 0.0)
    for j in range(d + 1):
        p = np.poly1d([1.0])
        for r in range(d + 1):
            if r != j:
                p = np.polymul(p, np.poly1d([1.0, -tau_root[r]])) / (tau_root[j] - tau_root[r])
        D[j] = p(1.0)
        p_der = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = p_der(tau_root[r])
        B[j] = np.polyint(p)(1.0)
    return C, D, B

# Start time counter
start_time = time.time()

# Load initialisation data (Replaces importfile('Static_MidDF.mat'))
init_data = sio.loadmat('../Data/Barcelona/initialisation/Static_MidDF.mat', squeeze_me=True, struct_as_record=False)
data_init = init_data['data'].init

# Optimal Control Problem - Dynamics and objective function
L = sf

# Continuous time dynamics and objective function
f_dyn = ca.Function('f_dyn', [x, u, y, pv], [dx, L], ['x', 'u', 'y', 'pv'], ['dx', 'L'])

# Change of variable
f_sf = ca.Function('sf', [x, kappa], [sf], ['x', 'kappa'], ['sf'])

# Path Constraints
if pt.ATD == 0:
    nh = 9
elif pt.ATD == 1:
    nh = 10

# Constraints setup
BrTh_1 = (T_motor_n * T_brake_n) / 1e-3
ltx_eq = ((fx_fl*ca.cos(delta) + fx_fr*ca.cos(delta) - fy_fl*ca.sin(delta) - fy_fr*ca.sin(delta) + fx_rl + fx_rr + f_drag0)*(vp.hcg/vp.l) + (f_dragRW)*(vp.hw/vp.l) - ltx)/1e-3
lty_eq = (((fy_fl*ca.cos(delta) + fy_fr*ca.cos(delta) + fx_fl*ca.sin(delta) + fx_fr*ca.sin(delta) + fy_rl + fy_rr)*(vp.hcg/vp.t) + f_side*(vp.hw/vp.t)) - lty)/1e-3
motor_power = (pt.Pmax - Om_motor * T_motor) / pt.Pmax
motor_rpm = (pt.OMmax - Om_motor) / pt.OMmax

mu_lim_fl = (mu_fl**2 - (fx_fl**2 + fy_fl**2) / fz_fl**2) / mu_fl**2
mu_lim_fr = (mu_fr**2 - (fx_fr**2 + fy_fr**2) / fz_fr**2) / mu_fr**2
mu_lim_rl = (mu_rl**2 - (fx_rl**2 + fy_rl**2) / fz_rl**2) / mu_rl**2
mu_lim_rr = (mu_rr**2 - (fx_rr**2 + fy_rr**2) / fz_rr**2) / mu_rr**2

if pt.ATD == 0:
    hnames = ['mu_lim_fl', 'mu_lim_fr', 'mu_lim_rl', 'mu_lim_rr', 'motor_power', 'motor_rpm', 'ltx_eq', 'lty_eq', 'BrTh_1']
    h = ca.vertcat(mu_lim_fl, mu_lim_fr, mu_lim_rl, mu_lim_rr, motor_power, motor_rpm, ltx_eq, lty_eq, BrTh_1)
    h_lb = np.array([0, 0, 0, 0, 0, 0, -1, -1, -1])
    h_ub = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
elif pt.ATD == 1:
    ATD_eq = (1 - (ATD_FL + ATD_FR + ATD_RL + ATD_RR))
    hnames = ['mu_lim_fl', 'mu_lim_fr', 'mu_lim_rl', 'mu_lim_rr', 'motor_power', 'motor_rpm', 'ltx_eq', 'lty_eq', 'BrTh_1', 'ATD_eq']
    h = ca.vertcat(mu_lim_fl, mu_lim_fr, mu_lim_rl, mu_lim_rr, motor_power, motor_rpm, ltx_eq, lty_eq, BrTh_1, ATD_eq)
    h_lb = np.array([0, 0, 0, 0, 0, 0, -1, -1, -1, -1e-3])
    h_ub = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-3])

if nh != h.shape[0]:
    raise ValueError('Number of path constraints of the OCP is not consistent')

h_eq = ca.Function('h_eq', [x, u, y, pv], [h], ['x', 'u', 'y', 'pv'], ['h'])

# NLP - Collocation setup
tau = ca.collocation_points(OPT_d, 'legendre')
C, D, B = collocation_coeff(tau)

# Discretisation
N = int(np.round(track['s'][-1] / OPT_ds))
s_knot = np.linspace(np.min(track['s']), np.max(track['s']), N + 1)
dsk = np.diff(s_knot)

s_col = np.kron(dsk, tau) + np.kron(np.concatenate([[0], np.cumsum(dsk[:-1])]), np.ones(OPT_d))
s_full = np.zeros(N * (OPT_d + 1) + 1)
s_full[:-1] = np.kron(s_knot[:-1], np.concatenate([[1], np.zeros(OPT_d)])) + np.concatenate([np.zeros(N), s_col.reshape(OPT_d, N).T.flatten()])
s_full[-1] = s_knot[-1]

# Independent variables
k_knot = interp1d(track['s'], track['k'], kind='linear', fill_value="extrapolate")(s_knot)
k_col = interp1d(track['s'], track['k'], kind='linear', fill_value="extrapolate")(s_col)
k_full = interp1d(track['s'], track['k'], kind='linear', fill_value="extrapolate")(s_full)

pv_knot = k_knot.reshape(1, -1)
pv_col = k_col.reshape(1, -1)
pv_full = k_full.reshape(1, -1)

# Initial Guesses
s_init_track = np.linspace(np.min(track['s']), np.max(track['s']), data_init.x_opt.shape[1])

x_opt_init = interp1d(s_init_track, data_init.x_opt, axis=1, kind='linear', fill_value="extrapolate")(s_knot)
u_opt_init = interp1d(s_init_track, data_init.u_opt, axis=1, kind='linear', fill_value="extrapolate")(s_knot)
y_opt_init = interp1d(s_init_track, data_init.y_opt, axis=1, kind='linear', fill_value="extrapolate")(s_knot)

vx_0 = x_opt_init[0, :]
vy_0 = x_opt_init[1, :]
r_0 = x_opt_init[2, :]
n_0 = x_opt_init[3, :]
eps_0 = x_opt_init[4, :]
Om_fl_0 = vx_0 / vp.Rw
Om_fr_0 = vx_0 / vp.Rw
Om_rl_0 = vx_0 / vp.Rw
Om_rr_0 = vx_0 / vp.Rw

T_motor_0 = u_opt_init[0, :]
T_brake_0 = u_opt_init[1, :]
delta_0 = u_opt_init[2, :]

activeAeroRW_0 = np.zeros(N + 1)
activeAeroFW_0 = np.zeros(N + 1)
activeAeroFL_0 = np.zeros(N + 1)
activeAeroFR_0 = np.zeros(N + 1)
activeAeroTW_0 = np.zeros(N + 1)
ATD_0 = 0.25 * np.ones(N + 1)

ltx_0 = y_opt_init[0, :]
lty_0 = y_opt_init[1, :]

# Collect initial guesses mapping
x0 = np.vstack([vx_0, vy_0, r_0, n_0, eps_0, Om_fl_0, Om_fr_0, Om_rl_0, Om_rr_0]) / x_s[:, None]
y0 = np.vstack([ltx_0, lty_0]) / y_s[:, None]

u0_list = [T_motor_0, T_brake_0]
if pt.ATD == 1:
    u0_list.extend([ATD_0, ATD_0, ATD_0, ATD_0])
if vp.ActAero == 1:
    u0_list.append(activeAeroRW_0)
elif vp.ActAero == 2:
    u0_list.extend([activeAeroFW_0, activeAeroRW_0])
elif vp.ActAero == 3:
    u0_list.extend([activeAeroFL_0, activeAeroFR_0, activeAeroRW_0, activeAeroTW_0])
u0_list.append(delta_0)

u0 = np.vstack(u0_list) / u_s[:, None]
xc0 = np.kron(x0[:, :-1], np.ones((1, OPT_d))).flatten('F')

# NLP Formulation
Xk = ca.SX.sym('Xk', nx, N + 1)
Uk = ca.SX.sym('Uk', nu, N + 1)
Yk = ca.SX.sym('Yk', ny, N + 1)
Xkj = ca.SX.sym('Xkj', nx, N * OPT_d)

duk = ca.diff(Uk.T).T / ca.repmat(dsk, nu, 1)
dyk = ca.diff(Yk.T).T / ca.repmat(dsk, ny, 1)
dxk = ca.diff(Xk.T).T / ca.repmat(dsk, nx, 1)

duk2 = ca.horzcat(ca.diff(duk.T).T, ca.diff(duk.T).T[:, -1])
dyk2 = ca.horzcat(ca.diff(dyk.T).T, ca.diff(dyk.T).T[:, -1])
dxk2 = ca.horzcat(ca.diff(dxk.T).T, ca.diff(dxk.T).T[:, -1])

J_s = 1

# Boundary Conditions
x0_min = np.maximum(x_min, Xi / x_s - OPT_e)
x0_max = np.minimum(x_max, Xi / x_s + OPT_e)
xf_min = np.maximum(x_min, Xf / x_s - OPT_e)
xf_max = np.minimum(x_max, Xf / x_s + OPT_e)

x0_min = np.nan_to_num(x0_min, nan=-ca.inf)
x0_max = np.nan_to_num(x0_max, nan=ca.inf)
xf_min = np.nan_to_num(xf_min, nan=-ca.inf)
xf_max = np.nan_to_num(xf_max, nan=ca.inf)

gb = [Xk[:, 0], Xk[:, -1]]
lbg = np.concatenate([x0_min, xf_min])
ubg = np.concatenate([x0_max, xf_max])

# Collocation Constraints
gck = []
J = 0

for k in range(N):
    Z = ca.horzcat(Xk[:, k], Xkj[:, OPT_d * k : OPT_d * (k + 1)])
    dPi = ca.mtimes(Z, C)
    
    X_k_col = Xkj[:, OPT_d * k : OPT_d * (k + 1)]
    U_k_col = ca.repmat(Uk[:, k + 1], 1, OPT_d)
    Y_k_col = ca.repmat(Yk[:, k + 1], 1, OPT_d)
    
    if OPT_uinter == 'linear':
        U_k_col += ca.kron(duk[:, k], tau.reshape(1, -1))
        Y_k_col += ca.kron(dyk[:, k], tau.reshape(1, -1))
        
    dXkj, Qk = f_dyn(X_k_col, U_k_col, Y_k_col, pv_col[:, OPT_d * k : OPT_d * (k + 1)])
    
    Xk_end = ca.mtimes(Z, D)
    gck.append(ca.vec(dsk[k] * dXkj - dPi))
    gck.append(Xk_end - Xk[:, k + 1])
    
    J += ca.mtimes(Qk, B) * dsk[k] / J_s + ca.sumsqr(ru * Uk[:, k + 1]) + ca.sumsqr(rdu * duk[:, k]) + ca.sumsqr(rdu2 * duk2[:, k]) + ca.sumsqr(rdy * dyk[:, k]) + ca.sumsqr(rdy2 * dyk2[:, k])

# Path Constraints
ghk = h_eq(Xk, Uk, Yk, pv_knot)
ghk_flat = ca.vec(ghk)

# Rate of inputs constraints
Sfk = f_sf(Xk, pv_knot)
duk_t = duk / ca.repmat(Sfk[:-1].T, nu, 1)
gduk = ca.vec(duk_t)

# Define NLP problem
w = ca.vertcat(ca.vec(Xk), ca.vec(Uk), ca.vec(Yk), ca.vec(Xkj))

lbw = np.concatenate([np.tile(x_min, N + 1), np.tile(u_min, N + 1), np.tile(y_min, N + 1), np.tile(x_min, N * OPT_d)])
ubw = np.concatenate([np.tile(x_max, N + 1), np.tile(u_max, N + 1), np.tile(y_max, N + 1), np.tile(x_max, N * OPT_d)])
w0 = np.concatenate([x0.flatten('F'), u0.flatten('F'), y0.flatten('F'), xc0])

g = ca.vertcat(*gb, *gck, ghk_flat, gduk)

lbg_full = np.concatenate([lbg, np.zeros((OPT_d + 1) * N * nx), np.tile(h_lb, N + 1), np.tile(duk_lb, N)])
ubg_full = np.concatenate([ubg, np.zeros((OPT_d + 1) * N * nx), np.tile(h_ub, N + 1), np.tile(duk_ub, N)])

nlp = {'f': J, 'x': w, 'g': g}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

print("Starting Solver...")
transcription_time = time.time() - start_time

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg_full, ubg=ubg_full)

solution_time = time.time() - start_time - transcription_time

# Postprocessing - Extract variables
w_opt = sol['x'].full().flatten()

idx_x = nx * (N + 1)
idx_u = idx_x + nu * (N + 1)
idx_y = idx_u + ny * (N + 1)

x_opt = w_opt[:idx_x].reshape((nx, N + 1), order='F') * x_s[:, None]
u_opt = w_opt[idx_x:idx_u].reshape((nu, N + 1), order='F') * u_s[:, None]
y_opt = w_opt[idx_u:idx_y].reshape((ny, N + 1), order='F') * y_s[:, None]

# Calculate final time (sum of sf)
t_opt_array = f_sf(x_opt / x_s[:, None], pv_knot).full().flatten()
lap_time = np.sum(t_opt_array[:-1] * dsk)

print(f"\nObjective function value: {sol['f'].full()[0][0]}")
print(f"Lap time: {lap_time:.3f} s")
print(f"Solution Time: {solution_time:.2f} s")

# Save output to .mat for Plotly to render
output_data = {
    'x_opt': x_opt,
    'u_opt': u_opt,
    'y_opt': y_opt,
    's_knot': s_knot,
    'lap_time': lap_time
}

os.makedirs('Results', exist_ok=True)
sio.savemat('Results/solution.mat', output_data)
print("Results saved to Results/solution.mat. Ready for plotResults.py")