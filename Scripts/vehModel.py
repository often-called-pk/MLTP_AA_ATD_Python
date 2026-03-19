# vehModel.py - Vehicle Model 
#
# Define vehicle model symbolic variables and equations using Casadi's 
# symbolic variables.

import casadi as ca
import numpy as np
import scipy.io as sio
import sys
import os
import math

# Adjust path to import Parameters and user options
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Parameters')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from Powertrain import pt
from vehParams import vp
from userOpts import OPT_e, duk_ub, duk_lb

# Import polynomial coefficients for aerodynamic functions
try:
    aero_data = sio.loadmat('../Aerodynamics/DATA_AA.mat', squeeze_me=True, struct_as_record=False)
    aero = aero_data['aero']
except FileNotFoundError:
    print("Warning: DATA_AA.mat not found. Aerodynamic coefficients will fail if not defined.")

class Struct:
    pass

if not hasattr(vp, 'FW_L'): vp.FW_L = Struct()
if not hasattr(vp, 'FW_R'): vp.FW_R = Struct()
if not hasattr(vp, 'RW'): vp.RW = Struct()
if not hasattr(vp, 'TW'): vp.TW = Struct()

## Vehicle model (A) - state variables
nx = 9 # number of state variables

# longitudinal velocity [m/s]
vx_n = ca.SX.sym('vx_n')
vx_s = 100
vx   = vx_s * vx_n

# lateral velocity [m/s]
vy_n = ca.SX.sym('vy_n')
vy_s = 10
vy   = vy_s * vy_n

# yaw rate [rad/s]
r_n = ca.SX.sym('yawrate_n')
r_s = 1
r   = r_s * r_n

# lateral distance to centreline [m] (left of centreline => n > 0; right => n < 0)
n_n = ca.SX.sym('n_n')
n_s = 5
n   = n_s * n_n

# angle to centreline tangent direction [rad]
eps_n = ca.SX.sym('eps_n')
eps_s = 1
eps   = eps_s * eps_n

# angular velocity front left tyre [rad/s]
Om_fl_n = ca.SX.sym('Om_fl_n')
Om_fl_s = vx_s / vp.Rw
Om_fl   = Om_fl_s * Om_fl_n

# angular velocity front right tyre [rad/s]
Om_fr_n = ca.SX.sym('Om_fr_n')
Om_fr_s = vx_s / vp.Rw
Om_fr   = Om_fr_s * Om_fr_n

# angular velocity rear left tyre [rad/s]
Om_rl_n = ca.SX.sym('Om_rl_n')
Om_rl_s = vx_s / vp.Rw
Om_rl   = Om_rl_s * Om_rl_n

# angular velocity rear right tyre [rad/s]
Om_rr_n = ca.SX.sym('Om_rr_n')
Om_rr_s = vx_s / vp.Rw
Om_rr   = Om_rr_s * Om_rr_n

## -State limits
vx_lim = (1 / vx_s) * np.array([OPT_e, pt.Vmax])
vy_lim = (1 / vy_s) * np.array([-10, 10])
r_lim  = (1 / r_s) * np.array([-math.pi/2, math.pi/2])
n_lim = (1 / n_s) * np.array([-4, 4]) # Constant track limits
eps_lim = (1 / eps_s) * np.array([-math.pi/4, math.pi/4])
Om_fl_lim = (1 / Om_fl_s) * np.array([OPT_e / vp.Rw, pt.Vmax / vp.Rw])
Om_fr_lim = (1 / Om_fr_s) * np.array([OPT_e / vp.Rw, pt.Vmax / vp.Rw])
Om_rl_lim = (1 / Om_rl_s) * np.array([OPT_e / vp.Rw, pt.Vmax / vp.Rw])
Om_rr_lim = (1 / Om_rr_s) * np.array([OPT_e / vp.Rw, pt.Vmax / vp.Rw])

# scaling factors
x_s = np.array([vx_s, vy_s, r_s, n_s, eps_s, Om_fl_s, Om_fr_s, Om_rl_s, Om_rr_s])

# states vector (scaled)
x = ca.vertcat(vx_n, vy_n, r_n, n_n, eps_n, Om_fl_n, Om_fr_n, Om_rl_n, Om_rr_n)

# limits
x_lim = np.vstack([vx_lim, vy_lim, r_lim, n_lim, eps_lim, Om_fl_lim, Om_fr_lim, Om_rl_lim, Om_rr_lim])
x_min = x_lim[:, 0]
x_max = x_lim[:, 1]

# Check number of states
if nx != x.shape[0]:
    raise ValueError('Number of states is not consistent')


## Vehicle model (A) - control variables (inputs)

# number of control variables
if pt.ATD == 0 and vp.ActAero == 0:
    nu = 3
elif pt.ATD == 0 and vp.ActAero == 1:
    nu = 4
elif pt.ATD == 0 and vp.ActAero == 2:
    nu = 5
elif pt.ATD == 0 and vp.ActAero == 3:
    nu = 7
elif pt.ATD == 1 and vp.ActAero == 0:
    nu = 7
elif pt.ATD == 1 and vp.ActAero == 1:
    nu = 8
elif pt.ATD == 1 and vp.ActAero == 2:
    nu = 9
elif pt.ATD == 1 and vp.ActAero == 3:
    nu = 11

# driving torque motor (Nm)
T_motor_n = ca.SX.sym('T_motor_n')
T_motor_s = pt.Tmax
T_motor = T_motor_s * T_motor_n
T_motor_lim = (1 / T_motor_s) * np.array([0, pt.Tmax])

# braking torque (Nm)
T_brake_n = ca.SX.sym('T_brake_n')
T_brake_s = vp.Tbrake_max
T_brake = T_brake_s * T_brake_n
T_brake_lim = (1 / T_brake_s) * np.array([-vp.Tbrake_max, 0])

# tyre steer angle (rad)
delta_n = ca.SX.sym('delta_n')
delta_s = math.pi / 8
delta = delta_s * delta_n
delta_lim = (1 / delta_s) * np.array([-math.pi/4, math.pi/4])

u_list = [T_motor_n, T_brake_n]
u_s_list = [T_motor_s, T_brake_s]
u_lim_list = [T_motor_lim, T_brake_lim]

# torque distribution
if pt.ATD == 1:
    ATD_FL_n = ca.SX.sym('ATD_FL_n')
    ATD_FL_s = 1
    ATD_FL = ATD_FL_s * ATD_FL_n
    ATD_FL_lim = (1 / ATD_FL_s) * np.array([0, 1])

    ATD_FR_n = ca.SX.sym('ATD_FR_n')
    ATD_FR_s = 1
    ATD_FR = ATD_FR_s * ATD_FR_n
    ATD_FR_lim = (1 / ATD_FR_s) * np.array([0, 1])

    ATD_RL_n = ca.SX.sym('ATD_RL_n')
    ATD_RL_s = 1
    ATD_RL = ATD_RL_s * ATD_RL_n
    ATD_RL_lim = (1 / ATD_RL_s) * np.array([0, 1])

    ATD_RR_n = ca.SX.sym('ATD_RR_n')
    ATD_RR_s = 1
    ATD_RR = ATD_RR_s * ATD_RR_n
    ATD_RR_lim = (1 / ATD_RR_s) * np.array([0, 1])
    
    u_list.extend([ATD_FL_n, ATD_FR_n, ATD_RL_n, ATD_RR_n])
    u_s_list.extend([ATD_FL_s, ATD_FR_s, ATD_RL_s, ATD_RR_s])
    u_lim_list.extend([ATD_FL_lim, ATD_FR_lim, ATD_RL_lim, ATD_RR_lim])

# aerodynamics (deg)
if vp.ActAero == 1:
    activeAeroRW_n = ca.SX.sym('activeAeroRW_n')
    activeAeroRW_s = 30
    activeAeroRW = activeAeroRW_s * activeAeroRW_n
    activeAeroRW_lim = (1 / activeAeroRW_s) * np.array([0, 30])
    u_list.append(activeAeroRW_n)
    u_s_list.append(activeAeroRW_s)
    u_lim_list.append(activeAeroRW_lim)
elif vp.ActAero == 2:
    activeAeroFW_n = ca.SX.sym('activeAeroFW_n')
    activeAeroFW_s = 10
    activeAeroFW = activeAeroFW_s * activeAeroFW_n
    activeAeroFW_lim = (1 / activeAeroFW_s) * np.array([0, 10])

    activeAeroRW_n = ca.SX.sym('activeAeroRW_n')
    activeAeroRW_s = 30
    activeAeroRW = activeAeroRW_s * activeAeroRW_n
    activeAeroRW_lim = (1 / activeAeroRW_s) * np.array([0, 30])
    
    u_list.extend([activeAeroFW_n, activeAeroRW_n])
    u_s_list.extend([activeAeroFW_s, activeAeroRW_s])
    u_lim_list.extend([activeAeroFW_lim, activeAeroRW_lim])
elif vp.ActAero == 3:
    activeAeroFL_n = ca.SX.sym('activeAeroFL_n')
    activeAeroFL_s = 10
    activeAeroFL = activeAeroFL_s * activeAeroFL_n
    activeAeroFL_lim = (1 / activeAeroFL_s) * np.array([0, 10])

    activeAeroFR_n = ca.SX.sym('activeAeroFR_n')
    activeAeroFR_s = 10
    activeAeroFR = activeAeroFR_s * activeAeroFR_n
    activeAeroFR_lim = (1 / activeAeroFR_s) * np.array([0, 10])

    activeAeroRW_n = ca.SX.sym('activeAeroRW_n')
    activeAeroRW_s = 30
    activeAeroRW = activeAeroRW_s * activeAeroRW_n
    activeAeroRW_lim = (1 / activeAeroRW_s) * np.array([0, 30])

    activeAeroTW_n = ca.SX.sym('activeAeroTW_n')
    activeAeroTW_s = 12
    activeAeroTW = activeAeroTW_s * activeAeroTW_n
    activeAeroTW_lim = (1 / activeAeroTW_s) * np.array([-12, 12])
    
    u_list.extend([activeAeroFL_n, activeAeroFR_n, activeAeroRW_n, activeAeroTW_n])
    u_s_list.extend([activeAeroFL_s, activeAeroFR_s, activeAeroRW_s, activeAeroTW_s])
    u_lim_list.extend([activeAeroFL_lim, activeAeroFR_lim, activeAeroRW_lim, activeAeroTW_lim])

u_list.append(delta_n)
u_s_list.append(delta_s)
u_lim_list.append(delta_lim)

u_s = np.array(u_s_list)
u = ca.vertcat(*u_list)
u_lim = np.vstack(u_lim_list)

u_min = u_lim[:, 0]
u_max = u_lim[:, 1]

# Check number of inputs
if nu != u.shape[0]:
    raise ValueError('Number of inputs is not consistent')

# Constraints on the rate of inputs
duk_ub = duk_ub / u_s
duk_lb = duk_lb / u_s

## Vehicle model (A) - additional variables
ny = 2 # Number of aux variables

# longitudinal load transfer [N]
ltx_n = ca.SX.sym('loadTransferX_n')
ltx_s = vp.m * vp.g * vp.hcg / vp.l
ltx = ltx_s * ltx_n

# lateral load transfer [N]
lty_n = ca.SX.sym('loadTransferY_n')
lty_s = vp.m * vp.g * vp.hcg / vp.t
lty = lty_s * lty_n

# scaling factors
y_s = np.array([ltx_s, lty_s])

# aux. variables vector (scaled)
y = ca.vertcat(ltx_n, lty_n)

# aux. variables limits
ltx_lim = (1 / ltx_s) * np.array([-vp.m * vp.g, vp.m * vp.g])
lty_lim = (1 / lty_s) * np.array([-vp.m * vp.g, vp.m * vp.g])

y_lim = np.vstack([ltx_lim, lty_lim])

y_min = y_lim[:, 0]
y_max = y_lim[:, 1]

# Check number of aux variables
if ny != y.shape[0]:
    raise ValueError('Number of aux. variables is not consistent')

## Vechicle model (B) - variables
kappa = ca.SX.sym('kappa') # kappa > 0 for left turns
pv = ca.vertcat(kappa) # collect variable parameters

## Vehicle model (B) - equations
## -aerodynamic force coefficients

if vp.ActAero == 0:
    # front wing - left
    vp.FW_L.Cl_front = aero.FW_L.Cl_front * vp.alpha_FL   
    vp.FW_L.Cl_rear  = aero.FW_L.Cl_rear  * vp.alpha_FL   
    vp.FW_L.Cl_left  = vp.FW_L.Cl_front + vp.FW_L.Cl_rear

    # front wing - right
    vp.FW_R.Cl_front = aero.FW_R.Cl_front * vp.alpha_FR   
    vp.FW_R.Cl_rear  = aero.FW_R.Cl_rear  * vp.alpha_FR   
    vp.FW_R.Cl_right = vp.FW_R.Cl_front + vp.FW_R.Cl_rear

    # rear wing - angle of attack
    vp.RW.Cl_front  = aero.RW.Cl_front[0]*(vp.alpha_RW**3) + aero.RW.Cl_front[1]*(vp.alpha_RW**2) + aero.RW.Cl_front[2]*vp.alpha_RW
    vp.RW.Cl_rear   = aero.RW.Cl_rear[0]*(vp.alpha_RW**5) + aero.RW.Cl_rear[1]*(vp.alpha_RW**4) + aero.RW.Cl_rear[2]*(vp.alpha_RW**3) + aero.RW.Cl_rear[3]*(vp.alpha_RW**2) + aero.RW.Cl_rear[4]*vp.alpha_RW
    vp.RW.Cl_left   = 0.5*(vp.RW.Cl_front + vp.RW.Cl_rear)  
    vp.RW.Cl_right  = 0.5*(vp.RW.Cl_front + vp.RW.Cl_rear)
    vp.RW.Cd        = aero.RW.Cd[0]*(vp.alpha_RW**2) + aero.RW.Cd[1]*(vp.alpha_RW**1)

    # rear wing - tilt angle
    vp.TW.Cl_left   = aero.TW.Cl_left[0]  * (vp.alpha_TW**2) + aero.TW.Cl_left[1]  * (vp.alpha_TW**1)
    vp.TW.Cl_right  = aero.TW.Cl_right[0]  * (vp.alpha_TW**2) + aero.TW.Cl_right[1]  * (vp.alpha_TW**1)
    vp.TW.Cl_rear   = vp.TW.Cl_left + vp.TW.Cl_right
    vp.TW.Cs_rear    = aero.TW.Cs_rear  * vp.alpha_TW

    # combined aerodynamic coefficients
    vp.Cl_front = vp.Cl0_front  +  vp.FW_L.Cl_front  +  vp.FW_R.Cl_front +  vp.RW.Cl_front
    vp.Cl_rear  = vp.Cl0_rear   +  vp.FW_L.Cl_rear   +  vp.FW_R.Cl_rear  +  vp.RW.Cl_rear  +  vp.TW.Cl_rear
    vp.Cl_left  = vp.Cl0_left   +  vp.FW_L.Cl_left   +  vp.RW.Cl_left    +  vp.TW.Cl_left
    vp.Cl_right = vp.Cl0_right  +  vp.FW_R.Cl_right  +  vp.RW.Cl_right   +  vp.TW.Cl_right
    vp.Cd       = vp.Cd0        +  vp.RW.Cd
    vp.Cs_front = vp.Cs0_front
    vp.Cs_rear  = vp.Cs0_rear   +  vp.TW.Cs_rear
               
elif vp.ActAero == 1:
    vp.FW_L.Cl_front = aero.FW_L.Cl_front * vp.alpha_FL
    vp.FW_L.Cl_rear  = aero.FW_L.Cl_rear  * vp.alpha_FL
    vp.FW_L.Cl_left  = vp.FW_L.Cl_front + vp.FW_L.Cl_rear

    vp.FW_R.Cl_front = aero.FW_R.Cl_front * vp.alpha_FR
    vp.FW_R.Cl_rear  = aero.FW_R.Cl_rear  * vp.alpha_FR
    vp.FW_R.Cl_right = vp.FW_R.Cl_front + vp.FW_R.Cl_rear

    vp.RW.Cl_front = aero.RW.Cl_front[0]*(activeAeroRW**3) + aero.RW.Cl_front[1]*(activeAeroRW**2) + aero.RW.Cl_front[2]*activeAeroRW
    vp.RW.Cl_rear  = aero.RW.Cl_rear[0]*(activeAeroRW**5) + aero.RW.Cl_rear[1]*(activeAeroRW**4) + aero.RW.Cl_rear[2]*(activeAeroRW**3) + aero.RW.Cl_rear[3]*(activeAeroRW**2) + aero.RW.Cl_rear[4]*activeAeroRW
    vp.RW.Cl_left   = 0.5*(vp.RW.Cl_front + vp.RW.Cl_rear)
    vp.RW.Cl_right  = 0.5*(vp.RW.Cl_front + vp.RW.Cl_rear)
    vp.RW.Cd        = aero.RW.Cd[0]*(activeAeroRW**2) + aero.RW.Cd[1]*(activeAeroRW**1)

    vp.TW.Cl_left   = aero.TW.Cl_left[0]  * (vp.alpha_TW**2) + aero.TW.Cl_left[1]  * (vp.alpha_TW**1)
    vp.TW.Cl_right  = aero.TW.Cl_right[0]  * (vp.alpha_TW**2) + aero.TW.Cl_right[1]  * (vp.alpha_TW**1)
    vp.TW.Cl_rear   = vp.TW.Cl_left + vp.TW.Cl_right
    vp.TW.Cs_rear    = aero.TW.Cs_rear  * vp.alpha_TW

    vp.Cl_front = vp.Cl0_front  +  vp.FW_L.Cl_front  +  vp.FW_R.Cl_front +  vp.RW.Cl_front
    vp.Cl_rear  = vp.Cl0_rear   +  vp.FW_L.Cl_rear   +  vp.FW_R.Cl_rear  +  vp.RW.Cl_rear  +  vp.TW.Cl_rear
    vp.Cl_left  = vp.Cl0_left   +  vp.FW_L.Cl_left   +  vp.RW.Cl_left    +  vp.TW.Cl_left
    vp.Cl_right = vp.Cl0_right  +  vp.FW_R.Cl_right  +  vp.RW.Cl_right   +  vp.TW.Cl_right
    vp.Cd       = vp.Cd0        +  vp.RW.Cd
    vp.Cs_front = vp.Cs0_front
    vp.Cs_rear  = vp.Cs0_rear   +  vp.TW.Cs_rear

elif vp.ActAero == 2:
    vp.FW_L.Cl_front = aero.FW_L.Cl_front * activeAeroFW
    vp.FW_L.Cl_rear  = aero.FW_L.Cl_rear  * activeAeroFW
    vp.FW_L.Cl_left  = vp.FW_L.Cl_front + vp.FW_L.Cl_rear

    vp.FW_R.Cl_front = aero.FW_R.Cl_front * activeAeroFW
    vp.FW_R.Cl_rear  = aero.FW_R.Cl_rear  * activeAeroFW
    vp.FW_R.Cl_right = vp.FW_R.Cl_front + vp.FW_R.Cl_rear

    vp.RW.Cl_front = aero.RW.Cl_front[0]*(activeAeroRW**3) + aero.RW.Cl_front[1]*(activeAeroRW**2) + aero.RW.Cl_front[2]*activeAeroRW
    vp.RW.Cl_rear  = aero.RW.Cl_rear[0]*(activeAeroRW**5) + aero.RW.Cl_rear[1]*(activeAeroRW**4) + aero.RW.Cl_rear[2]*(activeAeroRW**3) + aero.RW.Cl_rear[3]*(activeAeroRW**2) + aero.RW.Cl_rear[4]*activeAeroRW
    vp.RW.Cl_left  = 0.5*(vp.RW.Cl_front + vp.RW.Cl_rear)
    vp.RW.Cl_right = 0.5*(vp.RW.Cl_front + vp.RW.Cl_rear)
    vp.RW.Cd       = aero.RW.Cd[0]*(activeAeroRW**2) + aero.RW.Cd[1]*(activeAeroRW**1)

    vp.TW.Cl_left   = aero.TW.Cl_left[0]  * (vp.alpha_TW**2) + aero.TW.Cl_left[1]  * (vp.alpha_TW**1)
    vp.TW.Cl_right  = aero.TW.Cl_right[0]  * (vp.alpha_TW**2) + aero.TW.Cl_right[1]  * (vp.alpha_TW**1)
    vp.TW.Cl_rear   = vp.TW.Cl_left + vp.TW.Cl_right
    vp.TW.Cs_rear    = aero.TW.Cs_rear  * vp.alpha_TW

    vp.Cl_front = vp.Cl0_front  +  vp.FW_L.Cl_front  +  vp.FW_R.Cl_front +  vp.RW.Cl_front
    vp.Cl_rear  = vp.Cl0_rear   +  vp.FW_L.Cl_rear   +  vp.FW_R.Cl_rear  +  vp.RW.Cl_rear  +  vp.TW.Cl_rear
    vp.Cl_left  = vp.Cl0_left   +  vp.FW_L.Cl_left   +  vp.RW.Cl_left    +  vp.TW.Cl_left
    vp.Cl_right = vp.Cl0_right  +  vp.FW_R.Cl_right  +  vp.RW.Cl_right   +  vp.TW.Cl_right
    vp.Cd       = vp.Cd0        +  vp.RW.Cd
    vp.Cs_front = vp.Cs0_front
    vp.Cs_rear  = vp.Cs0_rear   +  vp.TW.Cs_rear

elif vp.ActAero == 3:
    vp.FW_L.Cl_front = aero.FW_L.Cl_front * activeAeroFL
    vp.FW_L.Cl_rear  = aero.FW_L.Cl_rear  * activeAeroFL
    vp.FW_L.Cl_left  = vp.FW_L.Cl_front + vp.FW_L.Cl_rear

    vp.FW_R.Cl_front = aero.FW_R.Cl_front * activeAeroFR
    vp.FW_R.Cl_rear  = aero.FW_R.Cl_rear  * activeAeroFR
    vp.FW_R.Cl_right = vp.FW_R.Cl_front + vp.FW_R.Cl_rear

    vp.RW.Cl_front = aero.RW.Cl_front[0]*(activeAeroRW**3) + aero.RW.Cl_front[1]*(activeAeroRW**2) + aero.RW.Cl_front[2]*activeAeroRW
    vp.RW.Cl_rear  = aero.RW.Cl_rear[0]*(activeAeroRW**5) + aero.RW.Cl_rear[1]*(activeAeroRW**4) + aero.RW.Cl_rear[2]*(activeAeroRW**3) + aero.RW.Cl_rear[3]*(activeAeroRW**2) + aero.RW.Cl_rear[4]*activeAeroRW
    vp.RW.Cl_left   = 0.5*(vp.RW.Cl_front + vp.RW.Cl_rear)
    vp.RW.Cl_right  = 0.5*(vp.RW.Cl_front + vp.RW.Cl_rear)
    vp.RW.Cd        = aero.RW.Cd[0]*(activeAeroRW**2) + aero.RW.Cd[1]*(activeAeroRW**1)

    vp.TW.Cl_left   = aero.TW.Cl_left[0]  * (activeAeroTW**2) + aero.TW.Cl_left[1]  * (activeAeroTW**1)
    vp.TW.Cl_right  = aero.TW.Cl_right[0]  * (activeAeroTW**2) + aero.TW.Cl_right[1]  * (activeAeroTW**1)
    vp.TW.Cl_rear   = vp.TW.Cl_left + vp.TW.Cl_right
    vp.TW.Cs_rear    = aero.TW.Cs_rear  * activeAeroTW

    vp.Cl_front = vp.Cl0_front  +  vp.FW_L.Cl_front  +  vp.FW_R.Cl_front +  vp.RW.Cl_front
    vp.Cl_rear  = vp.Cl0_rear   +  vp.FW_L.Cl_rear   +  vp.FW_R.Cl_rear  +  vp.RW.Cl_rear  +  vp.TW.Cl_rear
    vp.Cl_left  = vp.Cl0_left   +  vp.FW_L.Cl_left   +  vp.RW.Cl_left    +  vp.TW.Cl_left
    vp.Cl_right = vp.Cl0_right  +  vp.FW_R.Cl_right  +  vp.RW.Cl_right   +  vp.TW.Cl_right
    vp.Cd       = vp.Cd0        +  vp.RW.Cd
    vp.Cs_front = vp.Cs0_front
    vp.Cs_rear  = vp.Cs0_rear   +  vp.TW.Cs_rear

vp.Cl       = vp.Cl_front + vp.Cl_rear       
vp.Cl_fl    = vp.Cl_front * (vp.Cl_left / vp.Cl)
vp.Cl_fr    = vp.Cl_front * (vp.Cl_right / vp.Cl)
vp.Cl_rl    = vp.Cl_rear * (vp.Cl_left / vp.Cl)
vp.Cl_rr    = vp.Cl_rear * (vp.Cl_right / vp.Cl)

## -aerodynamic forces [N]
# downforce: positive in downwards direction
f_lift_fl = -0.5 * vp.rho * vp.Cl_fl * vp.A * vx**2
f_lift_fr = -0.5 * vp.rho * vp.Cl_fr * vp.A * vx**2
f_lift_rl = -0.5 * vp.rho * vp.Cl_rl * vp.A * vx**2
f_lift_rr = -0.5 * vp.rho * vp.Cl_rr * vp.A * vx**2
f_lift = f_lift_fl + f_lift_fr + f_lift_rl + f_lift_rr 

# drag: positive along longitudinal axis
f_drag  = 0.5 * vp.rho * vp.A * vx**2 * vp.Cd
f_drag0 = 0.5 * vp.rho * vp.A * vx**2 * vp.Cd0                      # drag of vehicle body excluding rear wing contribution
f_dragRW  = 0.5 * vp.rho * vp.A * vx**2 * (vp.Cd - vp.Cd0)

# side force: orginally positive towards the right hand side of vehicle, now positive to left hand side
f_side_fl = -0.5 * (0.5 * vp.rho * vp.Cs_front * vp.A * vx**2)
f_side_fr = -0.5 * (0.5 * vp.rho * vp.Cs_front * vp.A * vx**2)
f_side_rl = -0.5 * (0.5 * vp.rho * vp.Cs_rear * vp.A * vx**2)
f_side_rr = -0.5 * (0.5 * vp.rho * vp.Cs_rear * vp.A * vx**2)
f_side = f_side_fl + f_side_fr + f_side_rl + f_side_rr 

## -fz, vertical tyre forces [N]  |  ltx: [Acceleration +; Brake -]  |  lty: [Left turn +; Right turn -]
fz_fl = vp.Wfl0 + f_lift_fl - ltx/2 - (1 - vp.lty_dis)*lty             
fz_fr = vp.Wfr0 + f_lift_fr - ltx/2 + (1 - vp.lty_dis)*lty                
fz_rl = vp.Wrl0 + f_lift_rl + ltx/2 - vp.lty_dis*lty
fz_rr = vp.Wrr0 + f_lift_rr + ltx/2 + vp.lty_dis*lty 

## - tyre slip angles [rad]
sa_fl = delta - ca.atan((vp.l_f*r + vy) / (vx - r*vp.t/2))
sa_fr = delta - ca.atan((vp.l_f*r + vy) / (vx + r*vp.t/2))
sa_rl = ca.atan((vp.l_r*r - vy) / (vx - r*vp.t/2))
sa_rr = ca.atan((vp.l_r*r - vy) / (vx + r*vp.t/2))

## - tyre slip ratio
v_fl = ca.sqrt((vy + vp.l_f*r)**2 + (vx - r*vp.t/2)**2)
v_fr = ca.sqrt((vy + vp.l_f*r)**2 + (vx + r*vp.t/2)**2)
v_rl = ca.sqrt((vy - vp.l_r*r)**2 + (vx - r*vp.t/2)**2)
v_rr = ca.sqrt((vy - vp.l_r*r)**2 + (vx + r*vp.t/2)**2)
v_flx = v_fl * ca.cos(sa_fl)
v_frx = v_fr * ca.cos(sa_fr)
v_rlx = v_rl * ca.cos(sa_rl)
v_rrx = v_rr * ca.cos(sa_rr)
sx_fl = (vp.Rw*Om_fl - v_flx) / v_flx
sx_fr = (vp.Rw*Om_fr - v_frx) / v_frx
sx_rl = (vp.Rw*Om_rl - v_rlx) / v_rlx
sx_rr = (vp.Rw*Om_rr - v_rrx) / v_rrx

## - tyre load dependent friction
mu_fl = vp.tyre.mu + vp.tyre.pD2 * (fz_fl - vp.Fz0) / vp.Fz0
mu_fr = vp.tyre.mu + vp.tyre.pD2 * (fz_fr - vp.Fz0) / vp.Fz0
mu_rl = vp.tyre.mu + vp.tyre.pD2 * (fz_rl - vp.Fz0) / vp.Fz0
mu_rr = vp.tyre.mu + vp.tyre.pD2 * (fz_rr - vp.Fz0) / vp.Fz0

## -fx, longitudinal tire forces  [N]
Dx_fl = mu_fl * (fz_fl / vp.Fz0_shift)
Dx_fr = mu_fr * (fz_fr / vp.Fz0_shift)
Dx_rl = mu_rl * (fz_rl / vp.Fz0_shift)
Dx_rr = mu_rr * (fz_rr / vp.Fz0_shift)

fx_fl = Dx_fl * ca.sin(vp.tyre.cx * ca.atan(vp.tyre.bx*sx_fl - vp.tyre.ex * (vp.tyre.bx*sx_fl - ca.atan(vp.tyre.bx*sx_fl))))
fx_fr = Dx_fr * ca.sin(vp.tyre.cx * ca.atan(vp.tyre.bx*sx_fr - vp.tyre.ex * (vp.tyre.bx*sx_fr - ca.atan(vp.tyre.bx*sx_fr))))
fx_rl = Dx_rl * ca.sin(vp.tyre.cx * ca.atan(vp.tyre.bx*sx_rl - vp.tyre.ex * (vp.tyre.bx*sx_rl - ca.atan(vp.tyre.bx*sx_rl))))
fx_rr = Dx_rr * ca.sin(vp.tyre.cx * ca.atan(vp.tyre.bx*sx_rr - vp.tyre.ex * (vp.tyre.bx*sx_rr - ca.atan(vp.tyre.bx*sx_rr))))

## -fy, lateral tyre forces [N]
Dy_fl = mu_fl * (fz_fl / vp.Fz0_shift)
Dy_fr = mu_fr * (fz_fr / vp.Fz0_shift)
Dy_rl = mu_rl * (fz_rl / vp.Fz0_shift)
Dy_rr = mu_rr * (fz_rr / vp.Fz0_shift)

fy_fl = Dy_fl * ca.sin(vp.tyre.cy * ca.atan(vp.tyre.by*sa_fl - vp.tyre.ey * (vp.tyre.by*sa_fl - ca.atan(vp.tyre.by*sa_fl))))
fy_fr = Dy_fr * ca.sin(vp.tyre.cy * ca.atan(vp.tyre.by*sa_fr - vp.tyre.ey * (vp.tyre.by*sa_fr - ca.atan(vp.tyre.by*sa_fr))))
fy_rl = Dy_rl * ca.sin(vp.tyre.cy * ca.atan(vp.tyre.by*sa_rl - vp.tyre.ey * (vp.tyre.by*sa_rl - ca.atan(vp.tyre.by*sa_rl))))
fy_rr = Dy_rr * ca.sin(vp.tyre.cy * ca.atan(vp.tyre.by*sa_rr - vp.tyre.ey * (vp.tyre.by*sa_rr - ca.atan(vp.tyre.by*sa_rr))))

# -rolling resistance [N]
f_roll = vp.f * (fz_fl + fz_fr + fz_rl + fz_rr)

# e-motor speed (rad/s)
Om_motor = (Om_fl + Om_fr)/4 * vp.gear + (Om_rl + Om_rr)/4 * vp.gear

# - individual wheel torque
if pt.ATD == 0:
    T_fl = 0.5 * T_motor * vp.gear * (1 - vp.Tdist) + T_brake * vp.brkB
    T_fr = 0.5 * T_motor * vp.gear * (1 - vp.Tdist) + T_brake * vp.brkB
    T_rl = 0.5 * T_motor * vp.gear * vp.Tdist + T_brake * (1 - vp.brkB)
    T_rr = 0.5 * T_motor * vp.gear * vp.Tdist + T_brake * (1 - vp.brkB)
elif pt.ATD == 1:
    T_fl = ATD_FL * T_motor * vp.gear + ATD_FL * T_brake * 2
    T_fr = ATD_FR * T_motor * vp.gear + ATD_FR * T_brake * 2
    T_rl = ATD_RL * T_motor * vp.gear + ATD_RL * T_brake * 2
    T_rr = ATD_RR * T_motor * vp.gear + ATD_RR * T_brake * 2

# power powertrain
P_motor = T_motor * Om_motor

# Change of independent variable
sf = (1 - n * kappa) / (vx * ca.cos(eps) - vy * ca.sin(eps))

## Vehicle model (B) - state derivatives
dvx    = (fx_rl + fx_rr + (fx_fl + fx_fr)*ca.cos(delta) - (fy_fl + fy_fr)*ca.sin(delta) + vp.m*vy*r - f_drag - f_roll) * sf / vp.m
dvy    = (fy_rl + fy_rr + (fy_fl + fy_fr)*ca.cos(delta) + (fx_fl + fx_fr)*ca.sin(delta) - vp.m*vx*r + f_side) * sf / vp.m
dr     = (-(fy_rl + fy_rr)*vp.l_r + (fx_rr - fx_rl)*vp.t/2 + ((fy_fl + fy_fr)*ca.cos(delta) + (fx_fl + fx_fr)*ca.sin(delta))*vp.l_f \
       + ((fy_fl - fy_fr)*ca.sin(delta))*vp.t/2 + ((fx_fr - fx_fl)*ca.cos(delta))*vp.t/2 - (f_side_rl + f_side_rr)*vp.l_r + (f_side_fl + f_side_fr)*vp.l_f) * sf / vp.I_z

dn     = (vx * ca.sin(eps) + vy * ca.cos(eps)) * sf
deps   = sf * r - kappa

dOm_fl = sf * (T_fl - fx_fl * vp.Rw) / vp.Jw
dOm_fr = sf * (T_fr - fx_fr * vp.Rw) / vp.Jw
dOm_rl = sf * (T_rl - fx_rl * vp.Rw) / vp.Jw
dOm_rr = sf * (T_rr - fx_rr * vp.Rw) / vp.Jw

# State derivatives vector (scaled to match the scaled states vector)
dx = ca.vertcat(dvx, dvy, dr, dn, deps, dOm_fl, dOm_fr, dOm_rl, dOm_rr) / x_s