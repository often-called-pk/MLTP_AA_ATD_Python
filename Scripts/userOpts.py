# userOpts.py - User Options 

# Define user options for the optimisation problem and collocation method

# - Select aerodynamic configuration and torque distribution configuration

# - Select track data: it can be loaded from a matlab file containing at least two arrays of data for the distance 's' and the curvature 'k'. 
#   Optionally, it can also contain cartesian coordinates 'x' and 'y' 
#   (these are used for the plots of the track in the postprocessing, but the information for the optimisation problem is still taken from 's' and 'k'). 
#   Alternatively, the 's' and 'k' arrays can be defined here directly (three examples are provided).
#   Note that when using track data, raw data should be checked and preprocessed as required to avoid noisy signals, which can be detrimental for the collocation method.

# - Optimisation problem and collocation:
#   - Set initial and final conditions: set as nan to let the solver find the optimal value. (Use carefully since not every combination of boundary conditions is feasable).
#   - Collocation options
#   - Solver options
#   - Limits for the rate of inputs
#   - Regularisation factors

import numpy as np
import scipy.io as sio
import math
import sys
import os

# Adjust path to import Parameters
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Parameters')))
from Powertrain import pt
from vehParams import vp

## Selection of aerodynamic and torque distribution configuration:
# Options:
# - 'Static':   fixed front and rear wing angle
# - 'ActiveRW': fixed front wing angle and active rear wing control
# - 'Active':   active front and rear wing control
# - 'AALB':     active asymmetrical front and rear wing control (split front wing & tilt rear wing)

AeroConfig = 'AALB'  # select aerodynamic configuration 'Static', 'ActiveRW', 'Active', or 'AALB'
ATD = 'On'           # select active torque distribution 'On' or 'Off'

if AeroConfig == 'Static':         # static aero configuration -> front wing & rear wing AoA's determined in vehParams
    vp.ActAero = 0
elif AeroConfig == 'Active_RW':    # active aero configuration -> active rear wing AoA | static front wing AoA determined in vehParams
    vp.ActAero = 1
elif AeroConfig == 'Active':       # active aero configuration -> active symmetric front and rear wing AoA
    vp.ActAero = 2
elif AeroConfig == 'AALB':         # active aerodynamic load balancing configuration -> front wing AoA's & rear wing AoA/Tilt actively controlled
    vp.ActAero = 3

if ATD == 'Off':                   # active torque distribution OFF (4WD)
    pt.ATD = 0
elif ATD == 'On':                  # active torque distribution ON
    pt.ATD = 1

## Load powertrain and vehicle parameters
# Handled via imports above.

## Selection of track:
circuit = 'BCN'
track = {}

def simpleMA(data, window, offset):
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='same')

# -A) Self defined virtual tracks
if circuit == 'Hairpin':
    track['k'] = np.concatenate([np.zeros(75), 0.0975 * np.ones(16), np.zeros(73)]) # Hairpin
    track['k'] = simpleMA(track['k'], 10, 2) # Smoothen curvature signal
    track['s'] = np.linspace(0, 2 * len(track['k']), len(track['k'])) # (assume each element of the curvature array is spaced by 2m)
elif circuit == 'Straight':
    track['k'] = np.zeros(300) # Straight line
    track['k'] = simpleMA(track['k'], 10, 2) # Smoothen curvature signal
    track['s'] = np.linspace(0, 2 * len(track['k']), len(track['k'])) # (assume each element of the curvature array is spaced by 2m)
elif circuit == 'Sturn':
    track['k'] = np.concatenate([np.zeros(100), (math.pi/45) * np.ones(20), np.zeros(30), -(math.pi/45) * np.ones(20), np.zeros(100)]) # S-turn
    track['k'] = simpleMA(track['k'], 10, 2) # Smoothen curvature signal
    track['s'] = np.linspace(0, 2 * len(track['k']), len(track['k'])) # (assume each element of the curvature array is spaced by 2m)
elif circuit == 'VirtualTrack':
    track['k'] = np.concatenate([np.zeros(150), (math.pi/25) * np.ones(20), np.zeros(50), -(math.pi/125) * np.ones(90), -(math.pi/50) * np.ones(35), (math.pi/50) * np.ones(35), -(math.pi/500) * np.arange(1, 6.05, 0.05), np.zeros(200), -(math.pi/20) * np.ones(20), np.zeros(80)]) # Virtual track
    track['k'] = simpleMA(track['k'], 10, 2) # Smoothen curvature signal
    track['s'] = np.linspace(0, 2 * len(track['k']), len(track['k'])) # (assume each element of the curvature array is spaced by 2m)
# -B) Real circuits data 
elif circuit == 'BCN':
    mat_data = sio.loadmat('../Circuits/Barcelona_circuit.mat', squeeze_me=True)
    track.update(mat_data)
elif circuit == 'BCN_S1':
    mat_data = sio.loadmat('../Circuits/Barcelona_circuit_s1.mat', squeeze_me=True)
    track.update(mat_data)
elif circuit == 'BCN_S2':
    mat_data = sio.loadmat('../Circuits/Barcelona_circuit_s2.mat', squeeze_me=True)
    track.update(mat_data)
elif circuit == 'BCN_S3':
    mat_data = sio.loadmat('../Circuits/Barcelona_circuit_s3.mat', squeeze_me=True)
    track.update(mat_data)
elif circuit == 'Jarama':
    mat_data = sio.loadmat('../Circuits/Jarama_circuit.mat', squeeze_me=True)
    track.update(mat_data)
elif circuit == 'Spa':
    mat_data = sio.loadmat('../Circuits/Spa_circuit.mat', squeeze_me=True)
    track.update(mat_data)

## User options

# Optimal Control Problem - Boundary constraints
# -initial   ( x = [vx vy r n eps Om_fl Om_fr Om_rl Om_rr] )
vi = 61.7428 # inital velocity [m/s]

# initial velocities AWD [static, activeRW, active, AALB]: [60.4664  60.7631  60.7726  60.8071]
# initial velocities ATD [static, activeRW, active, AALB]: [60.5942  61.5     61.62    61.7428]
# initial velocities TyreOpt AWD [static, active, AALB]: [60.936  61.1497  61.227]
# initial velocities TyreOpt ATD [static, active, AALB]: [61.695  61.695   61.94]

Xi = np.array([vi, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
Xi_init = np.array([vi, 0, np.nan, np.nan, np.nan, np.nan, np.nan])

# -final
vf = np.nan # final velocity [m/s]
Xf = np.array([vf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
Xf_init = np.array([vf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

# Collocation
OPT_ds = 10         # collocation step (m)
OPT_d = 3           # degree of interpolating polynomials
OPT_uinter = 'linear' # assume 'linear' or 'constant' inputs
OPT_e = 1e-2        # Small quantity. Used as the slack for the path constraints and wherever helpful (like setting the initial guesses of the solver)

# Solver options
opts = {
    'ipopt': {
        'max_iter': 5000, # max number of iterations.
        'fixed_variable_treatment': 'make_constraint', # with default options, multipliers for the decision variables are wrong for equality constraints. Change the Tfixed_variable_treatmentT to hmake_constrainte or  relax_boundsf to obtain correct results (comment from Casadi's documentation. No difference has been observed in the solution or the solver when disabling this option)
        'tol': 1e-6, # accuracy to which the NLP is solver                                       (1e-8 standard)                              
        'acceptable_tol': 1e-5, # acceptable tolerance when tolerance can not be reached                    (1e-6 standard)
        'mu_init': 1e-1, # initial value of the barrier parameter                                    (1e-1 standard)
        'bound_push': 1e-2, # how much initial point is pushed into the feasible region                 (1e-2 standard)
        'bound_frac': 1e-2, # fraction of the variable bounds within which the initial point should lie (1e-2 standard)
        'constr_viol_tol': 1e-4, # tolerance for violation of constraints [inf_pr]                           (1e-4 standard)
        'dual_inf_tol': 1e-4, # tolerance for violation of optimality condition [inf_du]                  (1e-4 standard)
        'compl_inf_tol': 1e-4 # tolerance for complementary conditions                                    (1e-4 standard)
    }
}

class Struct(object): pass
c = Struct()
c.ub = Struct(); c.lb = Struct(); c.ru = Struct(); c.rdu = Struct(); c.rdu2 = Struct()

# -Constraints on the rate of inputs (units of each input per second) 
c.ub.T_motor = 2e4; c.lb.T_motor = -2e4       # upper & lower limits on rate of torque e-motor input  (Nm/s)
c.ub.T_brake = 1.45e5; c.lb.T_brake = -1.45e5 # upper & lower limits on rate of torque brake input    (Nm/s)
c.ub.ATD = 2e4; c.lb.ATD = -2e4               # upper & lower limits on rate of ATD input             (1/s)
c.ub.delta = 0.1; c.lb.delta = -0.1           # upper & lower limits on rate of steering angle input  (rad/s)
c.ub.FW = 20; c.lb.FW = -20                   # upper & lower limits on rate of FW angle input        (deg/s)
c.ub.RW = 60; c.lb.RW = -60                   # upper & lower limits on rate of RW angle input        (deg/s)
c.ub.TW = 24; c.lb.TW = -24                   # upper & lower limits on rate of TW angle input        (deg/s)

# Regularisation factors for the inputs (ru) and their first (rdu) and second (rdu2) derivatives
c.ru.T_motor = 0; c.rdu.T_motor = 0; c.rdu2.T_motor = 0      # torque e-motor input  (-)
c.ru.T_brake = 0; c.rdu.T_brake = 0; c.rdu2.T_brake = 0      # torque brake input    (-)
c.ru.delta = 0; c.rdu.delta = 10; c.rdu2.delta = 15          # steering angle input  (-)
c.ru.ATD = 0; c.rdu.ATD = 0; c.rdu2.ATD = 1.5                # ATD input             (-)
c.ru.FW = 0; c.rdu.FW = 0; c.rdu2.FW = 0.3                   # FW angle input        (-)
c.ru.RW = 0; c.rdu.RW = 0; c.rdu2.RW = 0.9                   # RW angle input        (-)
c.ru.TW = 0.005; c.rdu.TW = 0; c.rdu2.TW = 0.4               # TW angle input        (-)

# Regularisation factors for the states and aux variables                               
rdy  = np.array([0, 0]) # first derivative aux variables [ltx; lty_f; lty_r]
rdy2 = np.array([1, 0])

if pt.ATD == 0 and vp.ActAero == 0:
    duk_ub  = np.array([c.ub.T_motor, c.ub.T_brake, c.ub.delta])                                                  
    duk_lb  = np.array([c.lb.T_motor, c.lb.T_brake, c.lb.delta])    
    ru      = np.array([c.ru.T_motor, c.ru.T_brake, c.ru.delta]) 
    rdu     = np.array([c.rdu.T_motor, c.rdu.T_brake, c.rdu.delta]) 
    rdu2    = np.array([c.rdu2.T_motor, c.rdu2.T_brake, c.rdu2.delta]) 
elif pt.ATD == 0 and vp.ActAero == 1:
    duk_ub  = np.array([c.ub.T_motor, c.ub.T_brake, c.ub.RW, c.ub.delta])                                         
    duk_lb  = np.array([c.lb.T_motor, c.lb.T_brake, c.lb.RW, c.lb.delta])     
    ru      = np.array([c.ru.T_motor, c.ru.T_brake, c.ru.RW, c.ru.delta]) 
    rdu     = np.array([c.rdu.T_motor, c.rdu.T_brake, c.rdu.RW, c.rdu.delta]) 
    rdu2    = np.array([c.rdu2.T_motor, c.rdu2.T_brake, c.rdu2.RW, c.rdu2.delta]) 
elif pt.ATD == 0 and vp.ActAero == 2:
    duk_ub  = np.array([c.ub.T_motor, c.ub.T_brake, c.ub.FW, c.ub.RW, c.ub.delta])                                         
    duk_lb  = np.array([c.lb.T_motor, c.lb.T_brake, c.lb.FW, c.lb.RW, c.lb.delta])     
    ru      = np.array([c.ru.T_motor, c.ru.T_brake, c.ru.FW, c.ru.RW, c.ru.delta]) 
    rdu     = np.array([c.rdu.T_motor, c.rdu.T_brake, c.rdu.FW, c.rdu.RW, c.rdu.delta]) 
    rdu2    = np.array([c.rdu2.T_motor, c.rdu2.T_brake, c.rdu2.FW, c.rdu2.RW, c.rdu2.delta]) 
elif pt.ATD == 0 and vp.ActAero == 3:
    duk_ub  = np.array([c.ub.T_motor, c.ub.T_brake, c.ub.FW, c.ub.FW, c.ub.RW, c.ub.TW, c.ub.delta])              
    duk_lb  = np.array([c.lb.T_motor, c.lb.T_brake, c.lb.FW, c.lb.FW, c.lb.RW, c.lb.TW, c.lb.delta])    
    ru      = np.array([c.ru.T_motor, c.ru.T_brake, c.ru.FW, c.ru.FW, c.ru.RW, c.ru.TW, c.ru.delta])     
    rdu     = np.array([c.rdu.T_motor, c.rdu.T_brake, c.rdu.FW, c.rdu.FW, c.rdu.RW, c.rdu.TW, c.rdu.delta]) 
    rdu2    = np.array([c.rdu2.T_motor, c.rdu2.T_brake, c.rdu2.FW, c.rdu2.FW, c.rdu2.RW, c.rdu2.TW, c.rdu2.delta])
elif pt.ATD == 1 and vp.ActAero == 0:
    duk_ub  = np.array([c.ub.T_motor, c.ub.T_brake, c.ub.ATD, c.ub.ATD, c.ub.ATD, c.ub.ATD, c.ub.delta])                                                  
    duk_lb  = np.array([c.lb.T_motor, c.lb.T_brake, c.lb.ATD, c.lb.ATD, c.lb.ATD, c.lb.ATD, c.lb.delta])    
    ru      = np.array([c.ru.T_motor, c.ru.T_brake, c.ru.ATD, c.ru.ATD, c.ru.ATD, c.ru.ATD, c.ru.delta]) 
    rdu     = np.array([c.rdu.T_motor, c.rdu.T_brake, c.rdu.ATD, c.rdu.ATD, c.rdu.ATD, c.rdu.ATD, c.rdu.delta]) 
    rdu2    = np.array([c.rdu2.T_motor, c.rdu2.T_brake, c.rdu2.ATD, c.rdu2.ATD, c.rdu2.ATD, c.rdu2.ATD, c.rdu2.delta]) 
elif pt.ATD == 1 and vp.ActAero == 1:
    duk_ub  = np.array([c.ub.T_motor, c.ub.T_brake, c.ub.ATD, c.ub.ATD, c.ub.ATD, c.ub.ATD, c.ub.RW, c.ub.delta])                                                  
    duk_lb  = np.array([c.lb.T_motor, c.lb.T_brake, c.lb.ATD, c.lb.ATD, c.lb.ATD, c.lb.ATD, c.lb.RW, c.lb.delta])    
    ru      = np.array([c.ru.T_motor, c.ru.T_brake, c.ru.ATD, c.ru.ATD, c.ru.ATD, c.ru.ATD, c.ru.RW, c.ru.delta]) 
    rdu     = np.array([c.rdu.T_motor, c.rdu.T_brake, c.rdu.ATD, c.rdu.ATD, c.rdu.ATD, c.rdu.ATD, c.rdu.RW, c.rdu.delta]) 
    rdu2    = np.array([c.rdu2.T_motor, c.rdu2.T_brake, c.rdu2.ATD, c.rdu2.ATD, c.rdu2.ATD, c.rdu2.ATD, c.rdu2.RW, c.rdu2.delta])  
elif pt.ATD == 1 and vp.ActAero == 2:
    duk_ub  = np.array([c.ub.T_motor, c.ub.T_brake, c.ub.ATD, c.ub.ATD, c.ub.ATD, c.ub.ATD, c.ub.FW, c.ub.RW, c.ub.delta])                                                  
    duk_lb  = np.array([c.lb.T_motor, c.lb.T_brake, c.lb.ATD, c.lb.ATD, c.lb.ATD, c.lb.ATD, c.lb.FW, c.lb.RW, c.lb.delta])    
    ru      = np.array([c.ru.T_motor, c.ru.T_brake, c.ru.ATD, c.ru.ATD, c.ru.ATD, c.ru.ATD, c.ru.FW, c.ru.RW, c.ru.delta]) 
    rdu     = np.array([c.rdu.T_motor, c.rdu.T_brake, c.rdu.ATD, c.rdu.ATD, c.rdu.ATD, c.rdu.ATD, c.rdu.FW, c.rdu.RW, c.rdu.delta]) 
    rdu2    = np.array([c.rdu2.T_motor, c.rdu2.T_brake, c.rdu2.ATD, c.rdu2.ATD, c.rdu2.ATD, c.rdu2.ATD, c.rdu2.FW, c.rdu2.RW, c.rdu2.delta])
elif pt.ATD == 1 and vp.ActAero == 3:
    duk_ub  = np.array([c.ub.T_motor, c.ub.T_brake, c.ub.ATD, c.ub.ATD, c.ub.ATD, c.ub.ATD, c.ub.FW, c.ub.FW, c.ub.RW, c.ub.TW, c.ub.delta])                                                  
    duk_lb  = np.array([c.lb.T_motor, c.lb.T_brake, c.lb.ATD, c.lb.ATD, c.lb.ATD, c.lb.ATD, c.lb.FW, c.lb.FW, c.lb.RW, c.lb.TW, c.lb.delta])    
    ru      = np.array([c.ru.T_motor, c.ru.T_brake, c.ru.ATD, c.ru.ATD, c.ru.ATD, c.ru.ATD, c.ru.FW, c.ru.FW, c.ru.RW, c.ru.TW, c.ru.delta]) 
    rdu     = np.array([c.rdu.T_motor, c.rdu.T_brake, c.rdu.ATD, c.rdu.ATD, c.rdu.ATD, c.rdu.ATD, c.rdu.FW, c.rdu.FW, c.rdu.RW, c.rdu.TW, c.rdu.delta]) 
    rdu2    = np.array([c.rdu2.T_motor, c.rdu2.T_brake, c.rdu2.ATD, c.rdu2.ATD, c.rdu2.ATD, c.rdu2.ATD, c.rdu2.FW, c.rdu2.FW, c.rdu2.RW, c.rdu2.TW, c.rdu2.delta])   

# Constraints on the rate of inputs for the initalisation program (units of each input per second)  u = [Tdrive_n; T_brake_n; delta_n];
duk_ub_init = np.array([c.ub.T_motor, c.ub.T_brake, c.ub.delta])                                                                             
duk_lb_init = np.array([c.lb.T_motor, c.lb.T_brake, c.lb.delta])    

# Regularisation for the initalisation program
ru_init   = np.array([c.ru.T_motor, c.ru.T_brake, c.ru.delta])
rdu_init  = np.array([c.rdu.T_motor, c.rdu.T_brake, c.rdu.delta])
rdu2_init = np.array([c.rdu2.T_motor, c.rdu2.T_brake, c.rdu2.delta])
rdy_init  = rdy[0]
rdy2_init = rdy2[0]