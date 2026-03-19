import math
import scipy.io as sio
from Powertrain import pt, vp_init

class TyreParams:
    def __init__(self):
        self.mu  = 1.41   # friction coefficient (-)
        self.pD2 = -0.6   # variation of friction with load (mu sensitivity)
        self.bx  = 6      # stiffness factor (longitudinal)
        self.cx  = 2.3    # shape factor (longitudinal)
        self.ex  = 0.9    # curvature factor (longitudinal)
        self.by  = 6      # stiffness factor (lateral)
        self.cy  = 2.5    # shape factor (lateral)
        self.ey  = 0.5    # curvature factor (lateral)

class VehicleParams:
    def __init__(self):
        self.Rw = vp_init.Rw        # wheel radius (m)
        self.gear = vp_init.gear    # final drive ratio based on top speed requirement (-)
        self.ActAero = 0

        # Vehicle parameter inputs
        self.brkB  = 0.6766  # fraction of total brake force going to the front wheels (-)
        self.Tdist = 0.7281  # fraction of total torque going to the rear wheels (-)
        self.ksD   = 0.4620  # fraction of total roll stiffness for the rear axle (-)

        # Aerodynamic input
        self.alpha_FL = 10   # left front wing angle of attack [0 10] (deg)
        self.alpha_FR = 10   # right front wing angle of attack [0 10] (deg)
        self.alpha_RW = 8    # rear wing angle of attack [0 30] (deg)
        self.alpha_TW = 0    # rear wing tilt angle [-12 12] (deg)

        # Constants
        self.g   = 9.81      # gravitational acceleration (m/s^2)
        self.rho = 1.204     # air density (kg/m^3)

        # Mass Parameters
        self.mb  = 1300      # vehicle sprung mass (kg)
        self.md  = 75        # driver mass (kg)
        self.ms  = self.mb + self.md
        self.muf = 90        # unsprung mass front axle (kg)
        self.mur = 100       # unsprung mass rear axle (kg)
        self.m   = self.ms + self.muf + self.mur
        self.I_z = 1450      # yaw moment of inertia (kg*m^2)

        # Dimensions
        self.A   = 2.1596                   # vehicle reference area (m^2)
        self.t   = 1.5204145                # track width (m)
        self.l   = 2.786                    # wheelbase (m)
        self.wB  = 0.5                      # COG distribution front (-)
        self.l_f = self.l * (1 - self.wB)   # distance COG to front axle (m)
        self.l_r = self.l * self.wB         # distance COG to rear axle (m)
        self.hcg = 0.5                      # COG height above ground plane (m)
        self.huf = 0.2968771                # height COG unsprung mass front (m)
        self.hur = 0.2968771                # height COG unsprung mass rear (m)
        self.hw  = 1.28                     # height of rear wing (m)

        # Roll Centers and Stiffness
        self.hRCf = 0.07                                                        # height roll centre front (m)
        self.hRCr = 0.11                                                        # height roll centre rear (m)
        self.hRC  = (self.l_f * self.hRCr + self.l_r * self.hRCf) / self.l      # height roll centre axis at centre of gravity (m)
        self.d    = self.hcg - self.hRC                                         # distance between COG and roll axis (m)
        self.ksf  = (1 - self.ksD) * 2000                                       # roll stiffness front axle (Nm/deg)
        self.ksr  = self.ksD * 2000                                             # roll stiffness rear axle (Nm/deg)
        self.ks   = self.ksf + self.ksr                                         # total roll stiffness (Nm/deg)
        self.ksf_rad = self.ksf * (180 / math.pi)                               # roll stiffness front axle (Nm/rad)
        self.ksr_rad = self.ksr * (180 / math.pi)                               # roll stiffness rear axle (Nm/rad)
        self.ks_rad  = self.ksf_rad + self.ksr_rad                              # total roll stiffness (Nm/rad)

        # Tyre Parameters Setup
        self.Jw = 0.9        # wheel rotational inertia (kg*m^2)
        self.f  = 0.01       # rolling resistance coefficient (-)
        self.Fz0 = 4905      # nominal vertical wheel load (N)
        self.Fz0_shift = 1   # fraction shift in nominal wheel load
        self.tyre = TyreParams()

        # Static wheel loads
        self.Wfl0 = 0.5 * self.g * (self.muf + self.wB * self.ms)
        self.Wfr0 = 0.5 * self.g * (self.muf + self.wB * self.ms)
        self.Wrl0 = 0.5 * self.g * (self.mur + (1 - self.wB) * self.ms)
        self.Wrr0 = 0.5 * self.g * (self.mur + (1 - self.wB) * self.ms)

        # Brakes
        self.Tbrake_max = 4e3

        # Aerodynamics (Loaded from MAT file)
        try:
            aero_mat = sio.loadmat('../Aerodynamics/DATA_AA.mat', squeeze_me=True, struct_as_record=False)
            aero_veh = aero_mat['aero'].veh
            self.Cl0_right = aero_veh.Cl0_right
            self.Cl0_left  = aero_veh.Cl0_left
            self.Cl0_front = aero_veh.Cl0_front
            self.Cl0_rear  = aero_veh.Cl0_rear
            self.Cd0       = aero_veh.Cd0
            self.Cs0_front = aero_veh.Cs0_front
            self.Cs0_rear  = aero_veh.Cs0_rear
        except FileNotFoundError:
            print("Warning: DATA_AA.mat not found. Using default zeros for baseline aero.")
            self.Cl0_right = self.Cl0_left = self.Cl0_front = self.Cl0_rear = 0
            self.Cd0 = self.Cs0_front = self.Cs0_rear = 0

        self.Cl = self.Cl0_front + self.Cl0_rear
        self.Cd = self.Cd0

        # Lateral load transfer distribution
        self.lty_dis_f = (self.huf / self.t) * (self.muf / self.m) + ((self.l_r * self.hRCf) / (self.l * self.t)) * (self.ms / self.m) + ((self.ksf_rad * (self.d / (self.ks_rad - self.ms * self.g * self.d))) / self.t) * (self.ms / self.m) # fraction of load transfer reacted at front axle
        self.lty_dis_r = (self.hur / self.t) * (self.mur / self.m) + ((self.l_f * self.hRCr) / (self.l * self.t)) * (self.ms / self.m) + ((self.ksr_rad * (self.d / (self.ks_rad - self.ms * self.g * self.d))) / self.t) * (self.ms / self.m) # fraction of load transfer reacted at rear axle

        self.lty_dis = self.lty_dis_r / (self.lty_dis_f + self.lty_dis_r)

vp = VehicleParams()