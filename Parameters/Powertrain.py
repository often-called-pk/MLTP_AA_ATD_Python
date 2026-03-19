import math

class PowertrainParams:
    def __init__(self):
        # 2* E-motor Pmax continuous: 228kW, Tmax continuous 301kW, Wmax = 17750 rpm -> SPM242-176 helix
        self.Pmax  = 2 * 228 * 1000          # max continuous power e-motor (W)
        self.Tmax  = 2 * 301                 # max continuous torque e-motor (Nm)
        self.OMmax = 17750 * (math.pi / 30)  # max angular velocity e-motor (rad/s)
        self.Vmax  = 290 / 3.6               # required top speed - used for gear selection (m/s)
        self.eff   = 0.9                     # efficiency e-motor (-)
        self.ATD   = 0                       # Active Torque Distribution toggle (default off)

class VehicleParams_Init:
    def __init__(self, pt):
        self.Rw   = 0.3142857                # wheel radius (m)
        self.gear = (pt.OMmax * self.Rw) / pt.Vmax # final drive ratio based on top speed requirement (-)

pt = PowertrainParams()
vp_init = VehicleParams_Init(pt)