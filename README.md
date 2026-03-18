Python MLTP Framework: README
Description
The Minimum Lap Time Problem (MLTP) framework calculates the absolute fastest trajectory a specific vehicle can take around a given track. Rather than using simulation to guess and check lap times, it uses optimal control theory. It looks at the entire track at once and calculates the perfect racing line, steering inputs, throttle, braking, and active aerodynamic settings simultaneously.

How It Works
The framework treats the lap as a massive optimization problem built on direct collocation.

Track Discretization: The program divides the track into hundreds of small segments based on distance, not time.

Vehicle Model: The car is modeled as a 9-state system (tracking speed, track position, yaw, and individual wheel rotation). At each track segment, the physics engine calculates tire grip, downforce, and load transfer.

The Solver (IPOPT): The solver is given the task of minimizing the total time it takes to cover the track distance. It adjusts the control inputs (steering, pedals, wing angles) at every segment. If an adjustment causes a tire to exceed its physical grip limit (calculated via the Pacejka Magic Formula), the solver rejects it.

Convergence: The solver iterates until it finds the absolute limit of the vehicle's physical capabilities along the perfect path.

System Requirements
Python 3.8+

casadi (Version 3.5.5 or later)

numpy

scipy

plotly

Framework Structure
Aerodynamics/: Contains baseline aerodynamic maps derived from CFD.

Circuits/: Track data files (.mat).

Parameters/:

Powertrain.py: Motor and drivetrain specifications.

vehParams.py: Vehicle mass, dimensions, and tire characteristics.

Scripts/:

MLTP.py: The core solver script.

vehModel.py: Defines the symbolic equations of motion and constraints.

userOpts.py: Configuration file for selecting tracks, aero modes, and solver limits.

plotResults.py: Generates interactive Plotly graphs and exports the .mat solution.

Getting Started
Install dependencies via pip install casadi numpy scipy plotly.

Define your track and vehicle configuration in Scripts/userOpts.py.

Run Scripts/MLTP.py to calculate the optimal lap.

The results will automatically render in your browser via Plotly and save as a .mat file in your data directory.
