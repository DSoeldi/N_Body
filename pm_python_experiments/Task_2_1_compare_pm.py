## calculate the particle mesh and assign the resulting accelerations back to the particles. 
## do the same thing with the brute force calculation and compare the two different results. 
## try out different grid spacings and their influence on the accuracy of the pm calculation.
import pm_2d_functions as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Disk data (for choice 2)\data0.txt'
galaxy = pd.read_csv(path, sep = '\t', index_col=0)
galaxy_pm = galaxy.copy()
################################
data = "1"
N = 1000
Borders = (-1,1)
# stepsize_list = [0.001,0.01,0.1,1]
################################
Grid = pm.generateGrid(N,Borders)

stepsize = 0.1
# N_list = [5, 10, 100, 1000]
N_list = [2,4,6,8,10,12,14,16,18,20,30,60,100, 200, 500, 1000, 2000]

def calculate_pm_error(galaxy_pm, galaxy_bf):
    """
    Calculate single RMS error metric between PM and brute force.
    
    Parameters:
    -----------
    galaxy_pm : DataFrame
        Galaxy after PM leapfrog step
    galaxy_bf : DataFrame
        Galaxy after brute force leapfrog step
    
    Returns:
    --------
    rms_error : float
        Root mean square position error
    """
    r_pm = galaxy_pm[['x', 'y']].values
    r_bf = galaxy_bf[['x', 'y']].values
    
    # RMS position error
    rms_error = np.sqrt(np.mean(np.sum((r_pm - r_bf)**2, axis=1)))
    
    return rms_error

error_list = []
calculation_time_list = []

print("Calculating brute force reference (this is the same for all grid sizes)...")
galaxy_bf = galaxy.copy()
galaxy_bf = pm.leapfrog_integration_brute_force(galaxy_bf, stepsize, softening = 0.118)  # Don't pass N!
galaxy_pm = galaxy.copy()
galaxy_pm = pm.leapfrog_integration(galaxy_pm, Grid, stepsize, 2)
for N in N_list:

    print(f"\nTesting grid size N = {N}...")
    
    # Calculate PM with current grid size
    galaxy_pm = np.zeros_like(galaxy_pm)
    galaxy_pm = galaxy.copy()
    Grid = pm.generateGrid(N, Borders)
    start = time()
    galaxy_pm = pm.leapfrog_integration(galaxy_pm, Grid, stepsize, N)

    # Compare to the fixed brute force reference
    error = calculate_pm_error(galaxy_pm, galaxy_bf)
    print(f"  RMS error for N={N}: {error:.6e}")
    error_list.append(error)
    calculation_time_list.append(time()-start)

name = "0_soft_0.118"
# Plot results Error
plt.figure(figsize=(7, 3))
plt.xlabel("Grid Size")
plt.loglog(N_list, error_list, 'o-', linewidth=1, markersize=2, label='Measured Error')
plt.ylabel("Root Mean Square Error / $pc$")
plt.tight_layout()
plt.grid(True, alpha=0.3, which='both')
plt.savefig(f'g{name}_gridsize_error_comparison_FIXED.png', dpi=200)
# plt.show()


# # Plot results Calculation Time
# plt.figure(figsize=(7, 4))
# plt.loglog(N_list, calculation_time_list, 'o-', linewidth=1, markersize=2, label='Measured Error')
# plt.xlabel("Grid Size")
# plt.ylabel("Calculation Time / $s$")
# plt.grid(True, alpha=0.3, which='both')
# plt.savefig(f'g{name}_gridsize_compTime_comparison_FIXED.png', dpi=200)
# # plt.show()


