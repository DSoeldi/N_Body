## calculate the particle mesh and assign the resulting accelerations back to the particles. 
## do the same thing with the brute force calculation and compare the two different results. 
## try out different grid spacings and their influence on the accuracy of the pm calculation.
import pm_2d_functions as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def velocity_calc(galaxy):
    vx = galaxy.vx.to_numpy()
    vy = galaxy.vy.to_numpy()
    velocity = np.linalg.norm([vx,vy], axis = 0)
    return velocity

path = r'C:\Users\UZH\OneDrive - Universität Zürich UZH\Dokumente\HS25\Computational Astrophysics\N_Body_Repo\N_Body\Prerequisites\Disk data (for choice 2)\data0.txt'
galaxy = pd.read_csv(path, sep = '\t', index_col=0)
galaxy_pm = galaxy.copy()
################################
data = "0"
N = 1000
Borders = (-1,1)
stepsize_list = [0.001,0.01,0.1,1]
################################
Grid = pm.generateGrid(N,Borders)
fig, ax = plt.subplots(5,1, figsize = [8,10])

## compare the velocity when calculated with different time step sizes
ax[0].hist(velocity_calc(galaxy_pm), bins=100, label = "velocity before timestep")
ax[0].legend()
for i in range(len(stepsize_list)):
    galaxy_pm = galaxy.copy()
    stepsize = stepsize_list[i]
    galaxy_pm = pm.leapfrog_integration(galaxy_pm, Grid, stepsize, N)
    ax[i+1].hist(velocity_calc(galaxy_pm), bins=100, label = f"stepsize = {stepsize}")
    ax[i+1].legend()

ax[2].set_ylabel("counts")
title = f"galaxy_plots/compare_dt_data0.png"
plt.xlabel('Velocity')
plt.savefig(title)
plt.cla()


## compare the velocity when calculated with different gridsizes
fig, ax = plt.subplots(5,1, figsize = [8,10])

stepsize = 0.001
N_list = [10, 100, 200, 500]
galaxy_pm = galaxy.copy()
ax[0].hist(velocity_calc(galaxy_pm), bins=100, label = "velocity before timestep")
ax[0].legend()
for i in range(len(N_list)):
    Grid = pm.generateGrid(N,Borders)
    galaxy_pm = galaxy.copy()
    N = N_list[i]
    galaxy_pm = pm.leapfrog_integration(galaxy_pm, Grid, stepsize, N)
    ax[i+1].hist(velocity_calc(galaxy_pm), bins=100, label = f"Gridsize = {N}")
    ax[i+1].legend()
ax[2].set_ylabel("counts")
title = f"galaxy_plots/compare_N_data0.png"
plt.xlabel('Velocity')
plt.savefig(title)
plt.cla()


################# comparison of particle mesh and direct summation
# galaxy_bf = galaxy_pm.copy()
# galaxy_bf = pm.leapfrog_integration_brute_force(galaxy_bf, stepsize, N)